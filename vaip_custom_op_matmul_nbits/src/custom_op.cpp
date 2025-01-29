/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

// must include cxx_api.hpp before custom_op.hpp otherise
// VAIP_ORT_API_VERSION is not defined we cannot use OrtAPI here.
#if defined(_WIN32)
#  include <intrin.h>
#else
#  include "ryzenai/dynamic_dispatch/utils/instruction_cache.hpp"
#  include "ryzenai/dynamic_dispatch/utils/instruction_registry.hpp"
#  include <ryzenai/dynamic_dispatch/xrt_context/xrt_context.hpp>
#  include <x86intrin.h>
#endif
#include <mmintrin.h>
#include <xmmintrin.h>

#include "onnxruntime_api.hpp"
#include <immintrin.h>

#include "./custom_op.hpp"
#include "vitis/ai/env_config.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <glog/logging.h>
#include <iostream>
#include <iterator>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>
#include <utility>
#include <vector>

#if defined(_WIN32)
#  pragma warning(disable : 4996)
#  pragma once

#endif
namespace fs = std::filesystem;
std::vector<int> grp_sizes_;
std::vector<std::vector<int>> n_sizes_;

#define OUT_TYPE int32_t
DEF_ENV_PARAM(ENABLE_MLADF, "1")
namespace vaip_matmul_nbits_custom_op {
template <typename T>
static void writeToFile(std::string filename, const T* data, size_t size) {

  std::ofstream file(filename, std::ios::binary);

  if (!file) {
    throw std::ios_base::failure("Failed to open file");
  }
  file.write((char*)(data), size * sizeof(T));

  if (!file) {
    throw std::ios_base::failure("Failed to write data to file");
  }
  file.close();
}
std::shared_ptr<void> MyCustomOp::gemm__ = nullptr;

MyCustomOp::MyCustomOp(std::shared_ptr<const PassContext> context,
                       const std::shared_ptr<MetaDefProto>& meta_def,
                       onnxruntime::Model* model)
    : CustomOpImp(context, meta_def, model) {
  // MatMulNBits node index
  cnt = stoi(meta_def->generic_param().at("cnt"));
  // Extracting the attribute information from ONNX model
  k_k = stoi(meta_def->generic_param().at("K"));
  k_n = stoi(meta_def->generic_param().at("N"));
  k_bits = stoi(meta_def->generic_param().at("bits"));
  k_block_size = stoi(meta_def->generic_param().at("block_size"));

  // Reading the constant weights and scales file
  std::string inputbin_wts = meta_def->generic_param().at("wts_file");
  std::string inputbin_scl = meta_def->generic_param().at("scl_file");

  std::string inputbin_zp;
  if (meta_def->generic_param().contains("zp_file")) {
    k_asymmetric = 1;
    inputbin_zp = meta_def->generic_param().at("zp_file");
  }

  // Get weights
  unsigned int size_wts = (unsigned int)fs::file_size(inputbin_wts);
  std::vector<int8_t> wts;
  wts.reserve(size_wts);
  auto iwts = std::ifstream(inputbin_wts, std::ios::in | std::ios::binary);
  iwts.read((char*)wts.data(), size_wts);

  // Get Scales
  unsigned int size_scl = (unsigned int)fs::file_size(inputbin_scl);
  std::vector<float> scl;
  scl.reserve(size_scl / sizeof(float));
  auto iscl = std::ifstream(inputbin_scl, std::ios::in | std::ios::binary);
  iscl.read((char*)scl.data(), size_scl);

  // Get Zero Point
  size_t kblks = k_k / k_block_size;
  int64_t zp_shape = (k_n * std::floor((float)((kblks + 1) * k_bits) / 8.0f));

  std::vector<int8_t> zero_pt;
  if (k_asymmetric) {
    unsigned int size_zp = (unsigned int)fs::file_size(inputbin_zp);
    zero_pt.reserve(size_zp);
    auto izp = std::ifstream(inputbin_zp, std::ios::in | std::ios::binary);
    izp.read((char*)(zero_pt.data()), size_zp);
  } else {
    // zero_pt.reserve(k_k * k_n / k_block_size);
    // memset(zero_pt.data(), 0, k_k * k_n / k_block_size);
    zero_pt.reserve(zp_shape * 2);
    memset(zero_pt.data(), 0, zp_shape * 2);
  }

  // Get Bias
  std::string bias_bin;
  std::vector<float> bias(k_n, 0); // fill with zeros
  if (meta_def->generic_param().contains("bias_file")) {
    bias_bin = meta_def->generic_param().at("bias_file");
    auto bfile = std::ifstream(bias_bin, std::ios::in | std::ios::binary);
    bfile.read((char*)bias.data(), k_n * sizeof(float));
  }

  // Ryzen-AI implementation

  // Re-arrage / expand weights / scales
  std::vector<int8_t> const_wts(k_k * k_n, 0);
  std::vector<float> const_scl(k_k * k_n / k_block_size);
  // fill this with zeros for Symmetric quantization
  // std::vector<int8_t> const_zps(k_k * k_n / k_block_size, 0);
  std::vector<int8_t> const_zps(zp_shape * 2, 0);

  // Original weights are in NxK/2 packed as uint8
  // Convert to KXN uint8
  for (int64_t i = 0; i < k_k; i += 2) {
    for (int64_t j = 0; j < k_n; j++) {
      auto srcv = wts[j * k_k / 2 + i / 2];
      auto src0 = (srcv & 0xf) - 8;
      auto src1 = ((srcv & 0xf0) >> 4) - 8;
      const_wts[i * k_n + j] = static_cast<int8_t>(src0);
      const_wts[(i + 1) * k_n + j] = static_cast<int8_t>(src1);
    }
  }

  // Original Scales are in Nx(K/BlockSize) shape
  // Convert to (K/BLOCK_SIZE)xN shape
  for (int i = 0; i < k_n; i++) {
    for (int j = 0; j < kblks; j++) {
      const_scl[j * k_n + i] = scl[i * kblks + j];
    }
  }

  // fill this with zeros for Symmetric quantization
  // Each row of zero points was padded to have an even length "kblks_pad"
  if (k_asymmetric) {
    int kblks_pad = 2 * zp_shape / k_n;
    for (int i = 0; i < k_n; i++) {
      for (int j = 0; j < kblks_pad; j = j + 2) {
        int8_t zpv;
        // zpv = zero_pt[(i * (kblks / 2)) + (j / 2)];
        zpv = zero_pt[(i * kblks_pad) / 2 + j / 2];
        const_zps[j * k_n + i] = (zpv & 0xf) - 8;
        const_zps[(j + 1) * k_n + i] = ((zpv & 0xf0) >> 4) - 8;
      }
    }
  }

  // Update N / Group size
  n_sizes_.push_back({k_k, k_n});
  grp_sizes_.push_back(k_block_size);

  init_op_mladf_dd(const_wts, const_zps, const_scl, bias);

#ifdef _WIN32
  // Input size for token phase
  input_data_ = (uint16_t*)_aligned_malloc(k_k * sizeof(uint16_t), 64);
#else
  input_data_ = (uint16_t*)aligned_alloc(64, k_k * sizeof(uint16_t));
#endif

  if (input_data_ == nullptr) {
    throw std::runtime_error("Unable to create memory for ryzenai-matmulnbits");
  }
}

MyCustomOp::~MyCustomOp() {
#ifdef _WIN32
  if (input_data_)
    _aligned_free(input_data_);
  input_data_ = nullptr;
#else
  if (input_data_) {
    free(input_data_);
    input_data_ = nullptr;
  }
  if (cnt == 0) {
    gemm__.reset();
    ryzenai::dynamic_dispatch::xrt_context::destroy_ctx_map();
  }
#endif
}

// Kernel compute
void MyCustomOp::Compute(const OrtApi* api, OrtKernelContext* context) const {
  if (Ort::Global<void>::api_ == nullptr) {
    Ort::Global<void>::api_ = api;
  }
#ifdef PROFILE_MATMULNBITS
  std::chrono::time_point<std::chrono::high_resolution_clock> exec_start,
      exec_stop;
  std::chrono::time_point<std::chrono::high_resolution_clock> preproc_start,
      preproc_end;
  std::chrono::time_point<std::chrono::high_resolution_clock> kernel_start,
      kernel_end;
#endif
  USE_TIMER_MATMULNBITS(exec_start = std::chrono::high_resolution_clock::now());
  Ort::KernelContext ctx(context);
  // Extracting the input and output information
  auto input_tensor = ctx.GetInput(0);                      // Input activations
  auto input_data = input_tensor.GetTensorData<uint16_t>(); // bfloat16 input
  auto input_shape = input_tensor.GetTensorTypeAndShapeInfo().GetShape();

  // Update output shape
  std::vector<int64_t> out_shape;
  for (unsigned i = 0; i < (input_shape.size() - 1); i++)
    out_shape.push_back(input_shape[i]);
  out_shape.push_back(n_sizes_[cnt][1]);

  // Create output tensor
  auto output_tensor = ctx.GetOutput(
      0, {out_shape.begin(), out_shape.end()}); // Output activation

  auto out = output_tensor.GetTensorMutableData<uint16_t>();

  // Execute
  execute_mladf_dd(input_data, out, input_shape, n_sizes_[cnt], grp_sizes_[cnt],
                   cnt);

  USE_TIMER_MATMULNBITS(kernel_end = std::chrono::high_resolution_clock::now());
  USE_TIMER_MATMULNBITS(exec_stop = std::chrono::high_resolution_clock::now());

#ifdef PROFILE_MATMULNBITS
  std::stringstream _csv_out;
  _csv_out << "total execution, preproc, kernel_exec"
           << "\n";
  _csv_out << (exec_stop - exec_start) / std::chrono::microseconds(1) << ",";
  _csv_out << (preproc_end - preproc_start) / std::chrono::microseconds(1)
           << ",";
  _csv_out << (kernel_end - kernel_start) / std::chrono::microseconds(1) << ",";
  std::ofstream csv_file_out("./matmulnbits.csv",
                             std::ios::app | std::ios::out);

  csv_file_out << _csv_out.str() << std::endl;
#endif
}
} // namespace vaip_matmul_nbits_custom_op
