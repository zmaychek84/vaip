/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

// must include cxx_api.hpp before custom_op.hpp otherise
// VAIP_ORT_API_VERSION is not defined we cannot use OrtAPI here.
#if defined(_WIN32)
#  include <intrin.h>
#else
#  include <x86intrin.h>
#endif
#include <mmintrin.h>
#include <ostream>
#include <xmmintrin.h>

#include "onnxruntime_api.hpp"
#include <immintrin.h>

#include "./custom_op.hpp"
#include "reporter.hpp"
#include "vitis/ai/env_config.hpp"
#include "vitis/ai/profiling.hpp"
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

#pragma once
#include "ryzenai/dynamic_dispatch/utils/instruction_registry.hpp"
#if __has_include(<ryzenai/dynamic_dispatch/ops/mladfmatmulbias/mladfmatmulbias.hpp>)
#  include <ryzenai/dynamic_dispatch/ops/mladfmatmulbias/mladfmatmulbias.hpp>
#else
#  include <ops/mladfmatmulbias/mladfmatmulbias.hpp>
#endif
#if defined(_WIN32)
#  pragma warning(disable : 4996)
#endif
namespace fs = std::filesystem;

#define MLADF_VERSION "v1"
#define MAX_SEQ_LENGTH 3072

namespace vaip_matmul_nbits_custom_op {

void MyCustomOp::init_op_mladf_dd(std::vector<int8_t> b,
                                  std::vector<int8_t> zeros,
                                  std::vector<float> scales,
                                  std::vector<float> bias) {

  // Create mladfmatmulbias operator handle
  std::string mladf_version_(MLADF_VERSION);
  if (cnt == 0) {
    std::map<std::string, std::any> mm_attrs;
    mm_attrs["op_version"] = mladf_version_;
    // Create operator instance
    gemm__ = std::make_shared<
        ryzenai::mladfmatmulbias<uint16_t, int8_t, uint16_t, uint16_t>>(
        "bfloat16", "int4", "bfloat16", true, mm_attrs);
  }
  // Get operator
  auto ptr = (ryzenai::mladfmatmulbias<uint16_t, int8_t, uint16_t, uint16_t>*)
                 gemm__.get();
  // Weights shape
  std::vector<size_t> b_shape_dd = {static_cast<size_t>(k_k),
                                    static_cast<size_t>(k_n)};
  // Constant tensors
  Tensor weight_tensor = {b.data(), b_shape_dd, "int4"};
  Tensor bias_tensor = {bias.data(), {(size_t)k_block_size, 0}, "float"};
  Tensor scales_tensor = {scales.data(), {(size_t)k_block_size, 0}, "float"};
  Tensor zeros_tensor = {zeros.data(), b_shape_dd, "int4"};
  std::vector<Tensor> constant_tensors = {weight_tensor, bias_tensor,
                                          scales_tensor, zeros_tensor};
  // Initialize constant tensors (setting up XRT BOs)
  std::map<std::string, std::any> attrs;
  attrs["default_shape"] = 1;
  attrs["op_version"] = mladf_version_;
  attrs["max_m"] = MAX_SEQ_LENGTH;
  attrs["group_size"] = k_block_size;
  ptr->initialize_const_params(constant_tensors, attrs);
}

void MyCustomOp::execute_mladf_dd(const uint16_t* input_data, uint16_t* out,
                                  std::vector<int64_t> input_shape,
                                  std::vector<int> wts_shape, int grp_size,
                                  int run_cnt) const {
  __TIC__(GEMM_PRE)
  auto ptr = (ryzenai::mladfmatmulbias<uint16_t, int8_t, uint16_t, uint16_t>*)
                 gemm__.get();
  // NPU implementation
  int M = input_shape[0] * input_shape[1];
  std::vector<size_t> a_shape = {static_cast<size_t>(M),
                                 static_cast<size_t>(input_shape[2])};
  std::vector<size_t> c_shape = {static_cast<size_t>(M),
                                 static_cast<size_t>(wts_shape[1])};
  std::vector<size_t> wts_shape_dd = {static_cast<size_t>(wts_shape[0]),
                                      static_cast<size_t>(wts_shape[1])};
  // Set shapes for input/weights/output
  ptr->set_shape(a_shape, wts_shape_dd, grp_size);

  // Output tensor
  Tensor output_tensor = {out, c_shape, "bfloat16"};
  std::vector<Tensor> output_tensors = {output_tensor};
  // Input tensor
  Tensor input_tensor = {(int16_t*)input_data, a_shape, "bfloat16"};
  std::vector<Tensor> input_tensors = {input_tensor};
  __TOC__(GEMM_PRE)

  __TIC__(GEMM_RUN)
  // Execute
  ptr->execute_internal(input_tensors, output_tensors, run_cnt);
  __TOC__(GEMM_RUN)
}

} // namespace vaip_matmul_nbits_custom_op
