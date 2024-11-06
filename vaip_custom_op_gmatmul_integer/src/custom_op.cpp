/*
 *     The Xilinx Vitis AI Vaip in this distribution are provided under the
 * following free and permissive binary-only license, but are not provided in
 * source code form.  While the following free and permissive license is similar
 * to the BSD open source license, it is NOT the BSD open source license nor
 * other OSI-approved open source license.
 *
 *      Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc.
 *
 *      Redistribution and use in binary form only, without modification, is
 * permitted provided that the following conditions are met:
 *
 *      1. Redistributions must reproduce the above copyright notice, this list
 * of conditions and the following disclaimer in the documentation and/or other
 * materials provided with the distribution.
 *
 *      2. The name of Xilinx, Inc. may not be used to endorse or promote
 * products redistributed with this software without specific prior written
 * permission.
 *
 *      THIS SOFTWARE IS PROVIDED BY XILINX, INC. "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL XILINX, INC. BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 *      PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE
 */
#include "onnxruntime_api.hpp"

#include "./custom_op.hpp"
#include <filesystem>
#include <fstream>
#include <glog/logging.h>
#include <iostream>
#include <sstream>
#pragma once
#include "qlinear_2/qlinear_2.hpp"
#if defined(_WIN32)
#  pragma warning(disable : 4996)
#endif

namespace fs = std::filesystem;
using namespace ryzenai;

#define OUT_TYPE int32_t

namespace vaip_gmatmul_integer_custom_op {
MyCustomOp::MyCustomOp(std::shared_ptr<const PassContext> context,
                       const std::shared_ptr<MetaDefProto>& meta_def,
                       onnxruntime::Model* model)
    : CustomOpImp(context, meta_def, model) {
  impl_ = meta_def->generic_param().at("impl");

  bool pack_weights = true;

  auto splits = meta_def->generic_param().at("wts_shape_dim_split");
  std::string sep = ",";
  size_t pos = 0;
  while ((pos = splits.find(sep)) != std::string::npos) {
    wts_shape_dim_split_.push_back(stoi(splits.substr(0, pos)));
    splits.erase(0, pos + sep.length());
  }

  std::string inputbin_wts = meta_def->generic_param().at("wts_file");
  auto shape_0 = stoi(meta_def->generic_param().at("wts_shape_dim_0"));
  auto shape_1 = stoi(meta_def->generic_param().at("wts_shape_dim_1"));

  wts_shape_ = std::make_tuple(shape_0, shape_1);

  unsigned int size = (unsigned int)fs::file_size(inputbin_wts);
  int8_t* wts = (int8_t*)malloc(size);

  auto infile = std::ifstream(inputbin_wts, std::ios::in | std::ios::binary);
  for (unsigned i = 0; infile.read(&((char*)wts)[i], sizeof(int8_t)); i++)
    ;

  if (impl_ == "v1") {
    const std::string& a_dtype = "int8";
    const std::string& b_dtype = "int8";
    const std::string& c_dtype = "int32";
    gemm_ = std::make_shared<qlinear_2<int8_t, int8_t, OUT_TYPE>>(
        a_dtype, b_dtype, c_dtype);
    qlinear_2<int8_t, int8_t, OUT_TYPE>* ptr =
        (qlinear_2<int8_t, int8_t, OUT_TYPE>*)gemm_.get();
    ptr->initialize_weights(wts, wts_shape_);
  } else {
    throw std::runtime_error(
        "ERROR : # Implementaion is not available for this device");
  }

  wts_sum_.reserve(std::get<1>(wts_shape_));
  for (int i = 0; i < std::get<1>(wts_shape_); i++) {
    wts_sum_[i] = 0;
    for (int j = 0; j < std::get<0>(wts_shape_); j++) {
      wts_sum_[i] += wts[j * std::get<1>(wts_shape_) + i];
    }
  }
  free(wts);
}

MyCustomOp::~MyCustomOp() {}

void MyCustomOp::Compute(const OrtApi* api, OrtKernelContext* context) const {
  if (Ort::Global<void>::api_ == nullptr) {
    Ort::Global<void>::api_ = api;
  }
#ifdef PROFILE_GMATMULINTEGER
  std::chrono::time_point<std::chrono::high_resolution_clock> exec_start,
      exec_stop;
  std::chrono::time_point<std::chrono::high_resolution_clock> preproc_start,
      preproc_end;
  std::chrono::time_point<std::chrono::high_resolution_clock> kernel_start,
      kernel_end;
  std::chrono::time_point<std::chrono::high_resolution_clock> scale_start,
      scale_end;
#endif

  USE_TIMER_GMATMULINTEGER(exec_start =
                               std::chrono::high_resolution_clock::now());
  Ort::KernelContext ctx(context);
  auto input_tensor = ctx.GetInput(0);
  auto input_data = input_tensor.GetTensorData<uint8_t>();
  auto input_shape = input_tensor.GetTensorTypeAndShapeInfo().GetShape();
  auto input_tensor1 = ctx.GetInput(1); // Zero point
  auto input_zero_point = input_tensor1.GetTensorData<uint8_t>();

  std::tuple<int, int> input_s =
      std::make_tuple((int)input_shape[input_shape.size() - 2],
                      (int)input_shape[input_shape.size() - 1]);

  auto _exec_start = std::chrono::high_resolution_clock::now();
  size_t in_size = std::get<0>(input_s) * std::get<1>(input_s);
  size_t out_size = std::get<0>(input_s) * std::get<1>(wts_shape_);

  USE_TIMER_GMATMULINTEGER(preproc_start =
                               std::chrono::high_resolution_clock::now());
  std::vector<int8_t> input_data1(in_size, 0);
  std::vector<OUT_TYPE> out_tmp(out_size, 0);

  for (size_t i = 0; i < in_size; i++) {
    input_data1[i] = (int8_t)(input_data[i] - 128);
  }
  USE_TIMER_GMATMULINTEGER(preproc_end =
                               std::chrono::high_resolution_clock::now());

  USE_TIMER_GMATMULINTEGER(kernel_start =
                               std::chrono::high_resolution_clock::now());

  qlinear_2<int8_t, int8_t, OUT_TYPE>* ptr =
      (qlinear_2<int8_t, int8_t, OUT_TYPE>*)gemm_.get();
  ptr->execute(input_data1.data(), input_s, out_tmp.data());

  USE_TIMER_GMATMULINTEGER(kernel_end =
                               std::chrono::high_resolution_clock::now());
  USE_TIMER_GMATMULINTEGER(scale_start =
                               std::chrono::high_resolution_clock::now());

  std::vector<int64_t> out_shape;
  for (unsigned i = 0; i < (input_shape.size() - 1); i++)
    out_shape.push_back(input_shape[i]);

  int offset = 0;
  for (int l = 0; l < wts_shape_dim_split_.size(); l++) {
    // Reference:
    // https://leimao.github.io/article/Neural-Networks-Quantization/#Quantized%20Matrix%20Multiplication:~:text=Quantized%20Matrix%20Multiplication-,Quantized%20Matrix%20Multiplication%20Mathematics,-Suppose%20we%20have
    // Assuming that the zero point of weight is zero
    int wts_shape_1 = wts_shape_dim_split_[l];
    out_shape.push_back(wts_shape_1);
    auto output_tensor = ctx.GetOutput(l, {out_shape.begin(), out_shape.end()});
    auto out_base = output_tensor.GetTensorMutableData<int32_t>();

    for (int i = 0; i < std::get<0>(input_s); i++) {
      for (int j = 0; j < wts_shape_1; j++) {
        int32_t temp = out_tmp[i * std::get<1>(wts_shape_) + j + offset];
        out_base[i * wts_shape_1 + j] =
            (temp - ((input_zero_point[0] - 128) * (wts_sum_[j + offset])));
      }
    }
    out_shape.pop_back();
    offset += wts_shape_1;
  }
  USE_TIMER_GMATMULINTEGER(scale_end =
                               std::chrono::high_resolution_clock::now());
  USE_TIMER_GMATMULINTEGER(exec_stop =
                               std::chrono::high_resolution_clock::now());

#ifdef PROFILE_GMATMULINTEGER
  std::stringstream _csv_out;
  _csv_out << "total execution, preproc, kernel_exec, scale_cal"
           << "\n";
  _csv_out << (exec_stop - exec_start) / std::chrono::microseconds(1) << ",";
  _csv_out << (preproc_end - preproc_start) / std::chrono::microseconds(1)
           << ",";
  _csv_out << (kernel_end - kernel_start) / std::chrono::microseconds(1) << ",";
  _csv_out << (scale_end - scale_start) / std::chrono::microseconds(1) << ",";
  std::ofstream csv_file_out("./gmatmulinteger.csv",
                             std::ios::app | std::ios::out);

  csv_file_out << _csv_out.str() << std::endl;
#endif
}
} // namespace vaip_gmatmul_integer_custom_op
