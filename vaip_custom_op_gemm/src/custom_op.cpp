/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

// must include cxx_api.hpp before custom_op.hpp otherise
// VAIP_ORT_API_VERSION is not defined we cannot use OrtAPI here.
#include "onnxruntime_api.hpp"
#include <stdint.h>
#include <stdio.h>
#include <vcruntime.h>

#include "../../xrt_shared_context/xrt_shared_context.hpp"

#include "vitis/ai/env_config.hpp"
#include "vitis/ai/profiling.hpp"

#include <xir/graph/graph.hpp>

#include "custom_op.hpp"
#pragma once
#include "qlinear_2/qlinear_2.hpp"

#include <filesystem>
#include <fstream>
#include <glog/logging.h>
#include <iostream>
#include <numeric>
#include <sstream>

namespace fs = std::filesystem;
using namespace ryzenai;

DEF_ENV_PARAM(DEBUG_GEMM_CUSTOM_OP, "0")
DEF_ENV_PARAM_2(XLNX_VART_FIRMWARE, "", std::string)
#define LOG_THIS(n) LOG_IF(INFO, ENV_PARAM(DEBUG_GEMM_CUSTOM_OP) >= n)

namespace vaip_gemm_custom_op {

static std::string get_env_variable(const std::string& var,
                                    const std::string& default_val = {}) {
#ifdef _WIN32
  char* value = nullptr;
  size_t size = 0;
  errno_t err = _dupenv_s(&value, &size, var.c_str());
  std::string result =
      (!err && (value != nullptr)) ? std::string{value} : default_val;
  free(value);
#else
  const char* value = std::getenv(var.c_str());
  std::string result = (value != nullptr) ? std::string{value} : default_val;
#endif
  return result;
}

MyCustomOp::MyCustomOp(std::shared_ptr<const PassContext> context,
                       const std::shared_ptr<MetaDefProto>& meta_def,
                       onnxruntime::Model* model)
    : CustomOpImp(context, meta_def, model) { //,

  y_scale_ = std::stof(meta_def->generic_param().at("wts_scale"));
  x_scale_ = std::stof(meta_def->generic_param().at("in_scale"));
  input_zp_ = std::stoi(meta_def->generic_param().at("in_zp"));

  impl_ = meta_def->generic_param().at("impl");

  // Backward compatibility.
  auto xclbin_file = ENV_PARAM(XLNX_VART_FIRMWARE);
  auto cfg_sess_opts = context->get_config_proto().provider_options();
  auto it = cfg_sess_opts.find("xclbin");
  if (it != cfg_sess_opts.end() && !it->second.empty()) {
    xclbin_file = it->second;
  }

  bool share_context = false;
  if (cfg_sess_opts.contains(vaip::Context::CTX_SHARE_OPTION_KEY)) {
    try {
      share_context =
          std::stoi(cfg_sess_opts.at(vaip::Context::CTX_SHARE_OPTION_KEY));
    } catch (...) {
      LOG_THIS(1) << "failed to convert provider option \""
                  << vaip::Context::CTX_SHARE_OPTION_KEY << "\" value \""
                  << cfg_sess_opts.at(vaip::Context::CTX_SHARE_OPTION_KEY)
                  << "\" to int, disable context sharing.";
    }
  }
  if (!share_context) {
    throw std::runtime_error("must enable share context in provider options");
  }

  auto device_id = 0;
  auto context_id = 0;
  context_ = vaip::Context::create_shared_context(*context, device_id,
                                                  context_id, xclbin_file);

  std::string inputbin = meta_def->generic_param().at("bias_file");
  bias_file_ = inputbin;

  std::string inputbin_wts = meta_def->generic_param().at("wts_file");
  wts_file_ = inputbin_wts;

  auto shape_0 = stoi(meta_def->generic_param().at("wts_shape_dim_1"));
  auto shape_1 = stoi(meta_def->generic_param().at("wts_shape_dim_0"));

  wts_shape_ = std::make_tuple(shape_0, shape_1);

  if (inputbin != "null") {
    bias_.resize(std::get<1>(wts_shape_));

    auto infile = std::ifstream(bias_file_, std::ios::in | std::ios::binary);
    for (unsigned i = 0; infile.read(((char*)&bias_[i]), sizeof(float)); i++)
      ;
  }

  unsigned int size = (unsigned int)fs::file_size(inputbin_wts);
  int8_t* wts = (int8_t*)malloc(size);

  auto infile = std::ifstream(inputbin_wts, std::ios::in | std::ios::binary);
  for (unsigned i = 0; infile.read(&((char*)wts)[i], sizeof(int8_t)); i++)
    ;

  if (impl_ == "v1") {

    const std::string& a_dtype = "int8";
    const std::string& b_dtype = "int8";
    const std::string& c_dtype = "int32";
    gemm_ = std::make_shared<qlinear_2<int8_t, int8_t, int32_t>>(
        a_dtype, b_dtype, c_dtype);
    qlinear_2<int8_t, int8_t, int32_t>* ptr =
        (qlinear_2<int8_t, int8_t, int32_t>*)gemm_.get();
    ptr->initialize_weights(wts, wts_shape_);

  } else {
    throw std::runtime_error(
        "ERROR : # Implementaion is not available for this version");
  }

  free(wts);
}

MyCustomOp::~MyCustomOp() {}

void MyCustomOp::Compute(const OrtApi* api, OrtKernelContext* context) const {
  if (Ort::Global<void>::api_ == nullptr) {
    Ort::Global<void>::api_ = api;
  }
  Ort::KernelContext ctx(context);
  auto input_tensor = ctx.GetInput(0);
  auto input_data = input_tensor.GetTensorData<int8_t>();
  auto input_shape = input_tensor.GetTensorTypeAndShapeInfo().GetShape();
  auto num_outputs = ctx.GetOutputCount();
#ifdef PROFILE_GEMM
  std::chrono::time_point<std::chrono::high_resolution_clock>
      _dynamic_scale_start, _dynamic_scale_end;
  std::chrono::time_point<std::chrono::high_resolution_clock> _bias_add_start,
      _bias_add_end;
#endif
  USE_TIMER_GEMM(_dynamic_scale_start =
                     std::chrono::high_resolution_clock::now());

  USE_TIMER_GEMM(_dynamic_scale_end =
                     std::chrono::high_resolution_clock::now());
  std::vector<int64_t> out_shape;
  int batch;
  if (input_shape.size() == 4) {
    out_shape.push_back(input_shape[0]);
    out_shape.push_back(input_shape[1]);
    out_shape.push_back(input_shape[2]);
    batch = input_shape[0] * input_shape[1];
  } else if (input_shape.size() == 3) {
    out_shape.push_back(input_shape[0]);
    out_shape.push_back(input_shape[1]);
    batch = input_shape[0];
  } else {
    for (unsigned i = 0; i < (input_shape.size() - 1); i++)
      out_shape.push_back(input_shape[i]);
    batch = 1;
  }
  out_shape.push_back(std::get<1>(wts_shape_));

  auto output_tensor = ctx.GetOutput(0, {out_shape.begin(), out_shape.end()});
  std::tuple<int, int> input_s =
      std::make_tuple((int)input_shape[input_shape.size() - 2] * batch,
                      (int)input_shape[input_shape.size() - 1]);

  auto out_base = output_tensor.GetTensorMutableData<float>();
  size_t in_size = std::accumulate(input_shape.begin(), input_shape.end(), 1,
                                   std::multiplies<size_t>());
  size_t out_size = std::accumulate(out_shape.begin(), out_shape.end(), 1,
                                    std::multiplies<size_t>());

  std::vector<int8_t> input_data1(in_size, 0);
  std::vector<int32_t> out_tmp(out_size, 0);

  for (size_t i = 0; i < in_size; i++) {
    input_data1[i] = (int8_t)(input_data[i] - input_zp_);
  }

  auto _exec_start = std::chrono::high_resolution_clock::now();

  qlinear_2<int8_t, int8_t, int32_t>* ptr =
      (qlinear_2<int8_t, int8_t, int32_t>*)gemm_.get();
  ptr->execute(input_data1.data(), input_s, out_tmp.data());

  for (int i = 0; i < out_size; i++) {
    out_base[i] = (float)(out_tmp[i] * x_scale_ * y_scale_);
  }
  if (bias_file_ != "null") {
    USE_TIMER_GEMM(_bias_add_start = std::chrono::high_resolution_clock::now());
    for (int i = 0; i < std::get<0>(input_s); i++) {
      for (int j = 0; j < std::get<1>(wts_shape_); j++) {
        int index = i * std::get<1>(wts_shape_) + j;
        out_base[index] += bias_[j];
      }
    }

    USE_TIMER_GEMM(_bias_add_end = std::chrono::high_resolution_clock::now());
  }

  auto _exec_stop = std::chrono::high_resolution_clock::now();
  auto exec_total = (_exec_stop - _exec_start) / std::chrono::microseconds(1);
#ifdef PROFILE_GEMM
  std::stringstream _csv_out;
  _csv_out << "input scale calculate, bias add"
           << "\n";
  _csv_out << (_dynamic_scale_end - _dynamic_scale_start) /
                  std::chrono::microseconds(1)
           << ",";
  _csv_out << (_bias_add_end - _bias_add_start) / std::chrono::microseconds(1)
           << ",";
  std::ofstream csv_file_out("./gemmop.csv", std::ios::app | std::ios::out);

  csv_file_out << _csv_out.str() << std::endl;
#endif
  auto out = out_base;
}
} // namespace vaip_gemm_custom_op
