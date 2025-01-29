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

DEF_ENV_PARAM(DEBUG_GEMM_DYNAMIC_CUSTOM_OP, "0")
DEF_ENV_PARAM_2(XLNX_VART_FIRMWARE, "", std::string)
#define LOG_THIS(n) LOG_IF(INFO, ENV_PARAM(DEBUG_GEMM_DYNAMIC_CUSTOM_OP) >= n)

namespace vaip_gemm_dynamic_custom_op {

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

  a_scale_ = std::stof(meta_def->generic_param().at("a_in_scale"));
  a_input_zp_ = std::stoi(meta_def->generic_param().at("a_in_zp"));
  b_scale_ = std::stof(meta_def->generic_param().at("b_in_scale"));
  b_input_zp_ = std::stoi(meta_def->generic_param().at("b_in_zp"));

  impl_ = meta_def->generic_param().at("impl");

  // Backward compatibility.
  auto xclbin_file = ENV_PARAM(XLNX_VART_FIRMWARE);
  auto cfg_sess_opts = context->get_config_proto().provider_options();
  auto it = cfg_sess_opts.find("xclbin");
  if (it != cfg_sess_opts.end() && !it->second.empty()) {
    xclbin_file = it->second;
  }

  std::string file = "dynamic_profile_log.csv";
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

  if (impl_ == "v1") {
    const std::string& a_dtype = "int8";
    const std::string& b_dtype = "int8";
    const std::string& c_dtype = "int32";
    gemm_ = std::make_shared<qlinear_2<int8_t, int8_t, int32_t>>(
        a_dtype, b_dtype, c_dtype);
    qlinear_2<int8_t, int8_t, int32_t>* ptr =
        (qlinear_2<int8_t, int8_t, int32_t>*)gemm_.get();
  } else {
    throw std::runtime_error(
        "ERROR : # Implementaion is not available for this version");
  }
}

MyCustomOp::~MyCustomOp() {}

void MyCustomOp::Compute(const OrtApi* api, OrtKernelContext* context) const {
  if (Ort::Global<void>::api_ == nullptr) {
    Ort::Global<void>::api_ = api;
  }
  Ort::KernelContext ctx(context);
  auto in1shape = ctx.GetInput(0).GetTensorTypeAndShapeInfo().GetShape();
  auto in2shape = ctx.GetInput(1).GetTensorTypeAndShapeInfo().GetShape();

  int a_index = 0;
  int b_index = 1;

  if (in1shape[3] != in2shape[2]) {
    a_index = 1;
    b_index = 0;
  }

  auto a_input_tensor = ctx.GetInput(a_index);
  auto a_intensor_data = a_input_tensor.GetTensorData<int8_t>();
  auto a_input_shape = a_input_tensor.GetTensorTypeAndShapeInfo().GetShape();
  auto b_input_tensor = ctx.GetInput(b_index);
  auto b_intensor_data = b_input_tensor.GetTensorData<int8_t>();
  auto b_input_shape = b_input_tensor.GetTensorTypeAndShapeInfo().GetShape();

  auto num_outputs = ctx.GetOutputCount();

  auto shape_0 = b_input_shape[2];
  auto shape_1 = b_input_shape[3];

  wts_shape_ = std::make_tuple((int)shape_0, (int)shape_1);
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
  if (a_input_shape.size() == 4) {
    out_shape.push_back(a_input_shape[0]);
    out_shape.push_back(a_input_shape[1]);
    out_shape.push_back(a_input_shape[2]);
    batch = a_input_shape[0] * a_input_shape[1];
  } else if (a_input_shape.size() == 3) {
    out_shape.push_back(a_input_shape[0]);
    out_shape.push_back(b_input_shape[1]);
    batch = a_input_shape[0];
  } else {
    for (unsigned i = 0; i < (a_input_shape.size() - 1); i++)
      out_shape.push_back(a_input_shape[i]);
    batch = 1;
  }
  out_shape.push_back(std::get<1>(wts_shape_));

  auto output_tensor = ctx.GetOutput(0, {out_shape.begin(), out_shape.end()});

  std::tuple<int, int> a_input_s =
      std::make_tuple((int)a_input_shape[a_input_shape.size() - 2],
                      (int)a_input_shape[a_input_shape.size() - 1]);
  size_t a_2d_size = std::get<0>(a_input_s) * std::get<1>(a_input_s);

  std::tuple<int, int> b_input_s =
      std::make_tuple((int)b_input_shape[b_input_shape.size() - 2],
                      (int)b_input_shape[b_input_shape.size() - 1]);
  size_t b_2d_size = std::get<0>(b_input_s) * std::get<1>(b_input_s);

  std::tuple<int, int> c_output_s =
      std::make_tuple((int)out_shape[out_shape.size() - 2],
                      (int)out_shape[out_shape.size() - 1]);
  size_t c_2d_size = std::get<0>(c_output_s) * std::get<1>(c_output_s);

  auto out_base = output_tensor.GetTensorMutableData<float>();

  size_t a_in_size = std::accumulate(a_input_shape.begin(), a_input_shape.end(),
                                     1, std::multiplies<size_t>());

  size_t b_in_size = std::accumulate(b_input_shape.begin(), b_input_shape.end(),
                                     1, std::multiplies<size_t>());

  size_t out_size = std::accumulate(out_shape.begin(), out_shape.end(), 1,
                                    std::multiplies<size_t>());

  std::vector<int8_t> a_input_data(a_in_size, 0);
  std::vector<int8_t> b_input_data(b_in_size, 0);
  std::vector<int32_t> out_tmp(out_size, 0);

  for (size_t i = 0; i < a_in_size; i++) {
    a_input_data[i] = (int8_t)(a_intensor_data[i] - a_input_zp_);
  }

  for (size_t i = 0; i < b_in_size; i++) {
    b_input_data[i] = (int8_t)(b_intensor_data[i] - b_input_zp_);
  }

  auto _exec_start = std::chrono::high_resolution_clock::now();

  qlinear_2<int8_t, int8_t, int32_t>* ptr =
      (qlinear_2<int8_t, int8_t, int32_t>*)gemm_.get();

  for (int bat_id = 0; bat_id < batch; bat_id++) {
    int32_t* out_ptr = out_tmp.data() + (bat_id * c_2d_size);
    int8_t* a_ptr = a_input_data.data() + (bat_id * a_2d_size);
    int8_t* b_ptr = b_input_data.data() + (bat_id * b_2d_size);

    // Fill B Matrix
    ptr->initialize_weights(b_ptr, wts_shape_);
    ptr->execute(a_ptr, a_input_s, out_ptr);
  }

  for (int i = 0; i < out_size; i++) {
    out_base[i] = (float)(out_tmp[i] * a_scale_ * b_scale_);
  }

  auto out = out_base;
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
}
} // namespace vaip_gemm_dynamic_custom_op
