/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

#include "./custom_op.hpp"
#include "fstream"
#include "iostream"
#include "onnxruntime_api.hpp"
#include <cstdint>
#include <glog/logging.h>
#include <sstream>
#include <vector>

namespace vaip_gather_add_custom_op {

MyCustomOp::MyCustomOp(std::shared_ptr<const PassContext> context,
                       const std::shared_ptr<MetaDefProto>& meta_def,
                       onnxruntime::Model* model)
    : CustomOpImp(context, meta_def, model) {
  ifm_dim_0_ = stoi(meta_def->generic_param().at("ifm_dim_0"));
  ifm_dim_1_ = stoi(meta_def->generic_param().at("ifm_dim_1"));
  indeces_shape_ = stoi(meta_def->generic_param().at("indeces_shape"));

  act_zp_ = stoi(meta_def->generic_param().at("act_zp"));
  wt_zp_ = stoi(meta_def->generic_param().at("wts_zp"));
  act_scale_ = stof(meta_def->generic_param().at("act_scale"));
  wt_scale_ = stof(meta_def->generic_param().at("wts_scale"));

  // weight tensor as bin file
  auto dd_cache_dir = context->get_log_dir();
  auto data_file = dd_cache_dir / meta_def->generic_param().at("data_file");
  auto data_file_opt = context->read_file_c8(
      std::filesystem::path(data_file).filename().string());
  if (!data_file_opt.has_value()) {
    std::cerr << "Error reading file: " << data_file << std::endl;
  }
  auto file = data_file_opt.value();
  in_data_.resize(file.size() / sizeof(uint8_t));
  memcpy(in_data_.data(), file.data(), file.size());

  auto wts_file = dd_cache_dir / meta_def->generic_param().at("wts_file");
  std::vector<uint8_t> wts_data_tmp;
  std::streamsize sizew;
  auto wts_data_opt = context->read_file_u8(
      std::filesystem::path(wts_file).filename().string());
  if (!wts_data_opt.has_value()) {
    std::cerr << "Error reading file: " << wts_file << std::endl;
  }
  wts_data_tmp = wts_data_opt.value();
  sizew = wts_data_tmp.size();
  wts_data_.resize(sizew / sizeof(uint8_t));
  for (int i = 0; i < sizew; i++)
    wts_data_[i] = ((float)wts_data_tmp[i] - wt_zp_) * wt_scale_;
}

MyCustomOp::~MyCustomOp() {}

void MyCustomOp::Compute(const OrtApi* api, OrtKernelContext* context) const {
  if (Ort::Global<void>::api_ == nullptr) {
    Ort::Global<void>::api_ = api;
  }
  Ort::KernelContext ctx(context);

  auto num_inputs = ctx.GetInputCount();

  auto num_outputs = ctx.GetOutputCount();

  auto input_tensor = ctx.GetInput(0);

  auto tensor_info = input_tensor.GetTensorTypeAndShapeInfo();
  auto indeces = input_tensor.GetTensorData<int64_t>();

  auto output_tensor = ctx.GetOutput(0, {1, indeces_shape_, ifm_dim_1_});

  auto out_data = output_tensor.GetTensorMutableData<float>();
  for (size_t i = 0; i < indeces_shape_; i++) {

    int idx_v = (int)indeces[i];

    for (int j = 0; j < ifm_dim_1_; j++) {
      uint8_t temp = in_data_[idx_v * ifm_dim_1_ + j];
      float val = ((float)temp - act_zp_) * act_scale_;
      out_data[i * ifm_dim_1_ + j] = val + wts_data_[i * ifm_dim_1_ + j];
    }
  }
}
} // namespace vaip_gather_add_custom_op
