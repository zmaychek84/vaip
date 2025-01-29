/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

#include "onnxruntime_api.hpp"

#include <glog/logging.h>
#include <sstream>

#include "custom_op.hpp"
#include "norm.hpp"
#include "resize_down.hpp"
#include "vitis/ai/profiling.hpp"
#include "xf_aie_host_utils.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>

DEF_ENV_PARAM(DEBUG_RESIZE_NORM_CUSTOM_OP, "0")
DEF_ENV_PARAM_2(XLNX_VART_FIRMWARE, "", std::string)
#define LOG_THIS(n) LOG_IF(INFO, ENV_PARAM(DEBUG_RESIZE_NORM_CUSTOM_OP) >= n)

namespace vaip_resize_norm_custom_op {

MyCustomOp::MyCustomOp(std::shared_ptr<const PassContext> context,
                       const std::shared_ptr<MetaDefProto>& meta_def,
                       onnxruntime::Model* model)
    : CustomOpImp(context, meta_def, model),
      kernel_name_resize_(PPGetPSKernelName(PP_RESIZE_DOWN)),
      kernel_name_norm_(PPGetPSKernelName(PP_NORM)) {
  // Get node attributes
  if (meta_def_->generic_param().contains("input_shape")) {
    std::string shape = meta_def_->generic_param().at("input_shape");
    input_shape_ = string_to_shape(shape);
    LOG_THIS(1) << "Input Shape: " << shape;
  } else {
    LOG_THIS(1) << "No attribute \"input_shape\"\n";
  }
  if (meta_def_->generic_param().contains("output_shape")) {
    std::string shape = meta_def_->generic_param().at("output_shape");
    output_shape_ = string_to_shape(shape);
    LOG_THIS(1) << "Output Shape: " << shape;
  } else {
    LOG_THIS(1) << "No attribute \"output_shape\"\n";
  }
  if (meta_def_->generic_param().contains("transpose")) {
    std::string val = meta_def_->generic_param().at("transpose");
    en_transpose_ = atoi(val.c_str());
    LOG_THIS(1) << "transpose: " << val;
  } else {
    LOG_THIS(1) << "No attribute \"transpose\"\n";
  }
  if (meta_def_->generic_param().contains("alpha_fbits")) {
    std::string val = meta_def_->generic_param().at("alpha_fbits");
    fl_bits_.push_back(atoi(val.c_str()));
    LOG_THIS(1) << "alpha fbits: " << val;
  } else {
    LOG_THIS(1) << "No attribute \"alpha fbits\"\n";
  }
  if (meta_def_->generic_param().contains("beta_fbits")) {
    std::string val = meta_def_->generic_param().at("beta_fbits");
    fl_bits_.push_back(atoi(val.c_str()));
    LOG_THIS(1) << "beta fbits: " << val;
  } else {
    LOG_THIS(1) << "No attribute \"beta fbits\"\n";
  }
  if (meta_def_->generic_param().contains("output_fbits")) {
    std::string val = meta_def_->generic_param().at("output_fbits");
    fl_bits_.push_back(atoi(val.c_str()));
    LOG_THIS(1) << "output fbits: " << val;
  } else {
    LOG_THIS(1) << "No attribute \"output fbits\"\n";
  }
  if (meta_def_->generic_param().contains("Mean")) {
    std::string val = meta_def_->generic_param().at("Mean");
    mean_ = string_to_float_shape(val);
    LOG_THIS(1) << "mean: " << val;
  } else {
    LOG_THIS(1) << "No attribute \"mean\"\n";
  }
  if (meta_def_->generic_param().contains("StdDev")) {
    std::string val = meta_def_->generic_param().at("StdDev");
    stddev_ = string_to_float_shape(val);
    LOG_THIS(1) << "std dev: " << val;
  } else {
    LOG_THIS(1) << "No attribute \"std dev\"\n";
  }
  if (meta_def_->generic_param().contains("size")) {
    std::string val = meta_def_->generic_param().at("size");
    target_shape_ = string_to_shape(val);
    LOG_THIS(1) << "out size: " << val;
  } else {
    LOG_THIS(1) << "No attribute \"size\"\n";
  }

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

  // Get attributes
  auto attrs = context_->get_attrs();

  // Get/Create Device
  LOG_THIS(1) << "Using XRT device from vaip";

  // Create kernels
  context_->create_kernel(kernel_name_resize_, kernel_name_resize_);
  context_->create_kernel(kernel_name_norm_, kernel_name_norm_);

  LOG_THIS(1) << "Kernel Name: " << kernel_name_resize_;
  LOG_THIS(1) << "Kernel Name: " << kernel_name_resize_;

  // Get device/kernel
  auto device = attrs->get_attr<xrt::device*>("xrt_device");
  auto k_resize = attrs->get_attr<xrt::kernel*>(kernel_name_resize_.c_str());
  auto k_norm = attrs->get_attr<xrt::kernel*>(kernel_name_norm_.c_str());

  // Create compute kernel
  kernel_resize_ = std::make_unique<ResizeDown>(
      *device, *k_resize, input_shape_, output_shape_, target_shape_);
  kernel_norm_ =
      std::make_unique<Normalize>(*device, *k_norm, input_shape_, output_shape_,
                                  target_shape_, fl_bits_, mean_, stddev_);

  // Create Sub BO
  // 32kb alignment
  uint32_t alignment = 32 << 10;
  auto sub_bo_size_rsz =
      ((kernel_resize_->get_instr_size() + (alignment - 1)) / alignment) *
      alignment;
  auto sub_bo_size_nrm =
      ((kernel_norm_->get_instr_size() + (alignment - 1)) / alignment) *
      alignment;
  // get sub-bo offset
  size_t offset = 0;
  if (attrs->has_attr("bo_offset"))
    offset = attrs->get_attr<size_t>("bo_offset");

  LOG_THIS(1) << "SRAM BO Start offset: " << offset;

  auto bo_sram = attrs->get_attr<xrt::bo*>("bo_sram");
  // create sub-bo for kernels
  instr_bo_resize_ = xrt::bo(*bo_sram, sub_bo_size_rsz, offset);
  offset += sub_bo_size_rsz;

  instr_bo_norm_ = xrt::bo(*bo_sram, sub_bo_size_nrm, offset);
  offset += sub_bo_size_nrm;

  LOG_THIS(1) << "SRAM BO End offset: " << offset;

  // update offset for sub-bo creation
  attrs->set_attr<size_t>("bo_offset", offset);

  // set attr for norm output precision bits
  attrs->set_attr<int>("norm_out_fl", fl_bits_[2]);

  // Sync Instructions
  kernel_resize_->sync_instructions(instr_bo_resize_);
  kernel_norm_->sync_instructions(instr_bo_norm_);
}

MyCustomOp::~MyCustomOp() {}

void MyCustomOp::Compute(const OrtApi* api, OrtKernelContext* context) const {
  if (Ort::Global<void>::api_ == nullptr) {
    Ort::Global<void>::api_ = api;
  }
  __TIC__(ResizeNormCompute);
  Ort::KernelContext ctx(context);
  auto num_inputs = ctx.GetInputCount();
  auto num_outputs = ctx.GetOutputCount();

#ifdef DEBUG
  LOG(INFO) << "Num inputs: " << num_inputs << ", "
            << "Num outputs: " << num_outputs << " ";

  // input info
  for (auto idx = 0u; idx < num_inputs; ++idx) {
    auto tensor = ctx.GetInput(idx);
    auto tensor_info = tensor.GetTensorTypeAndShapeInfo();
    auto tensor_type = tensor_info.GetElementType();
    auto tensor_shape = tensor_info.GetShape();
    LOG(INFO) << "Input -> Tensor type: " << tensor_type
              << ", Shape: " << shape_to_string(tensor_shape);
  }
  // output info
  for (auto idx = 0u; idx < num_outputs; ++idx) {
    auto tensor = ctx.GetOutput(idx, output_shape_);
    auto tensor_info = tensor.GetTensorTypeAndShapeInfo();
    auto tensor_type = tensor_info.GetElementType();
    auto tensor_shape = tensor_info.GetShape();
    LOG(INFO) << "Output->Tensor type : " << tensor_type
              << ", Shape = " << shape_to_string(tensor_shape);
  }
#endif

  CHECK_EQ(num_inputs, 1u);
  CHECK_EQ(num_outputs, 1u);

  auto input_tensor = ctx.GetInput(0);
  auto input_raw = input_tensor.GetTensorData<uint8_t>();
  auto tensor_info = input_tensor.GetTensorTypeAndShapeInfo();
  auto element_num = tensor_info.GetElementCount();
  // Copy input data to BO
  auto input_host = kernel_resize_->get_host_buffer_in();
  memcpy((void*)input_host, (void*)input_raw, element_num);
  auto attrs = context_->get_attrs();
  // Run AIE Kernel (Resize)
  if (attrs->has_attr(kernel_name_resize_.c_str()) &&
      attrs->has_attr("bo_sram")) {
    auto bo_sram = const_cast<xrt::bo*>(&instr_bo_resize_);
    auto kernel = attrs->get_attr<xrt::kernel*>(kernel_name_resize_.c_str());
    kernel_resize_->exec(*kernel, *bo_sram);
  } else {
    throw std::runtime_error("Could not find kernel/BO");
  }

#ifdef DEBUG
  auto reout = kernel_resize_->get_host_buffer_out();
  FILE* refp = nullptr;
  fopen_s(&refp, "resize_out.txt", "w");
  for (int i = 0; i < target_shape_[0] * target_shape_[1] * target_shape_[2];
       ++i) {
    fprintf(refp, "%x\n", reout[i]);
  }
  fclose(refp);
#endif

  // Run AIE Kernel (Norm)
  if (attrs->has_attr(kernel_name_norm_.c_str()) &&
      attrs->has_attr("bo_sram")) {
    auto bo_sram = const_cast<xrt::bo*>(&instr_bo_norm_);
    auto kernel = attrs->get_attr<xrt::kernel*>(kernel_name_norm_.c_str());
    auto resize_out = kernel_resize_->get_output_bo();
    kernel_norm_->exec(*kernel, *bo_sram, resize_out);
  } else {
    throw std::runtime_error("Could not find kernel/BO");
  }

#ifdef DEBUG
  auto nmout = kernel_norm_->get_host_buffer_out();
  FILE* nmfp = nullptr;
  fopen_s(&nmfp, "norm_out.txt", "w");
  for (int i = 0; i < target_shape_[0] * target_shape_[1] * target_shape_[2];
       ++i) {
    fprintf(nmfp, "%x\n", nmout[i]);
  }
  fclose(nmfp);
#endif

  // Copy 3-channel data to output from AIE out buffer
  auto height = 0;
  auto width = 0;
  auto ch3 = 0;
  auto ch4 = 4;

  // Fill output shape
  if (en_transpose_) {
    height = static_cast<int>(output_shape_[2]);
    width = static_cast<int>(output_shape_[3]);
    ch3 = static_cast<int>(output_shape_[1]);
  } else {
    height = static_cast<int>(output_shape_[1]);
    width = static_cast<int>(output_shape_[2]);
    ch3 = static_cast<int>(output_shape_[3]);
  }

  auto output_tensor = ctx.GetOutput(0, output_shape_);
  auto output_raw = output_tensor.GetTensorMutableData<float>();
  auto norm_out = kernel_norm_->get_host_buffer_out();

  // get norm output fl bits
  auto norm_out_fl = attrs->get_attr<int>("norm_out_fl");
  if (en_transpose_) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        for (int c = 0; c < ch3; ++c) {
          int hwc_index = h * width * ch4 + w * ch4 + c;
          int chw_index = c * height * width + h * width + w;
          auto val = static_cast<float>(norm_out[hwc_index]) /
                     static_cast<float>(1 << norm_out_fl);
          output_raw[chw_index] = val;
        }
      }
    }
  } else {
    for (int row = 0; row < height; ++row) {
      for (int col = 0; col < width; ++col) {
        for (int ch = 0; ch < ch3; ++ch) {
          auto index_ch3 = (row * width * ch3) + (col * ch3) + ch;
          auto index_ch4 = (row * width * ch4) + (col * ch4) + ch;
          auto val = static_cast<float>(norm_out[index_ch4]) /
                     static_cast<float>(1 << norm_out_fl);
          output_raw[index_ch3] = val;
        }
      }
    }
  }
  __TOC__(ResizeNormCompute);
}
} // namespace vaip_resize_norm_custom_op
