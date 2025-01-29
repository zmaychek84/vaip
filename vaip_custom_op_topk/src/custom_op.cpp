/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

#include "onnxruntime_api.hpp"

#include <glog/logging.h>
#include <sstream>

#include "custom_op.hpp"
#include "topk.hpp"
#include "vitis/ai/profiling.hpp"
#include "xf_aie_host_utils.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>

DEF_ENV_PARAM(DEBUG_TOPK_CUSTOM_OP, "0")
DEF_ENV_PARAM_2(XLNX_VART_FIRMWARE, "", std::string)
#define LOG_THIS(n) LOG_IF(INFO, ENV_PARAM(DEBUG_TOPK_CUSTOM_OP) >= n)

namespace vaip_topk_custom_op {

MyCustomOp::MyCustomOp(std::shared_ptr<const PassContext> context,
                       const std::shared_ptr<MetaDefProto>& meta_def,
                       onnxruntime::Model* model)
    : CustomOpImp(context, meta_def, model),
      kernel_name_topk_(PPGetPSKernelName(PP_TOPK)) {
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
  context_->create_kernel(kernel_name_topk_, kernel_name_topk_);

  LOG_THIS(1) << "Kernel Name: " << kernel_name_topk_;

  // Get device/kernel
  auto device = attrs->get_attr<xrt::device*>("xrt_device");
  auto k_topk = attrs->get_attr<xrt::kernel*>(kernel_name_topk_.c_str());

  // Create compute kernel
  kernel_topk_ =
      std::make_unique<TopK>(*device, *k_topk, input_shape_, output_shape_);

  // Create Sub BO
  // 32kb alignment
  uint32_t alignment = 32 << 10;
  auto sub_bo_size_topk =
      ((kernel_topk_->get_instr_size() + (alignment - 1)) / alignment) *
      alignment;

  //  get sub-bo offset
  size_t offset = 0;
  if (attrs->has_attr("bo_offset"))
    offset = attrs->get_attr<size_t>("bo_offset");

  LOG_THIS(1) << "SRAM BO Start offset: " << offset;

  auto bo_sram = attrs->get_attr<xrt::bo*>("bo_sram");
  // create sub-bo for kernels
  instr_bo_topk_ = xrt::bo(*bo_sram, sub_bo_size_topk, offset);
  offset += sub_bo_size_topk;

  LOG_THIS(1) << "SRAM BO End offset: " << offset;

  // update offset for sub-bo creation
  attrs->set_attr<size_t>("bo_offset", offset);

  // Sync Instructions
  kernel_topk_->sync_instructions(instr_bo_topk_);
}

MyCustomOp::~MyCustomOp() {}

void MyCustomOp::Compute(const OrtApi* api, OrtKernelContext* context) const {
  if (Ort::Global<void>::api_ == nullptr) {
    Ort::Global<void>::api_ = api;
  }
  __TIC__(TopKCompute);
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
  CHECK_EQ(num_outputs, 2u);

  auto input_tensor = ctx.GetInput(0);
  auto input_raw = input_tensor.GetTensorData<uint16_t>();
  auto tensor_info = input_tensor.GetTensorTypeAndShapeInfo();
  auto element_num = tensor_info.GetElementCount();
  // Copy input data to BO
  auto input_host = kernel_topk_->get_host_buffer_in();
  memcpy((void*)input_host, (void*)input_raw, element_num);

  auto attrs = context_->get_attrs();
  // Run AIE Kernel (TopK)
  if (attrs->has_attr(kernel_name_topk_.c_str()) &&
      attrs->has_attr("bo_sram")) {
    auto bo_sram = const_cast<xrt::bo*>(&instr_bo_topk_);
    auto kernel = attrs->get_attr<xrt::kernel*>(kernel_name_topk_.c_str());
    kernel_topk_->exec(*kernel, *bo_sram);
  } else {
    throw std::runtime_error("Could not find kernel/BO");
  }

  int64_t out_size = 1;
  out_size = std::accumulate(output_shape_.begin(), output_shape_.end(),
                             out_size, std::multiplies());

  // Alignment needed for TOP-K output
  uint16_t alignment = 8;
  uint16_t align_out_size = static_cast<uint16_t>(
      ((out_size + (alignment - 1)) / alignment) * alignment);

  auto score_out = kernel_topk_->get_host_buffer_out();
  auto idx_out = score_out + align_out_size;

  auto tensor = ctx.GetOutput(0, output_shape_);
  auto output_raw0 = tensor.GetTensorMutableData<float>();
  auto tensor1 = ctx.GetOutput(1, output_shape_);
  auto output_raw1 = tensor1.GetTensorMutableData<int64_t>();

  for (int i = 0; i < out_size; ++i) {
    int temp = (int)((int)score_out[i] << 16);
    float* out_Bf = reinterpret_cast<float*>(&temp);
    output_raw0[i] = out_Bf[0];
    output_raw1[i] = idx_out[i];
  }
#ifdef DEBUG
  FILE* refp = nullptr;
  fopen_s(&refp, "topk.txt", "w");
  for (int i = 0; i < out_size; ++i) {
    fprintf(refp, "%d\n", idx_out[i]);
  }
  fclose(refp);
#endif

  __TOC__(TopKCompute);
}
} // namespace vaip_topk_custom_op
