/*
 *     The Xilinx Vitis AI Vaip in this distribution are provided under the
 * following free and permissive binary-only license, but are not provided in
 * source code form.  While the following free and permissive license is similar
 * to the BSD open source license, it is NOT the BSD open source license nor
 * other OSI-approved open source license.
 *
 *      Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights
 * reserved.
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
#include <glog/logging.h>
#include <sstream>

#include "custom_op.hpp"
#include "softmax.hpp"
#include "vitis/ai/profiling.hpp"
#include "xf_aie_host_utils.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>

DEF_ENV_PARAM(DEBUG_SOFTMAX_CUSTOM_OP, "0")
DEF_ENV_PARAM_2(XLNX_VART_FIRMWARE, "", std::string)
#define LOG_THIS(n) LOG_IF(INFO, ENV_PARAM(DEBUG_SOFTMAX_CUSTOM_OP) >= n)

namespace vaip_softmax_custom_op {

MyCustomOp::MyCustomOp(std::shared_ptr<const PassContext> context,
                       const std::shared_ptr<MetaDefProto>& meta_def,
                       onnxruntime::Model* model)
    : CustomOpImp(context, meta_def, model),
      kernel_name_softmax_(PPGetPSKernelName(PP_SOFTMAX)) {
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
  /* if (meta_def_->generic_param().contains("size")) {
    std::string val = meta_def_->generic_param().at("size");
    target_shape_ = string_to_shape(val);
    LOG_THIS(1) << "out size: " << val;
  } else {
    LOG_THIS(1) << "No attribute \"size\"\n";
  }*/

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
  context_->create_kernel(kernel_name_softmax_, kernel_name_softmax_);

  LOG_THIS(1) << "Kernel Name: " << kernel_name_softmax_;

  // Get device/kernel
  auto device = attrs->get_attr<xrt::device*>("xrt_device");
  auto k_softmax = attrs->get_attr<xrt::kernel*>(kernel_name_softmax_.c_str());

  // Create compute kernel
  kernel_softmax_ = std::make_unique<SoftMax>(*device, *k_softmax, input_shape_,
                                              output_shape_);

  // Create Sub BO
  // 32kb alignment
  uint32_t alignment = 32 << 10;
  auto sub_bo_size_softmax =
      ((kernel_softmax_->get_instr_size() + (alignment - 1)) / alignment) *
      alignment;

  size_t offset = 0;
  if (attrs->has_attr("bo_offset"))
    offset = attrs->get_attr<size_t>("bo_offset");

  LOG_THIS(1) << "SRAM BO Start offset: " << offset;

  auto bo_sram = attrs->get_attr<xrt::bo*>("bo_sram");
  // create sub-bo for kernels
  instr_bo_softmax_ = xrt::bo(*bo_sram, sub_bo_size_softmax, offset);
  offset += sub_bo_size_softmax;

  // instr_bo_norm_ = xrt::bo(*bo_sram, sub_bo_size_nrm, offset);
  // offset += sub_bo_size_nrm;

  LOG_THIS(1) << "SRAM BO End offset: " << offset;

  // update offset for sub-bo creation
  attrs->set_attr<size_t>("bo_offset", offset);

  // set attr for norm output precision bits
  // attrs->set_attr<int>("norm_out_fl", fl_bits_[2]);

  // Sync Instructions
  kernel_softmax_->sync_instructions(instr_bo_softmax_);
  // kernel_norm_->sync_instructions(instr_bo_norm_);
}

MyCustomOp::~MyCustomOp() {}

void MyCustomOp::Compute(const OrtApi* api, OrtKernelContext* context) const {
  if (Ort::Global<void>::api_ == nullptr) {
    Ort::Global<void>::api_ = api;
  }
  __TIC__(SoftMaxCompute);
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
  auto input_raw = input_tensor.GetTensorData<uint16_t>();
  auto tensor_info = input_tensor.GetTensorTypeAndShapeInfo();
  auto element_num = tensor_info.GetElementCount();
  // Copy input data to BO
  auto input_host = kernel_softmax_->get_host_buffer_in();
  memcpy((void*)input_host, (void*)input_raw, element_num);

  auto attrs = context_->get_attrs();
  // Run AIE Kernel (SoftMax)
  if (attrs->has_attr(kernel_name_softmax_.c_str()) &&
      attrs->has_attr("bo_sram")) {
    auto bo_sram = const_cast<xrt::bo*>(&instr_bo_softmax_);
    auto kernel = attrs->get_attr<xrt::kernel*>(kernel_name_softmax_.c_str());
    kernel_softmax_->exec(*kernel, *bo_sram);
  } else {
    throw std::runtime_error("Could not find kernel/BO");
  }

#ifdef DEBUG
  auto out = kernel_softmax_->get_host_buffer_out1();
  FILE* refp = nullptr;
  fopen_s(&refp, "softmax.txt", "w");
  for (int i = 0; i < target_shape_[0] * target_shape_[1] * target_shape_[2];
       ++i) {
    fprintf(refp, "%x\n", reout[i]);
  }
  fclose(refp);
#endif

  __TOC__(SoftMaxCompute);
}
} // namespace vaip_softmax_custom_op
