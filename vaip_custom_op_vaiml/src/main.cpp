/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

#include <glog/logging.h>

#include "GT/GT_1_2/custom_op_gt_1_2.hpp"
#include "GT/GT_1_3/custom_op_gt_1_3.hpp"
#include "HT_1_2/custom_op_ht_1_2.hpp"
#include "vaip/vaip.hpp"
#include <iostream>

static vaip_core::ExecutionProvider* create_execution_provider_imp(
    std::shared_ptr<const vaip_core::PassContext> context,
    const vaip_core::MetaDefProto& meta_def) {

  auto& session_option = context->get_config_proto().provider_options();
  // FIXME: fix this section of xclbin path patch once preemption mode is in
  // default.
  {
    auto* mutable_session =
        const_cast<vaip_core::ConfigProto&>(context->get_config_proto())
            .mutable_provider_options();
    std::string xclbin_path = session_option.at("xclbin");
    std::string preempt_xclbin = "4x4_gt_ht_03";
    std::string non_preempt_xclbin = "4x4_gt_ht_04";
    std::size_t found = xclbin_path.find(non_preempt_xclbin);
    if (!context->get_provider_option("enable_preemption").has_value()) {
      (*mutable_session)["enable_preemption"] = "0";
    }
    if ((session_option.at("enable_preemption") == "1") &&
        found != std::string::npos) {
      xclbin_path.replace(found, non_preempt_xclbin.size(),
                          preempt_xclbin); // replace to preempt xclbin
    }
    (*mutable_session)["xclbin"] = xclbin_path;
  }

  auto model_version_ = session_option.at("model_name");
  if (model_version_ == "GT_v1.2") {
    return new vaip_core::ExecutionProviderImp<
        vaip_vaiml_custom_op::MyCustomOpGT1_2>(context, meta_def);
  } else if (model_version_ == "HT_v1.2") {
    return new vaip_core::ExecutionProviderImp<
        vaip_vaiml_custom_op::MyCustomOpHT1_2>(context, meta_def);
  } else if (model_version_ == "GT_v1.3" || model_version_ == "GTC_v1.0") {
    return new vaip_core::ExecutionProviderImp<
        vaip_vaiml_custom_op::MyCustomOpGT1_3>(context, meta_def);
  } else {
    throw std::invalid_argument("Unknown model");
  }
}

#if VAIP_CUSTOM_OP_VAIML_USE_DLL == 0
#  include <vitis/ai/plugin.hpp>
namespace {
static vitis::ai::StaticPluginRegister
    __register("vaip_custom_op_VAIML", "create_execution_provider",
               (void*)&create_execution_provider_imp);
} // namespace
extern "C" {
void* vaip_custom_op_vaiml__hook = &__register;
}
#else

extern "C" VAIP_PASS_ENTRY vaip_core::ExecutionProvider*
create_execution_provider(std::shared_ptr<const vaip_core::PassContext> context,
                          const onnxruntime::Graph* graph,
                          const vaip_core::MetaDefProto& meta_def) {
  return create_execution_provider_imp(context, graph, meta_def);
}
extern "C" {
void* vaip_custom_op_vaiml__hook = nullptr;
}
#endif
