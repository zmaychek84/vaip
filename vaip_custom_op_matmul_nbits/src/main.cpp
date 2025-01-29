/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

/*

**/
#include <glog/logging.h>

#include "./custom_op.hpp"
#include "vaip/vaip.hpp"
static vaip_core::ExecutionProvider* create_execution_provider_imp(
    std::shared_ptr<const vaip_core::PassContext> context,
    const vaip_core::MetaDefProto& meta_def) {
  return new vaip_core::ExecutionProviderImp<
      vaip_matmul_nbits_custom_op::MyCustomOp>(context, meta_def);
}

#if VAIP_CUSTOM_OP_MATMULNBITS_USE_DLL == 0
#  include <vitis/ai/plugin.hpp>
namespace {
static vitis::ai::StaticPluginRegister
    __register("vaip_custom_op_MATMULNBITS", "create_execution_provider",
               (void*)&create_execution_provider_imp);
} // namespace
extern "C" {
void* vaip_custom_op_matmul_nbits__hook = &__register;
}
#else

extern "C" VAIP_PASS_ENTRY vaip_core::ExecutionProvider*
create_execution_provider(std::shared_ptr<const vaip_core::PassContext> context,

                          const vaip_core::MetaDefProto& meta_def) {
  return create_execution_provider_imp(context, meta_def);
}
extern "C" {
void* vaip_custom_op_matmul_nbits__hook = nullptr;
}
#endif
