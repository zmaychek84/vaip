/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

#pragma once
#include "onnxruntime_api.hpp"
#include <set>
#include <vector>

namespace vaip_core {
struct OpDefInfo {
  void (*get_domains)(std::vector<Ort::CustomOpDomain>& domains);
};

template <typename T> struct ProcessorOpDefInfo {
  static OpDefInfo* op_fef_info() { return &info; }
  static OpDefInfo info;
};

template <typename T> OpDefInfo ProcessorOpDefInfo<T>::info = {T::process};
void set_vitis_ep_custom_ops(const std::set<std::string>&);
} // namespace vaip_core

#ifndef _WIN32
#  include <vaip/export.h>
#  define DEFINE_VAIP_OPDEF(cls, id)                                           \
    extern "C" VAIP_PASS_ENTRY vaip_core::OpDefInfo* vaip_op_def_info() {      \
      return ProcessorOpDefInfo<cls>::op_fef_info();                           \
    }                                                                          \
    extern "C" {                                                               \
    void* /* a hook var*/ id##__hook = nullptr;                                \
    }
#else
#  include <vitis/ai/plugin.hpp>
#  define DEFINE_VAIP_OPDEF(cls, id)                                           \
    static vaip_core::OpDefInfo* vaip_op_def_info() {                          \
      return ProcessorOpDefInfo<cls>::op_fef_info();                           \
    }                                                                          \
    namespace {                                                                \
    static vitis::ai::StaticPluginRegister                                     \
        __register(OUTPUT_NAME, "vaip_op_def_info", (void*)&vaip_op_def_info); \
    }                                                                          \
    extern "C" {                                                               \
    void* /* a hook var*/ id##__hook = &__register;                            \
    }
#endif
