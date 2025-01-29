/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

#include <glog/logging.h>

#include "./const_fold_transpose.hpp"
#include "vitis/ai/env_config.hpp"
using namespace vaip_core;
namespace {
struct ConstFoldTranspose {
  ConstFoldTranspose(IPass& pass) {}
  void process(IPass& self, Graph& graph) {
    vaip_pass_const_fold_transpose::ConstFoldTransposeRule().apply(&graph);
  };
};
} // namespace

DEFINE_VAIP_PASS(ConstFoldTranspose, vaip_pass_const_fold_transpose)
