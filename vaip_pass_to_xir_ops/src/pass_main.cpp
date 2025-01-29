/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

#include <glog/logging.h>

#include "./to_xir_ops_pass.hpp"

#include "vaip/pass.hpp"
#include "vaip/vaip.hpp"
#include "vitis/ai/env_config.hpp"
using namespace vaip_core;
namespace {
struct ToXir {
  ToXir(IPass& pass) {}
  void process(IPass& self, Graph& graph) {
    vaip_pass_to_xir_ops::to_xir_ops_pass(self, graph);
  }
};
} // namespace

DEFINE_VAIP_PASS(ToXir, vaip_pass_to_xir_ops)
