/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

#include "vaip/vaip.hpp"
#include "vitis/ai/env_config.hpp"
#include <glog/logging.h>
DEF_ENV_PARAM(DEBUG_MERGE_CONSECUTIVE_FIX, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_MERGE_CONSECUTIVE_FIX) >= n)
namespace {
using namespace vaip_core;
struct MergeConsecutiveFix {
  std::unique_ptr<Rule> create_rule() {
    auto builder = PatternBuilder();
    std::shared_ptr<Pattern> pat_fix =
        builder.node2("com.xilinx:fix", {builder.wildcard()});
    std::shared_ptr<Pattern> pat_fix_fix =
        builder.node2("com.xilinx:fix", {pat_fix});
    return Rule::create_rule(
        pat_fix_fix, [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
          auto ni_fix = binder[pat_fix->get_id()];
          auto ni_fix_fix = binder[pat_fix_fix->get_id()];
          NodeBuilder(*graph, self_)
              .clone_node(*ni_fix.node)
              .set_anchor_point1(*ni_fix_fix.node)
              .build();
          PASS_LOG(self_, 100) << "hello log";
          return true;
        });
  }
  MergeConsecutiveFix(IPass& self) : self_{self} {}
  void process(IPass& self, Graph& graph) { create_rule()->apply(&graph); }
  IPass& self_;
};
} // namespace
DEFINE_VAIP_PASS(MergeConsecutiveFix, vaip_pass_merge_consecutive_fix)
