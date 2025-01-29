/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

#include "vaip/vaip.hpp"
#include "vitis/ai/env_config.hpp"
#include <glog/logging.h>
namespace {
using namespace vaip_core;
struct MergeFix {
  MergeFix(IPass& self) : self_{self} {}
  static std::unique_ptr<Rule> create_rule(IPass* self) {
    auto builder = PatternBuilder();
    std::shared_ptr<Pattern> pat_float2fix =
        builder.node2("com.xilinx:float2fix", {builder.wildcard()});
    std::shared_ptr<Pattern> pat_fix2float =
        builder.node2("com.xilinx:fix2float", {pat_float2fix});
    return Rule::create_rule(
        pat_fix2float,
        [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
          auto ni_float2fix = binder[pat_float2fix->get_id()];
          auto ni_fix2float = binder[pat_fix2float->get_id()];
          NodeBuilder(*graph, *self)
              .clone_inputs(*ni_float2fix.node)
              .set_op_type("fix")
              .clone_attrs(*ni_float2fix.node)
              .set_anchor_point1(*ni_fix2float.node)
              .build();
          return true;
        });
  }
  void process(IPass& self, Graph& graph) { create_rule(&self)->apply(&graph); }

public:
  IPass& self_;
};
} // namespace

DEFINE_VAIP_PASS(MergeFix, vaip_pass_merge_fix)
