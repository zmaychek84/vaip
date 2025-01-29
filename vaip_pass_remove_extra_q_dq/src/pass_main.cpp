/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

#include <glog/logging.h>
#include <iostream>

#include "vaip/vaip.hpp"
#include "vitis/ai/env_config.hpp"

// test case :  #43
namespace {
using namespace vaip_core;
struct RemoveExtraQDq {
  std::unique_ptr<Rule> create_rule() {
    auto builder = PatternBuilder();
    std::shared_ptr<Pattern> gi_float2fix =
        builder.node2("com.xilinx:float2fix", {builder.graph_input()});
    std::shared_ptr<Pattern> gi_fix2float =
        builder.node2("com.xilinx:fix2float", {gi_float2fix});
    std::shared_ptr<Pattern> trans =
        builder.node2("com.xilinx:transpose", {gi_fix2float});
    std::shared_ptr<Pattern> trans_float2fix =
        builder.node2("com.xilinx:float2fix", {trans});
    std::shared_ptr<Pattern> trans_fix2float =
        builder.node2("com.xilinx:fix2float", {trans_float2fix});
    return Rule::create_rule(
        trans_fix2float,
        [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
          auto trans_ni = binder[trans->get_id()];
          auto fix2float_ni = binder[trans_fix2float->get_id()];
          CHECK(trans_ni.node != nullptr);
          CHECK(fix2float_ni.node != nullptr);

          graph_replace_node_arg(*graph, self_, *fix2float_ni.node_arg,
                                 *trans_ni.node_arg);

          graph_resolve(*graph, true);

          return true;
        });
  }

  RemoveExtraQDq(IPass& self) : self_{self} {}
  void process(IPass& self, Graph& graph) { create_rule()->apply(&graph); }

private:
  IPass& self_;
};
} // namespace

DEFINE_VAIP_PASS(RemoveExtraQDq, vaip_pass_remove_extra_q_dq)
