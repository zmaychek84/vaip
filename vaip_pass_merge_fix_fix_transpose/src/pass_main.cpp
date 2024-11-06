/*
 *     The Xilinx Vitis AI Vaip in this distribution are provided under the
 * following free and permissive binary-only license, but are not provided in
 * source code form.  While the following free and permissive license is similar
 * to the BSD open source license, it is NOT the BSD open source license nor
 * other OSI-approved open source license.
 *
 *      Copyright (C) 2022 Xilinx, Inc. All rights reserved.
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

#include "vaip/vaip.hpp"
#include "vitis/ai/env_config.hpp"
#include <glog/logging.h>
namespace {
using namespace vaip_core;
struct MergeFixFixTranspose {
  MergeFixFixTranspose(IPass& self) : self_{self} {}
  bool check_order(const NodeInput& ni) {
    return node_get_attr_ints(*ni.node, "order") ==
           gsl::span<const int64_t>{std::vector<int64_t>{0, 1, 2, 3}};
  }
  std::unique_ptr<Rule> create_rule_FF() {
    auto builder = PatternBuilder();
    auto pat_fix1 = builder.node2("com.xilinx:fix", {builder.wildcard()});
    auto pat_fix2 = builder.node2("com.xilinx:fix", {pat_fix1});
    auto pat_transpose = builder.node2("com.xilinx:transpose", {pat_fix2});
    return Rule::create_rule(
        pat_transpose,
        [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
          auto ni_transpose = binder[pat_transpose->get_id()];
          if (!check_order(ni_transpose))
            return false;
          auto ni_fix1 = binder[pat_fix1->get_id()];
          NodeBuilder(*graph, self_)
              .clone_inputs(*ni_fix1.node)
              .set_op_type("fix")
              .clone_attrs(*ni_fix1.node)
              .set_anchor_point1(*ni_transpose.node)
              .build();
          return true;
        });
  }
  std::unique_ptr<Rule> create_rule_FF2() {
    auto builder = PatternBuilder();
    auto pat_fix1 = builder.node2("com.xilinx:fix", {builder.wildcard()});
    auto pat_fix2 = builder.node2("com.xilinx:float2fix", {pat_fix1});
    auto pat_fix3 = builder.node2("com.xilinx:fix2float", {pat_fix2});
    auto pat_transpose = builder.node2("com.xilinx:transpose", {pat_fix3});
    return Rule::create_rule(
        pat_transpose,
        [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
          auto ni_transpose = binder[pat_transpose->get_id()];
          if (!check_order(ni_transpose))
            return false;
          auto ni_fix1 = binder[pat_fix1->get_id()];
          auto ni_fix3 = binder[pat_fix3->get_id()];
          NodeBuilder(*graph, self_)
              .clone_inputs(*ni_fix1.node)
              .set_op_type("fix")
              .clone_attrs(*ni_fix3.node)
              .set_anchor_point1(*ni_transpose.node)
              .build();
          return true;
        });
  }
  std::unique_ptr<Rule> create_rule_FF3() {
    auto builder = PatternBuilder();
    auto pat_input = builder.wildcard();
    auto pat_scale = builder.wildcard();
    auto pat_zp = builder.wildcard();
    auto pat_q = builder.node2("com.xilinx:quantize_linear",
                               {pat_input, pat_scale, pat_zp});
    auto pat_dq = builder.node2("com.xilinx:dequantize_linear",
                                {pat_q, pat_scale, pat_zp});
    auto pat_q1 = builder.node2("com.xilinx:quantize_linear",
                                {pat_dq, pat_scale, pat_zp});
    auto pat_dq1 = builder.node2("com.xilinx:dequantize_linear",
                                 {pat_q1, pat_scale, pat_zp});
    auto pat_transpose = builder.node2("com.xilinx:transpose", {pat_dq1});
    return Rule::create_rule(
        pat_transpose,
        [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
          auto ni_transpose = binder[pat_transpose->get_id()];
          if (!check_order(ni_transpose))
            return false;
          auto ni_q = binder[pat_q->get_id()];
          auto ni_q1 = binder[pat_q1->get_id()];
          auto ni_dq1 = binder[pat_dq1->get_id()];
          const auto& new_q1 = NodeBuilder(*graph, self_)
                                   .clone_inputs(*ni_q.node)
                                   .set_op_type("quantize_linear")
                                   .clone_attrs(*ni_q1.node)
                                   .set_anchor_point1(*ni_q1.node)
                                   .build();
          std::vector<const NodeArg*> inputs{
              &node_get_output_node_arg(new_q1),
              binder[pat_scale->get_id()].node_arg,
              binder[pat_zp->get_id()].node_arg};
          NodeBuilder(*graph, self_)
              .set_input_node_args(inputs)
              .set_op_type("dequantize_linear")
              .clone_attrs(*ni_dq1.node)
              .set_anchor_point1(*ni_transpose.node)
              .build();
          return true;
        });
  }
  void process(IPass& self, Graph& graph) {
    std::vector<std::unique_ptr<BaseRule>> rules;
    rules.push_back(create_rule_FF());
    rules.push_back(create_rule_FF2());
    rules.push_back(create_rule_FF3());
    BaseRule::create_rule_chain(std::move(rules))->apply(&graph);
  }

public:
  IPass& self_;
};
} // namespace

DEFINE_VAIP_PASS(MergeFixFixTranspose, vaip_pass_merge_fix_fix_transpose)
