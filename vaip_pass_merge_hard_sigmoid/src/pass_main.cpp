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

#include <glog/logging.h>

#include "vaip/vaip.hpp"
#include "vitis/ai/env_config.hpp"

// test case :  # 32
DEF_ENV_PARAM(DEBUG_MERGE_HARD_SIGMOID, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_MERGE_HARD_SIGMOID) >= n)
namespace {
using namespace vaip_core;
struct MergeHardSigmoid {
  std::unique_ptr<Rule> create_rule() {
    auto builder = PatternBuilder();
    std::shared_ptr<Pattern> pat_add_a = builder.wildcard();
    std::shared_ptr<Pattern> pat_add_b = builder.xir_const_op();
    std::shared_ptr<Pattern> pat_add =
        builder.node2("com.xilinx:add", {pat_add_a, pat_add_b});
    std::shared_ptr<Pattern> pat_relu6 =
        builder.node2("com.xilinx:relu6", {pat_add});
    std::shared_ptr<Pattern> pat_mul_b = builder.xir_const_op();
    std::shared_ptr<Pattern> pat_mul =
        builder.node2("com.xilinx:mul", {pat_relu6, pat_mul_b});
    return Rule::create_rule(
        pat_mul, [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
          auto add_a = binder[pat_add_a->get_id()];
          auto add_b = binder[pat_add_b->get_id()];
          auto add = binder[pat_add->get_id()];
          auto relu6 = binder[pat_relu6->get_id()];
          auto mul_b = binder[pat_mul_b->get_id()];
          auto mul = binder[pat_mul->get_id()];
          CHECK(add_a.node != nullptr);
          CHECK(add_b.node != nullptr);
          CHECK(add.node != nullptr);
          CHECK(relu6.node != nullptr);
          CHECK(mul_b.node != nullptr);
          CHECK(mul.node != nullptr);

          auto add_end = self_.get_const_data<float>(*add_b.node);

          // hard-sigmoid = relu6 ( x + 3) / 6, so add_end == 3 is matched
          if (add_end[0] != 3) {
            return false;
          }

          auto& hard_sigmoid = NodeBuilder(*graph, self_)
                                   .set_input_nodes({add_a.node})
                                   .set_op_type("hard_sigmoid")
                                   .clone_data_type(*relu6.node)
                                   .clone_shape(*relu6.node)
                                   .clone_attrs(*relu6.node)
                                   .set_anchor_point1(*relu6.node)
                                   .build();

          auto mul_end = self_.get_const_data<float>(*mul_b.node)[0] * 6.0f;
          MY_LOG(1) << "mul_end: " << mul_end;

          auto& new_mul_b = NodeBuilder(*graph, self_)
                                .clone_node(*mul_b.node)
                                .set_anchor_point2(*mul_b.node_arg, {"const"})
                                .build();
          self_.create_const(
              new_mul_b, {reinterpret_cast<char*>(&mul_end), sizeof(mul_end)});

          NodeBuilder(*graph, self_)
              .set_input_nodes({&hard_sigmoid, &new_mul_b})
              .clone_op_type(*mul.node)
              .clone_data_type(*mul.node)
              .clone_shape(*mul.node)
              .clone_attrs(*mul.node)
              .set_anchor_point1(*mul.node)
              .build();

          graph_resolve(*graph, true);

          return true;
        });
  }

  MergeHardSigmoid(IPass& self) : self_{self} {}
  void process(IPass& self, Graph& graph) { create_rule()->apply(&graph); }

private:
  IPass& self_;
};
} // namespace

DEFINE_VAIP_PASS(MergeHardSigmoid, vaip_pass_merge_hard_sigmoid)
