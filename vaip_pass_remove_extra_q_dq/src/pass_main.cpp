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
