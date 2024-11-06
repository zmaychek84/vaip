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

#include "./graph_output_add_node.hpp"

#include <vitis/ai/dim_calc.hpp>

#include "glog/logging.h"

#include "vitis/ai/env_config.hpp"

DEF_ENV_PARAM(DEBUG_GRAPH_OUTPUT_ADD_NODE, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_GRAPH_OUTPUT_ADD_NODE) >= n)
namespace vaip_pass_graph_output_add_node {

GraphOutputAddNodeRule::GraphOutputAddNodeRule() : Rule() {
  auto builder = PatternBuilder();
  output_ = builder.wildcard();
}

const Pattern* GraphOutputAddNodeRule::pattern() const { return output_.get(); }

bool GraphOutputAddNodeRule::action(onnxruntime::Graph* graph,
                                    binder_t& binder) const {
  auto output = binder[output_->get_id()];

  if (output.node != nullptr && output.node_arg != nullptr) {
    auto graph_outputs = graph_get_outputs(*graph);
    auto is_output =
        std::find(graph_outputs.begin(), graph_outputs.end(), output.node_arg);
    if (is_output != graph_outputs.end()) {
      auto output_node_arg_name = node_arg_get_name(*output.node_arg);
      auto consumers = graph_get_consumer_nodes(*graph, output_node_arg_name);

      MY_LOG(1) << "pattern.node_arg: " << node_arg_as_string(*output.node_arg);
      MY_LOG(1) << "consumers number: " << consumers.size();
      auto is_xilinx_domain = node_op_domain(*output.node) == "com.xilinx";
      if (consumers.size() >= 1 && is_xilinx_domain) {
        MY_LOG(1) << "do graph_output_add_node.";
        const std::string name = output_node_arg_name + std::string("_");
        auto& new_node_arg =
            VAIP_ORT_API(node_arg_clone)(*graph, *output.node_arg, name);
        graph_add_node(*graph, name, AnchorPoint::IDENTITY_OP,
                       "convert from GraphOutputAddNode pass.",
                       {output.node_arg}, {&new_node_arg},
                       NodeAttributesBuilder()
                           .add("data_type",
                                node_get_attr_string(*output.node, "data_type"))
                           .add("shape", node_get_output_shape(*output.node, 0))
                           .build(),
                       "com.xilinx");
      }
    }
  }
  return false;
}
} // namespace vaip_pass_graph_output_add_node
