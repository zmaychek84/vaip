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
DEF_ENV_PARAM(DEBUG_UNUSED_NODES, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_UNUSED_NODES) >= n)

using namespace vaip_core;

struct RemoveIsolatedNode {
  RemoveIsolatedNode(IPass& self) {}
  void process(IPass& self, Graph& graph) {
    std::vector<const Node*> leaf_nodes;
    auto& args = self.get_pass_proto().args();
    if (!args.empty()) {
      // for debugging purpose.
      std::vector<const NodeArg*> outputs;
      outputs.reserve(args.size());
      for (auto& arg : args) {
        auto node_arg = VAIP_ORT_API(graph_get_node_arg)(graph, arg);
        CHECK(node_arg != nullptr) << "cannot find node arg: " << arg;
        outputs.push_back(node_arg);
      }
      VAIP_ORT_API(graph_set_outputs)(graph, outputs);
    }
    auto all_nodes = graph_nodes(graph);
    auto graph_outputs = graph_get_outputs(graph);
    leaf_nodes.reserve(graph_outputs.size());
    for (auto n : all_nodes) {
      CHECK(n != nullptr);
      auto node_outputs = node_get_output_node_args(*n);
      auto found = std::any_of(node_outputs.begin(), node_outputs.end(),
                               [&graph_outputs](const NodeArg* x) {
                                 return std::find(graph_outputs.begin(),
                                                  graph_outputs.end(),
                                                  x) != graph_outputs.end();
                               });
      if (found) {
        leaf_nodes.push_back(n);
      }
    }
    VAIP_ORT_API(graph_reverse_dfs_from)
    (
        graph,      //
        leaf_nodes, //
        nullptr,    //
        [&all_nodes](const Node* n) mutable {
          all_nodes.erase(std::remove(all_nodes.begin(), all_nodes.end(), n),
                          all_nodes.end());
        }, //
        nullptr);
    MY_LOG(1) << "prepare to remove " << all_nodes.size() << " nodes";
    for (auto n : all_nodes) {
      MY_LOG(1) << "\tremove " << node_as_string(*n);
      VAIP_ORT_API(graph_remove_node)(graph, {n, nullptr});
    }
  }
};

DEFINE_VAIP_PASS(RemoveIsolatedNode, vaip_pass_remove_isolated_node)
