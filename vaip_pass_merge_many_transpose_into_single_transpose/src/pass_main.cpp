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
#include <unordered_map>
DEF_ENV_PARAM(DEBUG_MERGE_MANY_TRANSPOSE_INTO_SINGLE_TRANSPOSE, "0")
#define MY_LOG(n)                                                              \
  LOG_IF(INFO, ENV_PARAM(DEBUG_MERGE_MANY_TRANSPOSE_INTO_SINGLE_TRANSPOSE) >= n)
namespace {
using namespace vaip_core;
// test case: model #1
struct MergeManyTransposeIntoSingleTranspose {
  MergeManyTransposeIntoSingleTranspose(IPass& self) {}
  void preprocess(IPass& self, Graph& graph) {
    for (auto node_idx : graph_get_node_in_topoligical_order(graph)) {
      auto node_ptr = VAIP_ORT_API(graph_get_node)(graph, node_idx);
      CHECK(node_ptr != nullptr) << "node_idx " << node_idx << " ";
      auto& node = *node_ptr;
      auto output_args = node_get_output_node_args(node);
      for (auto output_arg : output_args) {
        auto output_name = node_arg_get_name(*output_arg);
        auto consumers = graph_get_consumer_nodes(graph, output_name);
        auto all_consumer_are_transpose = true;
        auto order = std::vector<int64_t>();
        for (auto consumer : consumers) {
          all_consumer_are_transpose =
              all_consumer_are_transpose &&
              node_is_op(*consumer, "transpose", "com.xilinx");
          if (!all_consumer_are_transpose) {
            break;
          }
          if (order.empty()) {
            auto first_order = node_get_attr_ints(*consumer, "order");
            order.assign(first_order.begin(), first_order.end());
          } else {
            // all inputs must have same transpose(order:=...);
            all_consumer_are_transpose =
                all_consumer_are_transpose &&
                node_get_attr_ints(*consumer, "order") ==
                    gsl::span<const int64_t>(order);
          }
        }
        if (all_consumer_are_transpose && consumers.size() >= 2) {
          for (auto i = 1u; i < consumers.size(); ++i) {
            map_[consumers[i]] = consumers[0];
          }
        }
      }
    }
  }
  // now remove reduntant transpose
  bool process(IPass& self, Graph& graph, const Node& node) {
    auto node_inputs = node_get_inputs(node);
    auto changed = false;
    auto node_input_args = std::vector<const NodeArg*>();
    node_input_args.reserve(node_inputs.size());
    for (auto ni : node_inputs) {
      auto new_input_arg = ni.node_arg;
      if (ni.node) {
        auto it = map_.find(ni.node);
        if (it != map_.end()) {
          new_input_arg = &(node_get_output_node_arg(*it->second));
          changed = true;
        }
      }
      node_input_args.push_back(new_input_arg);
    }
    if (changed) {
      NodeBuilder(graph, self)
          .set_input_node_args(node_input_args)
          .clone_op_type(node)
          .clone_attrs(node)
          .set_anchor_point1(node)
          .build();
    }
    return changed;
  }
  // provate variables.
  std::unordered_map<const Node*, const Node*> map_;
};
} // namespace

DEFINE_VAIP_PASS(MergeManyTransposeIntoSingleTranspose,
                 vaip_pass_merge_many_transpose_into_single_transpose)
