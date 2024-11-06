/*
 *     The Xilinx Vitis AI Vaip in this distribution are provided under the
 * following free and permissive binary-only license, but are not provided in
 * source code form.  While the following free and permissive license is similar
 * to the BSD open source license, it is NOT the BSD open source license nor
 * other OSI-approved open source license.
 *
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
// test case 41 see issue #611 #626 for more detail
#include "vaip/vaip.hpp"
#include "vitis/ai/env_config.hpp"
#include <glog/logging.h>
DEF_ENV_PARAM(DEBUG_MERGE_DUPLICATED_FIX, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_MERGE_DUPLICATED_FIX) >= n)
namespace {
using namespace vaip_core;
struct MergeDuplicatedFix {
  MergeDuplicatedFix(IPass& self) : self_{self} {}
  void preprocess(IPass& self, Graph& graph) {
    auto nodes = graph_nodes(graph);
    for (auto& node : nodes) {
      auto node_args = node_get_output_node_args(*node);
      if (node_args.size() == 1) {
        auto consumers =
            graph_get_consumer_nodes(graph, node_arg_get_name(*node_args[0]));
        auto num_of_consumers = consumers.size();
        if (num_of_consumers >= 2) {
          const auto& first_consumer = *consumers[0];
          auto is_first_consumer_fix_op =
              node_is_op(first_consumer, "fix", "com.xilinx");
          if (is_first_consumer_fix_op) {
            auto fix_point = node_get_attr_int(first_consumer, "fix_point");
            auto all_consumer_are_fix = true;
            for (auto i = 1u; i < num_of_consumers; ++i) {
              const auto& rest_consumer = *consumers[i];
              all_consumer_are_fix =
                  all_consumer_are_fix &&
                  node_is_op(rest_consumer, "fix", "com.xilinx");
              all_consumer_are_fix =
                  all_consumer_are_fix &&
                  fix_point == node_get_attr_int(rest_consumer, "fix_point");
            }
            if (all_consumer_are_fix) {
              common_fix_nodes_.emplace_back(consumers);
            }
          }
        }
      }
    }
  }

  void process(const IPass& self, Graph& graph) {
    for (auto& m : common_fix_nodes_) {
      const auto& first_fix_node = *m[0];
      for (auto i = 1u; i < m.size(); ++i) {
        const auto& first_node_arg = node_get_output_node_arg(first_fix_node);
        MY_LOG(1) << "replace "
                  << node_arg_get_name(node_get_output_node_arg(*m[i]))
                  << " with " << node_arg_get_name(first_node_arg);
        graph_replace_node_arg(graph, self_,
                               node_get_output_node_arg(*m[i]) /*from*/,
                               node_get_output_node_arg(first_fix_node) /*to*/);
      }
    }
  }

  IPass& self_;
  std::vector<std::vector<const Node*>> common_fix_nodes_;
};
} // namespace
DEFINE_VAIP_PASS(MergeDuplicatedFix, vaip_pass_merge_duplicated_fix)
