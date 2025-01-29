/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

#include "vaip/vaip.hpp"
#include "vitis/ai/env_config.hpp"
#include <glog/logging.h>
DEF_ENV_PARAM(DEBUG_MERGE_INPUT_TRANSPOSE, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_MERGE_INPUT_TRANSPOSE) >= n)
// testcase 43
namespace {
using namespace vaip_core;
struct MergeInputTranspose {
  MergeInputTranspose(IPass& self) : self_{self} {}
  void preprocess(IPass& self, Graph& graph) {
    auto graph_inputs = graph_get_inputs(graph);
    for (auto& graph_input : graph_inputs) {
      auto consumers =
          graph_get_consumer_nodes(graph, node_arg_get_name(*graph_input));
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
          all_consumer_are_transpose = all_consumer_are_transpose &&
                                       node_get_attr_ints(*consumer, "order") ==
                                           gsl::span<const int64_t>(order);
        }
      }
      if (all_consumer_are_transpose) {
        map_[graph_input] = consumers;
      }
    }
  }

  void process(const IPass& self, Graph& graph) {
    for (auto& m : map_) {
      auto graph_input_node_arg = m.first;
      auto transpose_nodes = m.second;
      for (auto transpose_node : transpose_nodes) {
        CHECK(graph_input_node_arg != nullptr)
            << "cannot find graph's input. "
            << "graph_input " << node_arg_as_string(*graph_input_node_arg)
            << " " //
            ;
        remove_transpose_node(graph, transpose_node, *graph_input_node_arg);
        VAIP_ORT_API(node_arg_set_shape_i64)
        (*graph_input_node_arg, node_get_output_shape(*transpose_node, 0));
      }
    }
  }

  bool remove_transpose_node(Graph& graph, const Node* transpose_node,
                             const NodeArg& graph_input) {
    auto transpose_consumers =
        graph_get_consumer_nodes(graph, node_get_output_name(*transpose_node));
    for (auto node : transpose_consumers) {
      auto changed = false;
      auto node_inputs = node_get_inputs(*node);
      auto node_input_args = std::vector<const NodeArg*>();
      node_input_args.resize(node_inputs.size());
      auto index = 0u;
      for (auto ni : node_inputs) {
        node_input_args[index] = ni.node_arg;
        if (ni.node == transpose_node) {
          node_input_args[index] = &graph_input;
          changed = true;
        }
        index = index + 1;
      }
      CHECK(changed) << "cannot find input arg index.";
      NodeBuilder(graph, self_)
          .set_input_node_args(node_input_args)
          .clone_op_type(*node)
          .clone_attrs(*node)
          .set_anchor_point1(*node)
          .build();
    }
    return true;
  }
  IPass& self_;
  std::map<const NodeArg*, std::vector<const Node*>> map_;
};
} // namespace

DEFINE_VAIP_PASS(MergeInputTranspose, vaip_pass_merge_input_transpose)
