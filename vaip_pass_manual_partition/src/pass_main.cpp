/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

#include <glog/logging.h>
//
#include "vaip/vaip.hpp"
#include "vitis/ai/env_config.hpp"

DEF_ENV_PARAM(DEBUG_MANUAL_PARTITION, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_MANUAL_PARTITION) >= n)
namespace {
using namespace vaip_core;
template <typename T>
static std::vector<const Node*> map_name_to_node(const Graph& graph,
                                                 const T& names) {
  auto ret = std::vector<const Node*>{};
  for (auto& name : names) {
    auto node = VAIP_ORT_API(graph_producer_node)(graph, name);
    CHECK(node != nullptr) << "cannot find : " << name;
    ret.push_back(node);
  }
  return ret;
}

template <typename T>
static bool all_exist_ops(const Graph& graph, const T& names) {
  auto ret = true;
  for (auto& name : names) {
    auto node = VAIP_ORT_API(graph_producer_node)(graph, name);
    if (node == nullptr) {
      MY_LOG(1) << "cannot find node: " << name;
      ret = false;
    }
  }
  return ret;
}

bool in_vector(const Node* node, const std::vector<const Node*> nodes) {
  return std::find(nodes.begin(), nodes.end(), node) != nodes.end();
}

struct ManualPartition {
  ManualPartition(IPass& self) : self_{self} {}
  void preprocess(IPass& self, Graph& graph) {
    auto& manual_partition_proto = self.get_pass_proto().manual_partition();
    auto& to_ops = manual_partition_proto.to_ops();
    auto& from_op_names = manual_partition_proto.from_ops();
    if (!all_exist_ops(graph, to_ops) || !all_exist_ops(graph, from_op_names)) {
      LOG(INFO) << "cancel manual partition pass, config to_ops or from_ops "
                   "can not find";
      return;
    }

    auto leaf_nodes = std::vector<const Node*>{};
    if (to_ops.empty()) {
      leaf_nodes = graph_get_output_nodes(graph);
    } else {
      leaf_nodes = map_name_to_node(graph, to_ops);
    }
    auto from_ops = map_name_to_node(graph, from_op_names);

    VAIP_ORT_API(graph_reverse_dfs_from)
    (
        graph,                                             //
        leaf_nodes,
        nullptr,                                           //
        [&](const Node* node) { nodes_.push_back(node); }, //
        [&](const Node* from, const Node* to) {
          return in_vector(to, from_ops);
        });
  }
  bool process(IPass& self, Graph& graph) {
    auto ret = false;
    for (auto node : nodes_) {
      if (node_is_op(*node, "const", "com.xilinx")) {
        continue;
      }
      auto outputs = node_get_output_node_args(*node);
      if (outputs.size() > 1) {
        continue;
      }
      if (node_arg_is_unknown_shape(*outputs[0])) {
        continue;
      }
      MY_LOG(1) << "to be change to unknown op : " << node_as_string(*node);
      auto shape = node_get_output_shape(*node, 0);
      NodeBuilder(graph, self)
          .clone_inputs(*node)
          .set_op_type("unknown")
          .clone_data_type(*node)
          .set_shape(shape)
          .set_anchor_point1(*node)
          .build();
      ret = true;
    }
    return ret;
  }

public:
  IPass& self_;
  std::vector<const Node*> nodes_;
};

} // namespace

DEFINE_VAIP_PASS(ManualPartition, vaip_pass_manual_partition)
