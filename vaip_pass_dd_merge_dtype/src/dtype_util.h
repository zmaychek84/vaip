/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */
#pragma once
#include "vaip/vaip.hpp"
#include "vitis/ai/env_config.hpp"

using namespace vaip_core;

namespace vaip::dtype_util {

[[maybe_unused]] static std::string summary(
    std::vector<std::pair<const std::string, const std::string>> nodetypes) {
  std::stringstream ss;
  ss << "[";
  for (const auto& item : nodetypes) {
    ss << "(" << item.first << "," << item.second << "),";
  }
  ss << "]";
  return ss.str();
}

[[maybe_unused]] static std::vector<
    std::pair<const std::string, const std::string>>
op_names(const std::vector<const Node*>& nodes) {
  std::vector<std::pair<const std::string, const std::string>> ret;
  std::transform(nodes.begin(), nodes.end(), std::back_inserter(ret),
                 [](const Node* node) {
                   return std::make_pair(VAIP_ORT_API(node_op_type)(*node),
                                         VAIP_ORT_API(node_op_domain)(*node));
                 });
  return ret;
}

struct NodeAttrContext {
  std::vector<const Node*> parent_ops;
  std::vector<const Node*> child_op;
  std::string precision;
  std::string node_op_type;
  std::vector<NodeInput> node_inputs;
  std::vector<const NodeArg*> node_outputs;
  const Graph* graph;
  const Node* node;
  std::string get_string_attribute(std::string attribute) {
    auto& attrs = node_get_attributes_ref(*node);
    auto shape_proto = node_attributes_get(attrs, attribute);
    return std::string(VAIP_ORT_API(attr_proto_get_string)(*shape_proto));
  }
  std::vector<int64_t> get_shape_from_attr(std::string attribute) {
    auto& attrs = node_get_attributes_ref(*node);
    auto shape_proto = node_attributes_get(attrs, attribute);
    auto shape = VAIP_ORT_API(attr_proto_get_ints)(*shape_proto);
    std::vector<int64_t> ret(shape.begin(), shape.end());
    return ret;
  }
  std::string node_name() { return VAIP_ORT_API(node_get_name)(*node); }
  std::vector<int64_t> get_input_shape_at(size_t index) {
    auto input_arg = node_inputs[index].node_arg;
    auto shape = node_arg_get_shape_i64(*input_arg);
    std::vector<int64_t> ret(shape.get()->begin(), shape.get()->end());
    return ret;
  }
  std::vector<int64_t> get_output_shape_at(size_t index) {
    auto output_arg = node_outputs[index];
    auto shape = node_arg_get_shape_i64(*output_arg);
    std::vector<int64_t> ret(shape.get()->begin(), shape.get()->end());
    return ret;
  }
};

static std::vector<const Node*> get_all_parent_nodes(const Node* cnode) {
  auto node_inputs = node_get_inputs(*cnode);
  std::vector<const Node*> ret;
  for (const auto& ni : node_inputs) {
    if (ni.node != nullptr) {
      ret.emplace_back(ni.node);
    }
  }
  return ret;
}

static std::vector<const Node*>
get_all_child_nodes(const onnxruntime::Graph& graph, const Node* node) {
  std::vector<const Node*> ret;
  for (const auto& output_arg : node_get_output_node_args(*node)) {
    std::string output_name = node_arg_get_name(*output_arg);
    std::vector<const onnxruntime::Node*> consumers =
        graph_get_consumer_nodes(graph, output_name);

    for (const auto consumer_node : consumers) {
      ret.push_back(consumer_node);
    }
  }
  return ret;
}

static NodeAttrContext build_context(const onnxruntime::Graph& graph,
                                     const Node* node, std::string precision) {
  auto parent_nodes = vaip::dtype_util::get_all_parent_nodes(node);

  auto child_nodes = vaip::dtype_util::get_all_child_nodes(graph, node);
  auto node_op = VAIP_ORT_API(node_op_type)(*node);
  auto node_inputs = node_get_inputs(*node);
  auto node_outputs = node_get_output_node_args(*node);

  NodeAttrContext ret = {parent_nodes, child_nodes,  precision, node_op,
                         node_inputs,  node_outputs, &graph,    node};
  return ret;
}
} // namespace vaip::dtype_util