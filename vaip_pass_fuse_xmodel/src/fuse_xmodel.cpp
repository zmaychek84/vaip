/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

#include "./fuse_xmodel.hpp"

#include <algorithm>
#include <glog/logging.h>
#include <set>
#include <vitis/ai/env_config.hpp>
DEF_ENV_PARAM(DEBUG_FUSE_XMODEL, "0")
DEF_ENV_PARAM_2(VAIP_FUSE_NODES, "", std::vector<int>)
DEF_ENV_PARAM(VAIP_ENABLE_TROUBLESHOOTING, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_FUSE_XMODEL) >= n)

namespace vaip_pass_fuse_xmodel {
static bool is_xilinx_op(const Node* node) {
  auto domain = VAIP_ORT_API(node_op_domain)(*node);
  auto ret = domain == "com.xilinx";
  if (ENV_PARAM(VAIP_ENABLE_TROUBLESHOOTING)) {
    auto node_index = VAIP_ORT_API(node_get_index)(*node);
    auto& param_nodes = ENV_PARAM(VAIP_FUSE_NODES);
    auto nodes = std::set<size_t>(param_nodes.begin(), param_nodes.end());
    ret = nodes.count(node_index) != 0;
  }
  return ret;
}

struct FusedXmodel {
  std::vector<const NodeArg*> input_node_args;
  std::vector<const Node*> boundary;
  std::vector<const NodeArg*> output_node_args;
  std::vector<int> mask;
  std::vector<const Node*> nodes;
  Graph* graph;
  std::vector<const NodeArg*> graph_outputs;

public:
  bool contains(const Node* node);
  void insert(const Node* node);
  void maybe_it_is_graph_output(const Node* down);
};

bool FusedXmodel::contains(const Node* node) {
  auto index = VAIP_ORT_API(node_get_index)(*node);
  return mask[index] != 0;
}

void FusedXmodel::insert(const Node* node) {
  auto index = VAIP_ORT_API(node_get_index)(*node);
  mask[index] = 1;
  nodes.push_back(node);
  MY_LOG(1) << "add node: " << node_as_string(*node) << " index= " << index
            << " nodes.size() = " << nodes.size();
}

void FusedXmodel::maybe_it_is_graph_output(const Node* node) {
  auto node_args = node_get_output_node_args(*node);
  for (auto node_arg : node_args) {
    auto it = std::find(graph_outputs.begin(), graph_outputs.end(), node_arg);
    auto is_graph_output = (it != graph_outputs.end());
    if (is_graph_output) {
      output_node_args.push_back(node_arg);
    }
  }
}
static bool fuse_node_rec_up(FusedXmodel& m,
                             const std::vector<const Node*>& boundary) {
  auto ret = false;
  for (auto up : boundary) {
    auto up_inputs = node_get_inputs(*up);
    for (auto& up_input : up_inputs) {
      if (up_input.node == nullptr) { // must be an graph inputs.
        m.input_node_args.push_back(up_input.node_arg);
        continue;
      }
      if (m.contains(up_input.node)) {
        // already processed.
        continue;
      }
      if (!is_xilinx_op(up_input.node)) {
        m.input_node_args.push_back(up_input.node_arg);
        continue;
      }
      // it is a xilinx op.
      m.boundary.push_back(up_input.node);
      m.insert(up_input.node);
      ret = true;
    }
  }
  return ret;
}

static const NodeArg* find_node_arg_between(const Node* from, const Node* to) {
  auto inputs = node_get_inputs(*to);
  for (auto input : inputs) {
    if (input.node == from) {
      return input.node_arg;
    }
  }
  return nullptr;
}

static bool fuse_node_rec_down(FusedXmodel& m,
                               const std::vector<const Node*>& boundary) {
  auto ret = false;
  for (auto down : boundary) {
    m.maybe_it_is_graph_output(down);
    auto consumers =
        graph_get_consumer_nodes(*m.graph, node_get_output_name(*down));
    auto output_node_args = std::set<const NodeArg*>{};
    MY_LOG(1) << "down process boundary: " << node_as_string(*down);
    for (auto c : consumers) {
      if (m.contains(c)) {
        MY_LOG(1) << "consumer already exists: " << node_as_string(*c);
        // already processed;
        continue;
      }
      if (!is_xilinx_op(c)) {
        auto output_node_arg = find_node_arg_between(down, c);
        if (output_node_arg == nullptr) {
          LOG(WARNING) << "cannot find node arg between "
                       << node_as_string(*down) << " => " << node_as_string(*c);
        } else {
          output_node_args.insert(output_node_arg);
        }
        MY_LOG(1) << "c is not a xilinx op: " << node_as_string(*c);
        continue;
      }
      // it is a xilinx op
      m.boundary.push_back(c);
      m.insert(c);
      ret = true;
    }
    for (auto out : output_node_args) {
      m.output_node_args.push_back(out);
    }
  }
  return ret;
}

static bool fuse_node_rec(FusedXmodel& m) {
  auto ret = false;
  std::vector<const Node*> boundary = std::move(m.boundary);
  ret = fuse_node_rec_up(m, boundary) || ret;
  ret = fuse_node_rec_down(m, boundary) || ret;
  return ret;
}

static bool fuse_node(Graph& graph, const Node* node, FusedXmodel& m) {
  if (VAIP_ORT_API(node_type_is_fused)(*node)) {
    return false;
  }
  if (!is_xilinx_op(node)) {
    return false;
  }
  auto from_nodes = std::vector<const Node*>{node};
  m.boundary = {node};
  m.insert(node);
  while (fuse_node_rec(m)) {
  }
  auto name = std::string();
  if (m.output_node_args.empty()) {
    static int id = 0;
    LOG(WARNING) << "xmodel has no outputs";
    name = std::string("noname_") + std::to_string(id++);
  } else {
    name = node_arg_get_name(*m.output_node_args[0]);
  }
  auto op_type = std::string("xmodel");
  auto nodes = std::vector<size_t>();
  nodes.reserve(m.nodes.size());
  for (auto i = 0u; i < m.nodes.size(); ++i) {
    auto index = (size_t)VAIP_ORT_API(node_get_index)(*m.nodes[i]);
    nodes.push_back(index);
  }
  auto inputs = std::vector<std::string>();
  inputs.reserve(m.input_node_args.size());
  for (auto i = 0u; i < m.input_node_args.size(); ++i) {
    inputs.push_back(node_arg_get_name(*m.input_node_args[i]));
  }
  auto outputs = std::vector<std::string>();
  outputs.reserve(m.output_node_args.size());
  for (auto i = 0u; i < m.output_node_args.size(); ++i) {
    outputs.push_back(node_arg_get_name(*m.output_node_args[i]));
  }
  auto constant_initializers = std::vector<std::string>();
  if (true) {
    MY_LOG(1) << "merge node: " << node_as_string(*node);
    MY_LOG(1) << "nodes:";
    for (auto n : m.nodes) {
      MY_LOG(1) << "\t\t" << node_as_string(*n);
    }
    MY_LOG(1) << "inputs:";
    for (auto n : m.input_node_args) {
      MY_LOG(1) << "\t\t" << node_arg_as_string(*n);
    }
    MY_LOG(1) << "outputs:";
    for (auto n : m.output_node_args) {
      MY_LOG(1) << "\t\t" << node_arg_as_string(*n);
    }
  }
  /* auto& fused_node*/ VAIP_ORT_API(graph_fuse)(
      graph, name, op_type, nodes, inputs, outputs, constant_initializers);
  // do we need to remove nodes? No. these nodes are removed by graph_fused.
  //
  // Note: some constant inputs becomse constant_initializers. they are copied,
  // but the name in subgraph is as same as the name in the parent.
  // this would not be a problem if all constant initializers are replaced with
  // Constant node.
  return true;
}

void fuse_xmodel(Graph& graph) {
  auto modified = false;
  do {
    modified = false;
    auto nodes = graph_get_node_in_topoligical_order_reverse(graph);
    size_t max_idx = 0;
    for (auto node : nodes) {
      max_idx = std::max(max_idx, node);
    }
    if (max_idx == 0) {
      LOG(WARNING) << "cannot find max node index";
      break;
    }
    for (auto node : nodes) {
      FusedXmodel m;
      m.mask.resize((size_t)max_idx + 1);
      m.graph = &graph;
      m.graph_outputs = graph_get_outputs(graph);
      modified = fuse_node(graph, VAIP_ORT_API(graph_get_node)(graph, node), m);
      if (modified) {
        break;
      }
    }
  } while (modified);
  return;
}

} // namespace vaip_pass_fuse_xmodel
