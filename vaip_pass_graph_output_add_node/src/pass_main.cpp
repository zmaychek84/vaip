/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

#include <glog/logging.h>

#include "./graph_output_add_node.hpp"

#include "vitis/ai/env_config.hpp"
using namespace vaip_core;
namespace {
struct GraphOutputAddNode {
  GraphOutputAddNode(IPass& self) {}
  void process(IPass& self, Graph& graph) {
    vaip_pass_graph_output_add_node::GraphOutputAddNodeRule().apply(&graph);
  }
};
} // namespace

DEFINE_VAIP_PASS(GraphOutputAddNode, vaip_pass_graph_output_add_node)
