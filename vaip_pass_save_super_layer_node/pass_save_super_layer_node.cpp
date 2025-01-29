/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

#include "vaip/vaip.hpp"
#include <glog/logging.h>

namespace {
using namespace vaip_core;

struct SaveSuperLayer {
  SaveSuperLayer(IPass& self) {}
  void process(IPass& pass, Graph& graph) {
    auto nodes = graph_nodes(graph);
    for (auto& node : nodes) {
      if (node_is_op(*node, "super_layer", "com.xilinx")) {
        auto& graph = VAIP_ORT_API(node_get_function_body)(*node);
        auto onnx = pass.get_cache_file_name(
                            VAIP_ORT_API(node_get_name)(*node) + ".onnx")
                        .u8string();
        VAIP_ORT_API(graph_save)
        (graph, onnx, pass.get_cache_file_name("data.bin").u8string(),
         std::numeric_limits<size_t>::max());
      }
    }
  }
};
} // namespace
DEFINE_VAIP_PASS(SaveSuperLayer, vaip_pass_save_super_layer_node)
