/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */
// testcase 110
#include "vaip/vaip.hpp"
#include "vitis/ai/env_config.hpp"
#include <glog/logging.h>
DEF_ENV_PARAM(DEBUG_REMOVE_BOTTOM_TRANSPOSE, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_REMOVE_BOTTOM_TRANSPOSE) >= n)

/// remove the bottom tranpose op.  Why? for model #100, it triggers a
/// strange bug, `xcompiler(feat_ipu_compiler_demo)` complains that
/// size is too large, I don't why, to work around this bug, we remove
/// the last op if it is `transpose`
///
namespace {
using namespace vaip_core;
struct RemoveBottomTranspose {
  RemoveBottomTranspose(const IPass& self) : self_{self} {}
  void process(const IPass& self, Graph& graph) {
    auto graph_outputs = graph_get_outputs(graph);
    std::vector<const NodeArg*> new_outputs;
    new_outputs.resize(graph_outputs.size());
    auto index = 0u;
    for (auto output : graph_outputs) {
      auto node =
          VAIP_ORT_API(graph_producer_node)(graph, node_arg_get_name(*output));
      if (node_is_op(*node, "transpose", "xilinx.com")) {
        new_outputs[index] = node_get_input_node_args(*node)[0];
      } else {
        new_outputs[index] = graph_outputs[index];
      }
      index = index + 1;
    }
    VAIP_ORT_API(graph_set_outputs)(graph, new_outputs);
  }
  const IPass& self_;
};
} // namespace
DEFINE_VAIP_PASS(RemoveBottomTranspose, vaip_pass_remove_bottom_transpose)
