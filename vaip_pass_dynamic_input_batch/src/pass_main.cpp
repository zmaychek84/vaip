/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

#include "vaip/vaip.hpp"
#include "vitis/ai/env_config.hpp"
#include <glog/logging.h>
DEF_ENV_PARAM(DEBUG_DYNAMIC_INPUT_BATCH, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_DYNAMIC_INPUT_BATCH) >= n)
namespace {
using namespace vaip_core;
struct DynamicInputBatch {
  DynamicInputBatch(IPass& self) {}

  // change the shape of input from -1 to 1  to support dynamic batching
  bool process(IPass& self, Graph& graph) {
    auto changed = false;
    auto graph_inputs = graph_get_inputs(graph);
    for (auto input : graph_inputs) {
      auto input_shape_ptr = node_arg_get_shape_i64(*input);
      CHECK(input_shape_ptr != nullptr)
          << node_arg_as_string(*input) << " shape absent";
      auto input_shape = *input_shape_ptr;
      if (input_shape.size() == 0) {
        // scalar
        continue;
      }
      for (auto i = 1u; i < input_shape.size(); i++) {
        CHECK_NE(input_shape[i], -1)
            << "don't support other shape dimension except batch"
            << "is -1 now";
      }
      if (input_shape[0] == -1) {
        MY_LOG(1) << "do graph_input_modify_batch.";
        input_shape[0] = 1;
        VAIP_ORT_API(node_arg_set_shape_i64)(*input, input_shape);
        changed = true;
        graph_resolve(graph, true);
      }
    }
    return changed;
  }

private:
};
} // namespace
DEFINE_VAIP_PASS(DynamicInputBatch, vaip_pass_dynamic_input_batch)
