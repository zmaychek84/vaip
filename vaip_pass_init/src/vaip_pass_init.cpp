/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

#include <glog/logging.h>

#include "vaip/pattern_zoo.hpp"
#include "vaip/vaip.hpp"
#include "vitis/ai/env_config.hpp"
using namespace vaip_core;
DEF_ENV_PARAM(XLNX_ENABLE_DUMP_ONNX_MODEL, "0")
DEF_ENV_PARAM(DEBUG_PATTERN_SAMPLE, "0")

struct InitPass {
  InitPass(IPass& self) {}
  void process(IPass& self, Graph& graph) {
    if (ENV_PARAM(XLNX_ENABLE_DUMP_ONNX_MODEL)) {
      auto log_dir = self.get_log_path();
      if (log_dir.empty()) {
        LOG(WARNING) << "log dir is empty, call saving onnx.onnx";
        return;
      }
      auto file = log_dir / "onnx.onnx";
      auto dat_file = "onnx.dat";
      VAIP_ORT_API(graph_save)
      (graph, file.u8string(), dat_file, std::numeric_limits<size_t>::max());
      LOG(INFO) << "save origin onnx model to " << file << " data in "
                << dat_file;
    }

    if (ENV_PARAM(DEBUG_PATTERN_SAMPLE)) {
      auto pattern_list = vaip::pattern_zoo::pattern_list();
      LOG(INFO) << "all pattern size " << pattern_list.size();
      for (auto p : pattern_list) {
        LOG(INFO) << " " << p;
      }
      auto pattern = vaip::pattern_zoo::get_pattern("Sample");
      CHECK(pattern != nullptr);
      LOG(INFO) << "Sample pattern : debug string " << pattern->debug_string();
      auto rule = Rule::create_rule(
          pattern,
          [=](onnxruntime::Graph* graph_ptr, binder_t& binder) -> bool {
            auto ni_input = binder["input_0"];
            LOG(INFO) << "found node input_0 : "
                      << node_as_string(*ni_input.node);
            auto ni_output = binder[pattern->get_id()];
            LOG(INFO) << "found node conv :" << node_as_string(*ni_output.node);
            return false;
          });
      rule->apply(&graph);
    }
    return;
  }
};

DEFINE_VAIP_PASS(InitPass, vaip_pass_init)
