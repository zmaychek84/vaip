/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */
// clang-format off
#include "vaip/vaip.hpp"
#include "../../vaip/src/stat.hpp"
#include "../../encryption/src/encryption.hpp"
#include "vitis/ai/env_config.hpp"
#include <filesystem>
#include <glog/logging.h>
#include <google/protobuf/util/json_util.h>
#include <set>
#include <sstream>
#include <vaip/vaip_ort_api.h>
#include "vaip/capability.pb.h"
#include "vaip/with_current_graph.hpp"
// clang-format on

DEF_ENV_PARAM(DEBUG_LEVEL1_SHELL_1, "0")
DEF_ENV_PARAM_2(XLNX_model_clone_external_data_threshold, "128", int64_t)
DEF_ENV_PARAM(XLNX_ENABLE_DUMP_XIR_MODEL, "0")
DEF_ENV_PARAM(XLNX_ENABLE_DUMP_CONSTANT, "0")
DEF_ENV_PARAM(VAIP_COMPILE_RESERVE_CONST_DATA, "0")
DEF_ENV_PARAM(DEBUG_SKIP_COMPILE_XMODEL, "0")
DEF_ENV_PARAM(DEBUG_SKIP_EXPORT_TO_XIR, "0")
DEF_ENV_PARAM(XLNX_ONNX_EP_VERBOSE, "0")
DEF_ENV_PARAM(XLNX_MINIMUM_NUM_OF_CONV, "-2")

#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_LEVEL1_SHELL_1) >= n)
#define LOG_VERBOSE(n)                                                         \
  LOG_IF(INFO, ENV_PARAM(XLNX_ONNX_EP_VERBOSE) >= n)                           \
      << "[XLNX_ONNX_EP_VERBOSE] "
namespace {
using namespace vaip_core;

struct Level1DynamicDispatch {
  Level1DynamicDispatch(IPass& self)
      : self_{self}, log_dir_{self.get_log_path()} {}
  void process_run_passes(Graph& cloned_graph) {

    auto& pass_proto = self_.get_pass_proto();
    all_passes_ = IPass::create_passes(self_.get_context(),
                                       pass_proto.pass_dpu_param().sub_pass());
    IPass::run_passes(all_passes_, cloned_graph);
  }

  void process(IPass& self, Graph& graph) {
    auto log_dir = self.get_log_path();
    if (log_dir.empty()) {
      LOG(WARNING) << "log dir is empty, call saving onnx.onnx";
      return;
    }
    // Save original graph
    auto file = log_dir / "onnx.onnx";
    auto dat_file = "onnx.dat";
    VAIP_ORT_API(graph_save)
    (graph, file.u8string(), dat_file, 128u);
    // Clone graph
    const auto& model = graph_get_model(graph);
    auto cloned_model = model_clone(model);
    auto& cloned_graph = VAIP_ORT_API(model_main_graph)(*cloned_model);

    // use std::ignore would cause it to destruct immediately
    auto graph_saver = WithCurrentGraph(&graph, &self);
    (void)graph_saver;
    try {
      // Process all fusion passes
      process_run_passes(cloned_graph);
      // save fused graph
      auto fused_file = log_dir / "fused_cxx.onnx";
      auto fused_dat_file = "fused_cxx.dat";
      VAIP_ORT_API(graph_save)
      (cloned_graph, fused_file.u8string(), fused_dat_file,
       std::numeric_limits<size_t>::max());
    } catch (const std::exception& e) {
      LOG(WARNING) << "Unexcepted exception: "
                   << "(Exception type: " << typeid(e).name() << ") "
                   << e.what();
      return;
    }
  }

  IPass& self_;

  const std::filesystem::path& log_dir_;
  std::vector<std::shared_ptr<IPass>> all_passes_;
};
} // namespace

DEFINE_VAIP_PASS(Level1DynamicDispatch, vaip_pass_level1_dd)
