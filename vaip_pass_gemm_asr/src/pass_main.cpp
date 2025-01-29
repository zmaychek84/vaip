/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

#include <glog/logging.h>

#include "vaip/vaip.hpp"
#include "vitis/ai/env_config.hpp"

// tvm headers
#include <dlpack/dlpack.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/target/codegen.h>
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_GEMM_ASR) >= n)
DEF_ENV_PARAM(DEBUG_GEMM_ASR, "0");
using namespace vaip_core;

/**
    com.xilinx:const() -> com.xilinx:fix(com.xilinx:const())
    where fix_point can find in PassContext
 */

using TvmIntArray = ::tvm::Array<::tvm::Integer>;
using TvmPackedFunc = ::tvm::PackedFunc;
namespace tvm_rt = ::tvm::runtime;
using TvmModule = ::tvm::runtime::Module;
using AsrTensorShape = std::vector<int64_t>;
using AsrShapeDict = std::map<std::string, AsrTensorShape>;
using AsrShapeVec = std::vector<AsrTensorShape>;

struct GemmAsrPass {
  GemmAsrPass(IPass& self) : self_{self} {}

  void get_required_info(
      Graph& graph, std::vector<std::string>& all_input_names,
      AsrShapeVec& input_shapes, std::vector<std::string>& all_output_names,
      AsrShapeVec& output_shapes, std::vector<std::string>& all_init_names,
      std::vector<size_t>& all_nodes, std::filesystem::path& save_path) {
    auto all_input_node_args = graph_get_inputs(graph);
    MY_LOG(1) << "all_input_node_arg size " << all_input_node_args.size();
    for (auto& node_arg : all_input_node_args) {
      auto shape = node_arg_get_shape_i64(*node_arg);
      auto& node_arg_name = VAIP_ORT_API(node_arg_get_name_unsafe)(*node_arg);
      auto consumers = graph_get_consumer_nodes(graph, node_arg_name);
      if (consumers.size() == 0) {
        MY_LOG(1) << "input " << node_arg_name
                  << " has no consumers, will be skipped";
        continue;
      }
      all_input_names.push_back(node_arg_name);
      input_shapes.push_back(*(shape.get()));
    }

    auto all_output_node_args = graph_get_outputs(graph);
    MY_LOG(1) << "all_output_node_args size " << all_output_node_args.size();
    for (auto& node_arg : all_output_node_args) {
      auto shape = node_arg_get_shape_i64(*node_arg);
      all_output_names.push_back(
          VAIP_ORT_API(node_arg_get_name_unsafe)(*node_arg));
      output_shapes.push_back(*(shape.get()));
    }
    // all nodes
    all_nodes = graph_get_node_in_topoligical_order(graph);
    MY_LOG(1) << "all_nodes size " << all_nodes.size();

    // all inits
    auto inits_map = VAIP_ORT_API(graph_get_all_initialized_tensors)(graph);

    for (auto& item : inits_map) {
      all_init_names.push_back(item.first);
    }
    MY_LOG(1) << " all_init_names size " << all_init_names.size();

    // save model to cache dir
    save_path = self_.get_context()->get_log_dir() / "cloned_graph.onnx";
    VAIP_ORT_API(graph_save)
    (graph, save_path.string(), "", std::numeric_limits<size_t>::max());
    MY_LOG(1) << "model saved to " << save_path;
  }

  void process(IPass& self, Graph& graph) {
    const auto& config = self.get_context()->get_config_proto();
    MY_LOG(1) << " cache_key " << config.cache_key() << " cache_dir "
              << config.cache_dir() << " log_dir "
              << self.get_context()->get_log_dir();

    auto& pass_proto = self_.get_pass_proto();
    auto& asr_config = pass_proto.pass_asr_config();

    // assume no dynamic shape for now
    // AsrShapeDict asr_input_shape_dict;
    // for (auto& item : asr_config.input_shapes()) {
    //   asr_input_shape_dict[item.first] = AsrTensorShape(
    //       item.second.shapes().begin(), item.second.shapes().end());
    // }

    // use gemm compiler
    const TvmPackedFunc* compile_func =
        tvm_rt::Registry::Get("tvm_onnx_import_and_compile");
    if (!compile_func) {
      LOG(FATAL) << "can't get tvm_onnx_import_and_compile from tvm registry";
    } else {
      MY_LOG(1) << "got tvm_onnx_import_and_compile from tvm registry";
    }

    std::vector<size_t> all_nodes;
    std::vector<std::string> all_inputs;
    std::vector<std::string> all_inits;
    std::vector<std::string> all_outputs;
    AsrShapeVec input_shapes;
    AsrShapeVec output_shapes;
    std::filesystem::path save_path;
    get_required_info(graph, all_inputs, input_shapes, all_outputs,
                      output_shapes, all_inits, all_nodes, save_path);

    ::tvm::Array<TvmIntArray> asr_input_shapes;

    for (auto& shape : input_shapes) {
      TvmIntArray asr_shape;
      for (auto dim : shape) {
        asr_shape.push_back(::tvm::Integer((int)dim));
      }
      asr_input_shapes.push_back(asr_shape);
    }

    TvmModule mod = (*compile_func)(
        save_path.string(), self.get_context()->get_config_proto().cache_key(),
        asr_config.target(), asr_config.target_host(), asr_config.opt_level(),
        true, asr_input_shapes, asr_config.build_version(),
        asr_config.aie_target(), asr_config.aiectrl_cfg(), asr_config.xclbin());
    auto asr_mod = std::make_unique<TvmModule>(mod);

    if (asr_mod->get()) {
      auto& metadef =
          self.fuse(graph, "mycustomasr", "custom_asr_type", all_nodes,
                    all_inputs, all_outputs, all_inits, "ASR");

      auto asr_param = metadef.mutable_asr_param();
      for (auto& shape : input_shapes) {
        (*(asr_param->add_input_shapes()->mutable_shapes())) = {shape.begin(),
                                                                shape.end()};
      }
      for (auto& shape : output_shapes) {
        (*(asr_param->add_output_shapes()->mutable_shapes())) = {shape.begin(),
                                                                 shape.end()};
      }
      MY_LOG(1) << "output_zero_copy " << asr_config.output_zero_copy();
      asr_param->set_output_zero_copy(asr_config.output_zero_copy());
      self_.add_context_resource(
          "asr_module", std::shared_ptr<void>(asr_mod.release(), [](void* p) {
            MY_LOG(1) << "calling asr module dtor";
            delete static_cast<TvmModule*>(p);
          }));
    }
  }
  ~GemmAsrPass() {}
  IPass& self_;
};

DEFINE_VAIP_PASS(GemmAsrPass, vaip_pass_gemm_asr)
