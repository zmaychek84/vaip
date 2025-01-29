/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

#include "vaip/vaip.hpp"

#include "vitis/ai/env_config.hpp"

#include <glog/logging.h>
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
DEF_ENV_PARAM(DEBUG_QUANTIZE_MODEL, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_QUANTIZE_MODEL) >= n)

namespace {
using namespace vaip_core;
struct QuantizeModel {
  QuantizeModel(IPass& self) : self_(self) {}

  void run(std::string in, std::string out, Graph& graph) {
    auto quantize_mode = self_.get_pass_proto().args()[0];
    auto inputs = graph_get_inputs(graph);
    CHECK_EQ(inputs.size(), 1)
        << "Currently only supports quantization of models with one input";
    auto input_shape_ptr = node_arg_get_shape_i64(*inputs[0]);
    if (nullptr == input_shape_ptr) {
      LOG(FATAL) << "Failed to get input shape";
    }
    auto input_shape = *input_shape_ptr;
    {
      auto inter = init_interpreter();
      py::list shape;
      for (const auto& elem : input_shape) {
        shape.append(py::cast(elem));
      }
      auto m = py::module::import("voe.tools.quantize_pass");
      m.attr("quantize_static")(in, out, shape, quantize_mode);
    }
  }

  bool need_to_quantize(Graph& graph) {
    auto nodes = graph_nodes(graph);
    for (auto n : nodes) {
      if (node_is_op(*n, "QuantizeLinear", ""))
        return false;
      else if (node_is_op(*n, "FixNeuron", ""))
        return false;
    }
    return true;
  }

  void process(IPass& self, Graph& graph) {
    if (need_to_quantize(graph)) {
      auto onnx = self.get_cache_file_name("before_quant.onnx").u8string();
      VAIP_ORT_API(graph_save)
      (graph, onnx, self.get_cache_file_name("data.bin").u8string(),
       std::numeric_limits<size_t>::max());
      auto quant = self.get_cache_file_name("quant.onnx").u8string();
      run(onnx, quant, graph);

      auto model_x = model_load(quant);
      auto& new_graph = VAIP_ORT_API(model_main_graph)(*model_x);
      graph_resolve(new_graph);

      self.add_context_resource(
          "__current_model",
          std::shared_ptr<void>((void*)model_x.release(), [](void* p) {
            VAIP_ORT_API(model_delete)((onnxruntime::Model*)p);
          }));
      self.add_context_resource(
          "__current_graph",
          std::shared_ptr<void>((void*)&new_graph, [](void* p) {}));
    } else
      MY_LOG(0) << "The current model is already a quantized model to skip the "
                   "quantizemodel pass";
  }

private:
  IPass& self_;
};
} // namespace
DEFINE_VAIP_PASS(QuantizeModel, vaip_pass_quantize_model)
