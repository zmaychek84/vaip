/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */
#include "qmha/qmha_processor.hpp"
#include "vaip/dd/dd_utils.hpp"
#include "vaip/pattern_zoo.hpp"
#include "vaip/vaip.hpp"
#include "vitis/ai/env_config.hpp"
#include <glog/logging.h>

DEF_ENV_PARAM(DEBUG_DD_MERGE_QMHAWINDOW, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_DD_MERGE_QMHAWINDOW) >= n)

/**
 * test case: m3uec
 *
 *
 * Replace pattern:
 *
 * From: <???>
 * To  : <???>
 */

// add the following line in your vaip_config.json
/*
    { "name": "vaip_pass_dd_merge_qmhawindow",
       "plugin": "vaip-pass_dd_merge_qmhawindow",
       "disabled": false
    }
*/
namespace {
using namespace vaip_core;
// clang-format off
#define INPUT_0 "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/qkv/Add_output_0_QuantizeLinear_Output"
// clang-format on
struct DdMergeQmhawindow {
  DdMergeQmhawindow(IPass& self) : self_{self} {}
  std::unique_ptr<Rule> create_rule(IPass* self) {
    auto pattern_ = vaip::pattern_zoo::get_pattern("QMHAWINDOW_0");
    return Rule::create_rule(
        pattern_, [=](onnxruntime::Graph* graph_ptr, binder_t& binder) -> bool {
          auto ni_input = binder[INPUT_0];
          auto ni_output = binder[pattern_->get_id()];
          MY_LOG(1) << "found node:" << node_as_string(*ni_output.node);
          auto const binder_params =
              std::unordered_map<std::string, std::vector<std::string>>{
                  // clang-format off
              {"QKT_input_qparams",{
        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Mul_4_output_0_scale",
        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Mul_4_output_0_zero_point",
        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/qkv/Add_output_0_scale",
        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/qkv/Add_output_0_zero_point",
                  }},
              {"QKT_output_qparams",{
        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/MatMul_output_0_scale",
        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/MatMul_output_0_zero_point",
                  }},
              {"MATMUL_input", {
                      "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Mul_4_output_0_DequantizeLinear_Output",
                      "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Transpose_4_output_0_DequantizeLinear_Output"
                  }},
              {"VSQKT_input_qparams",{
        "/blocks.3/blocks.3.0/channel_block/channel_attn/fn/Softmax_output_0_scale",
        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/softmax/Softmax_output_0_zero_point",
        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/qkv/Add_output_0_scale",
        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/qkv/Add_output_0_zero_point",
                  }},
              {"VSQKT_output_qparams",{
        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/MatMul_1_output_0_scale",
        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/MatMul_1_output_0_zero_point",
                  }},
              {"VSMATMUL_input", {
                      "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/softmax/Softmax_output_0_DequantizeLinear_Output",
                      "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Reshape_11_output_0_DequantizeLinear_Output"
                  }},
              {"softmax_input_qparams",{
        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/MatMul_output_0_scale",
        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/MatMul_output_0_zero_point",
                  }},
              {"softmax_output_qparams",{
        "/blocks.3/blocks.3.0/channel_block/channel_attn/fn/Softmax_output_0_scale",
        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/softmax/Softmax_output_0_zero_point",
                  }},
              {"MUL_input_qparams",{
        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/qkv/Add_output_0_scale",
        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/qkv/Add_output_0_zero_point",
                  }},
              {"MUL_weight_qparams",{
        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Constant_33_output_0_quantized",
        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Constant_33_output_0_scale",
        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Constant_33_output_0_zero_point",
                  }},
              {"MUL_output_qparams",{
        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Mul_4_output_0_scale",
        "/blocks.0/blocks.0.0/spatial_block/window_attn/fn/Mul_4_output_0_zero_point",
                  }},
      };
          // clang-format on
          auto processor =
              std::make_unique<vaip_dd_merge_qma::DdMergeQmhaProcessor>(
                  self_, graph_ptr, &binder, binder_params);
          std::vector<std::string> ns =
              vaip::dd::get_node_names(graph_ptr, binder);
          auto& node_arg_qdq_params = processor->process(pattern_->get_id());
          NodeBuilder(*graph_ptr, self_)
              .set_op_type("QMHAWINDOW", "com.xilinx")
              .set_input_node_args({ni_input.node_arg, &node_arg_qdq_params})
              .clone_data_type(*ni_output.node_arg)
              .clone_shape(*ni_output.node_arg)
              .add("nodes", ns)
              .add("in_dtypes", std::vector<std::string>({"uint16", "int64"}))
              .add("out_dtypes", std::vector<std::string>({"uint16"}))
              .set_anchor_point1(*ni_output.node_arg)
              .build();
          return true;
        });
  }
  // apply the rule
  void process(IPass& self, Graph& graph) { create_rule(&self)->apply(&graph); }

  IPass& self_;
};
} // namespace

DEFINE_VAIP_PASS(DdMergeQmhawindow, vaip_pass_dd_merge_qmhawindow)
