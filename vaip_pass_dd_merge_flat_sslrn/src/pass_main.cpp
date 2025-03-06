/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */
#ifdef __GNUC__
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wpedantic"
#  pragma GCC diagnostic ignored "-Wconversion"
#endif

#include "vaip/dd/coeffs.hpp"
#include "vaip/dd/dd_utils.hpp"

#include "vaip/pattern_zoo.hpp"

#ifdef __GNUC__
#  pragma GCC diagnostic pop
#endif
#include "vaip/vaip.hpp"
#include "vitis/ai/env_config.hpp"
#include <functional>
#include <glog/logging.h>
#include <numeric>

#include "vaip/tensor_proto.hpp"

DEF_ENV_PARAM(DEBUG_DD_MERGE_FLAT_SSLRN, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_DD_MERGE_FLAT_SSLRN) >= n)

namespace {
using namespace vaip_core;

struct flat_sslrn {
  flat_sslrn(IPass& self) : self_{self} {}

  std::unique_ptr<Rule> create_rule(IPass* self) {
    auto builder = PatternBuilder();

    auto input_0 =
        builder.wildcard(); //  id = 0  node_arg_name =
                            //  /model/embed_tokens/Gather_output_0_to_bf16_
    builder.bind("/model/embed_tokens/Gather_output_0_to_bf16_", input_0);
    auto Cast_0 = builder.node2(
        "Cast", {input_0}); //  id = 1  node_arg_name =
                            //  /model/embed_tokens/Gather_output_0_to_fp32_
    builder.bind("/model/embed_tokens/Gather_output_0_to_fp32_", Cast_0);
    auto input_1 =
        builder
            .wildcard(); //  id = 2  node_arg_name =
                         //  /model/layers.0/self_attn/o_proj/MatMul_output_0_DequantizeLinear_Output_to_bf16_1
    builder.bind("/model/layers.0/self_attn/o_proj/"
                 "MatMul_output_0_DequantizeLinear_Output_to_bf16_1",
                 input_1);
    auto Cast_1 = builder.node2(
        "Cast",
        {input_1}); //  id = 3  node_arg_name =
                    //  /model/layers.0/self_attn/o_proj/MatMul_output_0_DequantizeLinear_Output_to_bf16_1_to_fp32_
    builder.bind("/model/layers.0/self_attn/o_proj/"
                 "MatMul_output_0_DequantizeLinear_Output_to_bf16_1_to_fp32_",
                 Cast_1);
    auto Add_0 = builder.node2(
        "Add",
        {Cast_0,
         Cast_1}); //  id = 4  node_arg_name = /model/layers.0/Add_output_0
    builder.bind("/model/layers.0/Add_output_0", Add_0);
    auto Cast_2 = builder.node2(
        "Cast", {Add_0}); //  id = 5  node_arg_name =
                          //  /model/layers.0/Add_output_0_to_bf16_
    builder.bind("/model/layers.0/Add_output_0_to_bf16_", Cast_2);
    auto Cast_3 = builder.node2(
        "Cast", {Cast_2}); //  id = 6  node_arg_name =
                           //  /model/layers.0/Add_output_0_to_fp32_duplicate_
    builder.bind("/model/layers.0/Add_output_0_to_fp32_duplicate_", Cast_3);
    auto constant_0 =
        builder
            .constant(); //  id = 7  node_arg_name =
                         //  /model/layers.0/post_attention_layernorm/L2Norm_scale
    builder.bind("/model/layers.0/post_attention_layernorm/L2Norm_scale",
                 constant_0);
    auto SimplifiedLayerNormalization_0 = builder.node2(
        "SimplifiedLayerNormalization",
        {Cast_3,
         constant_0}); //  id = 8  node_arg_name =
                       //  /model/layers.0/post_attention_layernorm/Mul_1_output_0
    builder.bind("/model/layers.0/post_attention_layernorm/Mul_1_output_0",
                 SimplifiedLayerNormalization_0);
    auto Cast_4 = builder.node2(
        "Cast",
        {SimplifiedLayerNormalization_0}); //  id = 9  node_arg_name =
                                           //  /model/layers.0/post_attention_layernorm/Mul_1_output_0_to_bf16_
    builder.bind(
        "/model/layers.0/post_attention_layernorm/Mul_1_output_0_to_bf16_",
        Cast_4);
    auto seq =
        builder.sequence(std::vector<std::shared_ptr<vaip_core::Pattern>>{
            Cast_2,
            Cast_4,
        });

    return Rule::create_rule(
        seq, [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
          auto input_0_node = binder[input_0->get_id()];
          auto input_1_node = binder[input_1->get_id()];
          auto out_node_0 = binder[Cast_2->get_id()]; // add
          auto out_node_1 = binder[Cast_4->get_id()]; // rms
          std::vector<std::string> input_types{"bfloat16", "bfloat16"};
          std::vector<std::string> output_types{"bfloat16", "bfloat16"};
          auto a_shape = node_arg_get_shape_i64(*input_0_node.node_arg);
          auto b_shape = node_arg_get_shape_i64(*input_1_node.node_arg);
          auto c_shape = node_arg_get_shape_i64(*out_node_0.node_arg);

          auto nb = NodeBuilder(*graph, *self);
          nb.set_input_node_args(
              {input_1_node.node_arg, input_0_node.node_arg});
          nb.set_op_type("FlatRMSAdd", "com.xilinx");
          nb.clone_attrs(*out_node_0.node);
          nb.add("in_dtypes", input_types);
          nb.add("out_dtypes", output_types);
          nb.add("a_shape", *a_shape);
          nb.add("b_shape", *b_shape);
          nb.add("c_shape", *c_shape);
          nb.add("shape", *c_shape);
          nb.add("data_type", "bfloat16");
          nb.set_anchor_point1(*out_node_0.node_arg); // add output
          nb.add_output();
          nb.set_anchor_point1(*out_node_1.node_arg);
          nb.build_ex();
          return true; // return true if graph is modified.
        });
  }
  // apply the rule
  void process(IPass& self, Graph& graph) {
    // MY_LOG(1) << self_.get_pass_proto().name() << "[" <<
    // self_.get_pass_proto().plugin() << "] start processing graph";
    create_rule(&self)->apply(&graph);
    // MY_LOG(1) << self_.get_pass_proto().name() << "[" <<
    // self_.get_pass_proto().plugin() << "] finish processing graph";
  }

  IPass& self_;
};
} // namespace

DEFINE_VAIP_PASS(flat_sslrn, vaip_pass_dd_merge_flat_sslrn)
