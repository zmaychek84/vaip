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

DEF_ENV_PARAM(DEBUG_DD_MERGE_MLADFADD, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_DD_MERGE_MLADFADD) >= n)

namespace {
using namespace vaip_core;
struct mladfadd {
  mladfadd(IPass& self) : self_{self} {}

  std::unique_ptr<Rule> create_rule(IPass* self) {
    auto builder = PatternBuilder();

    auto input_0 =
        builder
            .wildcard(); //  id = 0  node_arg_name =
                         //  /model/layers.0/self_attn/o_proj/MatMul_output_0_DequantizeLinear_Output_to_bf16_1
    builder.bind("/model/layers.0/self_attn/o_proj/"
                 "MatMul_output_0_DequantizeLinear_Output_to_bf16_1",
                 input_0);
    auto Cast_0 = builder.node2(
        "Cast",
        {input_0}); //  id = 1  node_arg_name =
                    //  /model/layers.0/self_attn/o_proj/MatMul_output_0_DequantizeLinear_Output_to_bf16_1_to_fp32_
    builder.bind("/model/layers.0/self_attn/o_proj/"
                 "MatMul_output_0_DequantizeLinear_Output_to_bf16_1_to_fp32_",
                 Cast_0);
    auto input_1 =
        builder.wildcard(); //  id = 2  node_arg_name =
                            //  /model/embed_tokens/Gather_output_0_to_bf16_
    builder.bind("/model/embed_tokens/Gather_output_0_to_bf16_", input_1);
    auto Cast_1 = builder.node2(
        "Cast", {input_1}); //  id = 3  node_arg_name =
                            //  /model/embed_tokens/Gather_output_0_to_fp32_
    builder.bind("/model/embed_tokens/Gather_output_0_to_fp32_", Cast_1);
    auto Add_0 = builder.node2(
        "Add",
        {Cast_1,
         Cast_0}); //  id = 4  node_arg_name = /model/layers.0/Add_output_0
    builder.bind("/model/layers.0/Add_output_0", Add_0);
    auto cast_out = builder.node2(
        "Cast", {Add_0}); //  id = 5  node_arg_name =
                          //  /model/layers.0/Add_output_0_to_bf16_
    builder.bind("/model/layers.0/Add_output_0_to_bf16_", cast_out);

    return Rule::create_rule(
        cast_out, [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
          std::vector<std::string> ns = vaip::dd::get_node_names(graph, binder);
          auto input_0_node = binder[input_0->get_id()];
          auto input_1_node = binder[input_1->get_id()];
          auto out_node = binder[cast_out->get_id()];

          std::vector<std::string> input_types{"bfloat16", "bfloat16"};
          std::vector<std::string> output_types{"bfloat16"};
          NodeBuilder(*graph, *self)
              .set_input_node_args(
                  {input_0_node.node_arg, input_1_node.node_arg})
              .set_op_type("MLADFADD", "com.xilinx")
              .clone_attrs(*out_node.node)
              .add("nodes", ns)
              .set_anchor_point1(*out_node.node)
              .add("in_dtypes", input_types)
              .add("out_dtypes", output_types)
              .build();
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

DEFINE_VAIP_PASS(mladfadd, vaip_pass_dd_merge_mladfadd)
