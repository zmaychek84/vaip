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

DEF_ENV_PARAM(DEBUG_DD_MERGE_QSILU, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_DD_MERGE_QSILU) >= n)

/**
 * test case: <???>
 *
 *
 * Replace pattern:
 *
 * From: <???>
 * To  : <???>
 */

// add the following line in your vaip_config.json
/*
    { "name": "vaip_pass_dd_merge_qsilu",
       "plugin": "vaip-pass_dd_merge_qsilu",
       "disabled": false
    }
*/
namespace {
using namespace vaip_core;
static bool first_silu = true;
struct Silu {
  Silu(IPass& self) : self_{self} {}

  std::unique_ptr<Rule> create_rule(IPass* self) {
    auto builder = PatternBuilder();
    auto input_0 = builder.wildcard();
    auto input_1 = builder.wildcard();

    //// Cast inputs
    auto cast_0 = builder.node2("Cast", {input_0});
    auto cast_1 = builder.node2("Cast", {input_1});

    // Sigmoid
    auto sigmoid = builder.node2("Sigmoid", {cast_0});
    auto sigmoid_bf16 = builder.node2("Cast", {sigmoid});
    auto sigmoid_fp32 = builder.node2("Cast", {sigmoid_bf16});
    // Mul
    auto mul_1 = builder.node2("Mul", {cast_1, sigmoid_fp32});
    auto mul_1_bf16 = builder.node2("Cast", {mul_1});

    return Rule::create_rule(
        mul_1_bf16, [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
          // auto input = binder[input_->get_id()];
          std::vector<std::string> ns = vaip::dd::get_node_names(graph, binder);
          auto input_0_node = binder[input_0->get_id()];
          auto out_node = binder[mul_1_bf16->get_id()];
          auto out_shape = node_arg_get_shape_i64(*out_node.node_arg);

          std::vector<std::string> input_types{"bfloat16"};
          std::vector<std::string> output_types{"bfloat16"};
          NodeBuilder(*graph, *self)
              .set_input_node_args({input_0_node.node_arg}) //, &lrn_qdq_arg})
              .set_op_type("SILU", "com.xilinx")
              .clone_attrs(*out_node.node)
              .add("nodes", ns)
              .set_anchor_point1(*out_node.node)
              .add("in_dtypes", input_types)
              .add("out_dtypes", output_types)
              .add("shape", *(out_shape.get()))
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

DEFINE_VAIP_PASS(Silu, vaip_pass_dd_merge_silu)
