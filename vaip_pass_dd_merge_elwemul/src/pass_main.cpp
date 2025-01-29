/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */
#include <glog/logging.h>

#ifdef __GNUC__
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wpedantic"
#  pragma GCC diagnostic ignored "-Wconversion"
#endif

#include "vaip/dd/coeffs.hpp"
#include "vaip/dd/dd_utils.hpp"

#ifdef __GNUC__
#  pragma GCC diagnostic pop
#endif

#include "vaip/pattern_zoo.hpp"
#include "vaip/vaip.hpp"
#include "vitis/ai/env_config.hpp"

DEF_ENV_PARAM(DEBUG_DD_MERGE_ELWEMUL, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_DD_MERGE_ELWEMUL) >= n)
namespace {
using namespace vaip_core;

// filter out constant mul?
bool check_if_ele_mul(const Graph& graph, const NodeArg* input_0,
                      const NodeArg* input_1) {
  return (!node_arg_is_constant(graph, *input_0)) &&
         (!node_arg_is_constant(graph, *input_1));
}

struct Dd_merge_elwemul {
  Dd_merge_elwemul(IPass& self) : self_{self} {}
  std::unique_ptr<Rule> create_rule(IPass* self) {

    auto builder = PatternBuilder();
    auto input_0 = builder.wildcard();
    auto input_1 = builder.wildcard();

    auto input_0_cast = builder.node2("Cast", {input_0});
    // Up out cast fp32
    auto input_1_cast = builder.node2("Cast", {input_1});
    // Mul
    auto mul = builder.node2("Mul", {input_0_cast, input_1_cast});
    // DP in cast
    auto mul_cast = builder.node2("Cast", {mul});
    CHECK(mul_cast != nullptr) << "Pattern returned is null";

    return Rule::create_rule(
        mul_cast, [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
          auto input_0_node = binder[input_0->get_id()];
          auto input_1_node = binder[input_1->get_id()];
          auto out_node = binder[mul_cast->get_id()];

          if (!check_if_ele_mul(*graph, input_0_node.node_arg,
                                input_1_node.node_arg)) {
            return false;
          }

          std::vector<std::string> ns = vaip::dd::get_node_names(graph, binder);
          MY_LOG(1) << "found match at " << ns.front();

          auto node_name = node_arg_get_name(*out_node.node_arg);
          std::vector<std::string> input_types{"bfloat16", "bfloat16"};
          std::vector<std::string> output_types{"bfloat16"};

          // input order is changed
          NodeBuilder(*graph, *self)
              .set_input_node_args(
                  {input_0_node.node_arg, input_1_node.node_arg})
              .set_op_type("ELWMUL", "com.xilinx")
              .add("nodes", ns)
              .add("in_dtypes", input_types)
              .add("out_dtypes", output_types)
              .set_anchor_point1(*out_node.node)
              .build();
          return true;
        });
  }
  // apply the rule
  void process(IPass& self, Graph& graph) {
    MY_LOG(1) << self_.get_pass_proto().name() << "["
              << self_.get_pass_proto().plugin() << "] start processing graph";
    create_rule(&self)->apply(&graph);
    MY_LOG(1) << self_.get_pass_proto().name() << "["
              << self_.get_pass_proto().plugin() << "] finish processing graph";
  }

  IPass& self_;
};
} // namespace

DEFINE_VAIP_PASS(Dd_merge_elwemul, vaip_pass_dd_merge_elwemul)
