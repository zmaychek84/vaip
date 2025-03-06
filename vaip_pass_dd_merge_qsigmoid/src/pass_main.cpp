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

#ifdef __GNUC__
#  pragma GCC diagnostic pop
#endif
#include "vaip/pattern_zoo.hpp"
#include "vaip/vaip.hpp"
#include "vitis/ai/env_config.hpp"
#include <glog/logging.h>

DEF_ENV_PARAM(DEBUG_DD_MERGE_QSIGMOID, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_DD_MERGE_QSIGMOID) >= n)

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
    { "name": "vaip_pass_dd_merge_qsigmoid",
       "plugin": "vaip-pass_dd_merge_qsigmoid",
       "disabled": false
    }
*/
namespace {
using namespace vaip_core;
struct Dd_merge_qsigmoid {
  Dd_merge_qsigmoid(IPass& self) : self_{self} {}
  std::unique_ptr<Rule> create_rule(IPass* self) {

    auto com_microsoft_DequantizeLinear_2 =
        vaip::pattern_zoo::get_pattern("m_QSigmoid");
    CHECK(com_microsoft_DequantizeLinear_2 != nullptr)
        << "Pattern returned is null";

    return Rule::create_rule(
        com_microsoft_DequantizeLinear_2,
        [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
          auto attr_nodes = vaip::dd::get_node_names(graph, binder);

          auto a_node = binder["input_0"];
          auto a_shape = node_arg_get_shape_i64(*a_node.node_arg);
          auto as_node = binder["constant_0"];
          auto az_node = binder["constant_1"];
          auto a_sc =
              node_arg_get_const_data_as_float(*graph, *as_node.node_arg);
          auto a_zp = vaip::dd::get_zp_from_node(*graph, *az_node.node_arg);

          auto q_node = binder["com_microsoft_DequantizeLinear_2"];
          auto q_shape = node_arg_get_shape_i64(*q_node.node_arg);
          auto node_name = node_arg_get_name(*q_node.node_arg);
          MY_LOG(1) << "Matched QSigmoid.";
          std::vector<std::string> input_types{"uint16", "uint16"};
          std::vector<std::string> output_types{"bfloat16"};
          std::vector<int32_t> qdq_out(16, 0);
          std::string qdq_out_name = std::string(node_name + "_qdq_");

          auto lrn_coeffs = vaip::dd::qmatmulcalc::calc_lrn_coeff(a_sc, a_zp);
          qdq_out[3] = std::get<1>(lrn_coeffs);
          qdq_out[4] = std::get<0>(lrn_coeffs);
          qdq_out[5] = 1; // enable/disable

          auto& qdq_arg = vaip::dd::insert_named_tensor_in_graph<int32_t>(
              graph, qdq_out_name, qdq_out,
              std::vector({(int64_t)qdq_out.size()}));

          auto new_node = NodeBuilder(*graph, *self);
          new_node.set_input_node_args({a_node.node_arg, &qdq_arg});
          new_node.set_op_type("QSigmoid", "com.xilinx");
          new_node.set_anchor_point1(*q_node.node);
          new_node.add("nodes", attr_nodes);
          new_node.add("in_dtypes", input_types);
          new_node.add("out_dtypes", output_types);
          //  new_node.add("input1_shape", *(a_shape.get()));
          //  new_node.add("orig_output_shape", *(q_shape.get()));
          new_node.build();
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

DEFINE_VAIP_PASS(Dd_merge_qsigmoid, vaip_pass_dd_merge_qsigmoid)
