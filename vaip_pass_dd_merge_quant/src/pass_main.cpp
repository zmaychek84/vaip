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
#include "vaip/pattern_zoo.hpp"
#ifdef __GNUC__
#  pragma GCC diagnostic pop
#endif

#include "vaip/vaip.hpp"
#include "vitis/ai/env_config.hpp"

DEF_ENV_PARAM(DEBUG_DD_MERGE_QUANT, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_DD_MERGE_QUANT) >= n)

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
    { "name": "vaip_pass_dd_merge_quant",
       "plugin": "vaip-pass_dd_merge_quant",
       "disabled": false
    }
*/
namespace {
using namespace vaip_core;
struct Dd_merge_quant {
  Dd_merge_quant(IPass& self) : self_{self} {}
  std::unique_ptr<Rule> create_rule(IPass* self) {
    auto dq = vaip::pattern_zoo::get_pattern("m_q");
    CHECK(dq != nullptr) << "Pattern returned is null";

    return Rule::create_rule(
        dq, [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
          auto in_node = binder["input"];
          auto in_scale_node = binder["in_s"];
          auto in_zp_node = binder["in_z"];
          auto out_node = binder["dq"];

          std::vector<std::string> ns = vaip::dd::get_node_names(graph, binder);
          auto node_name = node_arg_get_name(*out_node.node_arg);
          // excluded quantop for mzdk5, should go to qop
          if (node_name == "input_1_QuantizeLinear_Output" ||
              node_name == "input_2_QuantizeLinear_Output" ||
              node_name == "input_1_q_to_dq") {
            MY_LOG(1) << "excluding " << node_name;
            return false;
          }
          MY_LOG(1) << "found match at " << ns.front();

          auto in_scale =
              node_arg_get_const_data_as_float(*graph, *in_scale_node.node_arg);
          auto in_zero_point =
              node_arg_get_const_data_as_u16(*graph, *in_zp_node.node_arg);
          std::vector<float> input_q_params{in_scale, float(in_zero_point)};

          auto [a_scale, a_zp] = vaip::dd::qmatmulcalc::calc_lrn_coeff(
              1.0f / in_scale, in_zero_point);

          std::vector<int32_t> qdq_tensor(16, 0);
          qdq_tensor[0] = a_zp;
          qdq_tensor[1] = a_scale;
          std::string coeff_name = std::string(node_name + "_qdq_");
          auto& qdq_arg = vaip::dd::insert_named_tensor_in_graph(
              graph, coeff_name, qdq_tensor,
              std::vector({(int64_t)qdq_tensor.size()}));

          // hard code for mzdk5, may need to change
          std::vector<std::string> input_types{"bfloat16", "int32"};
          std::vector<std::string> output_types{"uint16"};

          NodeBuilder(*graph, *self)
              .set_input_node_args({in_node.node_arg, &qdq_arg})
              .set_op_type("QuantOP", "com.xilinx")
              .add("nodes", ns)
              .add("in_dtypes", input_types)
              .add("out_dtypes", output_types)
              .add("input_q_params", input_q_params)
              .add("output_q_params", input_q_params)
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

DEFINE_VAIP_PASS(Dd_merge_quant, vaip_pass_dd_merge_quant)
