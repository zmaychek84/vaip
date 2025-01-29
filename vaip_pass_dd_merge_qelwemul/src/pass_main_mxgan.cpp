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

DEF_ENV_PARAM(DEBUG_DD_MERGE_QELWEMUL, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_DD_MERGE_QELWEMUL) >= n)

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
    { "name": "vaip_pass_dd_merge_qelwemul",
       "plugin": "vaip-pass_dd_merge_qelwemul",
       "disabled": false
    }
*/
namespace {
using namespace vaip_core;

// filter out constant mul?
bool check_if_ele_mul(const Graph& graph, const NodeArg* input_0,
                      const NodeArg* input_1) {
  return (!node_arg_is_constant(graph, *input_0)) &&
         (!node_arg_is_constant(graph, *input_1));
}

struct Dd_merge_qelwemul_mxgan {
  Dd_merge_qelwemul_mxgan(IPass& self) : self_{self} {}
  std::unique_ptr<Rule> create_rule(IPass* self) {

    auto com_microsoft_QuantizeLinear_0 =
        vaip::pattern_zoo::get_pattern("m_qelwemul");
    CHECK(com_microsoft_QuantizeLinear_0 != nullptr)
        << "Pattern returned is null";

    return Rule::create_rule(
        com_microsoft_QuantizeLinear_0,
        [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
          auto in_node_0 = binder["input_1"];
          auto in_0_scale_node = binder["constant_2"];
          auto in_0_zp_node = binder["constant_3"];
          auto in_0_scale = node_arg_get_const_data_as_float(
              *graph, *in_0_scale_node.node_arg);
          auto in_0_zp =
              vaip::dd::get_zp_from_node(*graph, *in_0_zp_node.node_arg);

          auto in_node_1 = binder["input_0"];
          auto in_1_scale_node = binder["constant_0"];
          auto in_1_zp_node = binder["constant_1"];
          auto in_1_scale = node_arg_get_const_data_as_float(
              *graph, *in_1_scale_node.node_arg);
          auto in_1_zp =
              vaip::dd::get_zp_from_node(*graph, *in_1_zp_node.node_arg);

          if (!check_if_ele_mul(*graph, in_node_0.node_arg,
                                in_node_1.node_arg)) {
            return false;
          }

          auto out_scale_node = binder["constant_4"];
          auto out_zp_node = binder["constant_5"];
          auto out_node = binder["com_microsoft_QuantizeLinear_0"];
          auto out_scale = node_arg_get_const_data_as_float(
              *graph, *out_scale_node.node_arg);
          auto out_zp =
              node_arg_get_const_data_as_u16(*graph, *out_zp_node.node_arg);
          auto out_shape = node_arg_get_shape_i64(*out_node.node_arg);

          std::vector<std::string> ns = vaip::dd::get_node_names(graph, binder);
          //   MY_LOG(1) << "found match at " << ns.front();
          //   auto [success, modified_b_input_scale, modified_b_input_zp] =
          //       get_qlinear_param(*graph, in_node_0.node);
          //   if (!success) {
          //     MY_LOG(1) << "parameters for modified b input not found";
          //   }

          std::vector<float> input_q_params{in_1_scale, float(in_1_zp),
                                            in_0_scale, float(in_0_zp)};
          std::vector<float> output_q_params{out_scale, float(out_zp)};
          auto [a_scale, a_zp] =
              vaip::dd::qmatmulcalc::calc_lrn_coeff(in_0_scale, in_0_zp);
          auto [b_scale, b_zp] =
              vaip::dd::qmatmulcalc::calc_lrn_coeff(in_1_scale, in_1_zp);
          auto [final_out_scale, final_out_zp] =
              vaip::dd::qmatmulcalc::calc_lrn_coeff(1 / out_scale, out_zp);

          int32_t amat_uint16 = 0; // is_matA_uint16
          int32_t cmat_uint16 = 0; // is_matC_uint16

          std::vector<int32_t> elt_coeffs(16, 0);
          elt_coeffs[0] = b_scale;
          elt_coeffs[1] = b_zp;
          elt_coeffs[2] = a_scale;
          elt_coeffs[3] = a_zp;
          elt_coeffs[4] = final_out_scale;
          elt_coeffs[5] = final_out_zp;
          elt_coeffs[6] = amat_uint16;
          elt_coeffs[7] = cmat_uint16;
          auto node_name = node_arg_get_name(*out_node.node_arg);

          if (node_name != "1016_QuantizeLinear_Output")
            return false;
          std::cout << node_name << std::endl; //
          std::string elt_coeff_name = std::string(node_name + "_qdq_");
          auto& elt_arg = vaip::dd::insert_named_tensor_in_graph<int32_t>(
              graph, elt_coeff_name, elt_coeffs,
              std::vector({(int64_t)elt_coeffs.size()}));

          // hard code for mzdk5, may need to change
          std::vector<std::string> input_types{"bfloat16", "uint16", "int32"};
          std::vector<std::string> output_types{"bfloat16"};

          // input order is changed
          NodeBuilder(*graph, *self)
              .set_input_node_args(
                  {in_node_1.node_arg, in_node_0.node_arg, &elt_arg})
              .set_op_type("QELWEMUL_mxgan", "com.xilinx")
              .add("nodes", ns)
              .add("input_shape", *out_shape) // same as python
              .add("orig_output_shape", *out_shape)
              .add("input_q_params", input_q_params)
              .add("output_q_params", output_q_params)
              //   .add("modified_b_input_scale",
              //        std::to_string(modified_b_input_scale))
              //   .add("modified_b_input_zp",
              //   std::to_string(modified_b_input_zp))
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

DEFINE_VAIP_PASS(Dd_merge_qelwemul_mxgan, vaip_pass_dd_merge_qelwemul_mxgan)