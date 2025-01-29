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
#include <functional>
#include <glog/logging.h>
#include <numeric>

DEF_ENV_PARAM(DEBUG_DD_MERGE_DPS, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_DD_MERGE_DPS) >= n)

namespace {
using namespace vaip_core;
struct DPS {
  DPS(IPass& self) : self_{self} {}

  std::unique_ptr<Rule> create_rule(IPass* self) {

    auto com_microsoft_QuantizeLinear_1 =
        vaip::pattern_zoo::get_pattern("m_dps");
    CHECK(com_microsoft_QuantizeLinear_1 != nullptr)
        << "Pattern returned is null";

    return Rule::create_rule(
        com_microsoft_QuantizeLinear_1,
        [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
          std::vector<std::string> ns = vaip::dd::get_node_names(graph, binder);

          MY_LOG(1) << "Pattern Matched : "
                    << node_arg_get_name(
                           *binder["com_microsoft_QuantizeLinear_1"].node_arg)
                    << std::endl;

          // Node extraction
          auto input = binder["input_0"];

          auto in_scale_node = binder["constant_0"];
          auto in_zp_node = binder["constant_1"];

          auto wt_node = binder["constant_5"];
          auto wt_scale_node = binder["constant_6"];
          auto wt_zp_node = binder["constant_7"];

          auto bias_node = binder["constant_2"];
          auto bias_scale_node = binder["constant_3"];
          auto bias_zp_node = binder["constant_4"];

          // Scale & zero point extraction
          auto output_node = binder["com_microsoft_QuantizeLinear_1"];

          auto out_scale_node = binder["constant_10"];
          auto out_zp_node = binder["constant_11"];
          auto out_scale = node_arg_get_const_data_as_float(
              *graph, *out_scale_node.node_arg);
          auto out_zp =
              node_arg_get_const_data_as_u16(*graph, *out_zp_node.node_arg);
          auto out_shape = node_arg_get_shape_i64(*output_node.node_arg);
          std::vector<float> output_q_params{out_scale, float(out_zp)};
          float in_sc =
              node_arg_get_const_data_as_float(*graph, *in_scale_node.node_arg);
          uint16_t in_zp =
              vaip::dd::get_zp_from_node(*graph, *in_zp_node.node_arg);

          auto wt = vaip::dd::get_const_as_uint16_t(*graph, *wt_node.node_arg);
          float wt_sc =
              node_arg_get_const_data_as_float(*graph, *wt_scale_node.node_arg);
          uint16_t wt_zp =
              vaip::dd::get_zp_from_node(*graph, *wt_zp_node.node_arg);

          auto bias_gsl = node_arg_get_const_data_as_i32s(
              *graph, *bias_node.node_arg); // int32
          std::vector<int32_t> bias(bias_gsl.begin(), bias_gsl.end());
          float bias_sc = node_arg_get_const_data_as_float(
              *graph, *bias_scale_node.node_arg);
          auto bias_zp =
              node_arg_get_const_data_as_i32(*graph, *bias_zp_node.node_arg);

          // Input args
          std::vector<uint16_t> dq_wt =
              vaip::dd::qmatmulcalc::dq_vec_to_bf16(wt, wt_sc, wt_zp);

          std::vector<uint16_t> dq_bias =
              vaip::dd::qmatmulcalc::dq_vec_to_bf16(bias, bias_sc, bias_zp);

          std::vector<int32_t> qdq_params(16, 0);
          qdq_params[3] = vaip::dd::qmatmulcalc::float_to_bfloat16(
              static_cast<float>(in_zp));
          qdq_params[4] = vaip::dd::qmatmulcalc::float_to_bfloat16(in_sc);
          qdq_params[5] = 1;

          // Adding input args to graph
          std::string name_wt_dq =
              std::string(node_arg_get_name(*wt_node.node_arg) + "_dq");
          auto& wt_dq_arg = vaip::dd::insert_named_tensor_in_graph<uint16_t>(
              graph, name_wt_dq, dq_wt,
              std::vector({(int64_t)dq_wt.size(), (int64_t)1}));

          std::string name_bias_dq =
              std::string(node_arg_get_name(*bias_node.node_arg) + "_dq");
          auto& bias_dq_arg = vaip::dd::insert_named_tensor_in_graph<uint16_t>(
              graph, name_bias_dq, dq_bias,
              std::vector({(int64_t)dq_bias.size()}));

          std::string name_qdq = std::string(
              node_arg_get_name(*bias_node.node_arg) + "_qdq_params");
          auto& qdq_param_arg = vaip::dd::insert_named_tensor_in_graph<int32_t>(
              graph, name_qdq, qdq_params,
              std::vector({(int64_t)qdq_params.size()}));

          // Add node to graph
          std::vector<std::string> in_dtypes{"uint16", "bfloat16", "bfloat16",
                                             "int32"}; //, "bfloat16"};
          std::vector<std::string> out_dtypes{"bfloat16"};

          NodeBuilder(*graph, *self)
              .set_input_node_args(
                  {input.node_arg, &wt_dq_arg, &bias_dq_arg, &qdq_param_arg})
              .set_op_type("DPS", "com.xilinx")
              // .clone_attrs(*q_node.node)
              .add("nodes", ns)
              .set_anchor_point1(*output_node.node)
              .add("in_dtypes", in_dtypes)
              // .add("Node_dtype", change_nodetypes(*q_node.node_arg))
              .add("out_dtypes", out_dtypes)
              //.add("input_q_params", input_q_params)
              .add("output_q_params", output_q_params)
              // .add("orig_output_shape", *(q_shape.get()))
              .add("orig_output_shape",
                   std::vector<int64_t>{(int64_t)1, (int64_t)1})
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

DEFINE_VAIP_PASS(DPS, vaip_pass_dd_merge_dps)
