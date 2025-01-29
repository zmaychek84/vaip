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

#include "qconv2matmul_common.hpp"
#include "vaip/vaip.hpp"
#include "vitis/ai/env_config.hpp"

DEF_ENV_PARAM(DEBUG_DD_MERGE_QCONV2MATMUL, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_DD_MERGE_QCONV2MATMUL) >= n)

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
    { "name": "vaip_pass_dd_merge_qconv2matmul_2",
       "plugin": "vaip-pass_dd_merge_qconv2matmul_2",
       "disabled": false
    }
*/
namespace {
using namespace vaip_core;
struct Dd_merge_qconv2matmul_2 {
  Dd_merge_qconv2matmul_2(IPass& self) : self_{self} {}
  std::unique_ptr<Rule> create_rule(IPass* self) {
    auto com_microsoft_QuantizeLinear_1 =
        vaip::pattern_zoo::get_pattern("m_qconv2matmul_2");
    CHECK(com_microsoft_QuantizeLinear_1 != nullptr)
        << "Pattern returned is null";

    return Rule::create_rule(
        com_microsoft_QuantizeLinear_1,
        [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
          // Conv -> Transpose
          auto in_node = binder["input_0"];
          auto in_scale_node = binder["constant_0"];
          auto in_zp_node = binder["constant_1"];
          auto out_scale_node = binder["constant_5"];
          auto out_zp_node = binder["constant_6"];
          auto act_node = binder["com_microsoft_DequantizeLinear_0"];
          auto kernel_node = binder["com_microsoft_DequantizeLinear_1"];
          // weights
          auto wt_node = binder["constant_2"];
          auto wt_scale_node = binder["constant_3"];
          auto wt_zp_node = binder["constant_4"];

          auto conv_node = binder["Conv_0"];
          auto transpose_node = binder["Transpose_0"];
          auto out_node = binder["com_microsoft_QuantizeLinear_1"];

          std::vector<std::string> ns = vaip::dd::get_node_names(graph, binder);
          MY_LOG(1) << "found match at " << ns.front();

          auto in_scale =
              node_arg_get_const_data_as_float(*graph, *in_scale_node.node_arg);
          auto in_zero_point =
              node_arg_get_const_data_as_u16(*graph, *in_zp_node.node_arg);
          auto out_scale = node_arg_get_const_data_as_float(
              *graph, *out_scale_node.node_arg);
          auto out_zero_point =
              node_arg_get_const_data_as_u16(*graph, *out_zp_node.node_arg);
          auto act_in_shape = node_arg_get_shape_i64(*act_node.node_arg);
          auto kernel_shape = node_arg_get_shape_i64(*kernel_node.node_arg);
          auto out_shape = node_arg_get_shape_i64(*conv_node.node_arg);
          auto orig_output_shape = node_arg_get_shape_i64(*out_node.node_arg);
          auto wt_name = node_arg_get_name(*wt_node.node_arg);
          auto transpose_perm_span =
              node_get_attr_ints(*transpose_node.node, "perm");
          std::vector<int64_t> transpose_perm(transpose_perm_span.begin(),
                                              transpose_perm_span.end());

          auto weights =
              node_arg_get_const_data_as_u8s(*graph, *wt_node.node_arg);
          auto weights_scale =
              node_arg_get_const_data_as_float(*graph, *wt_scale_node.node_arg);
          auto weights_zero_point =
              node_arg_get_const_data_as_u8(*graph, *wt_zp_node.node_arg);

          std::vector<float> input_q_params{in_scale, float(in_zero_point)};
          std::vector<float> output_q_params{out_scale, float(out_zero_point)};

          gsl::span<const int32_t> bias;

          auto [C0, C1, C2, conv_shift, shft_c2] =
              vaip::dd::qmatmulcalc::dq_uint16A_uint8W_conv_q_param_gen(
                  in_scale, in_zero_point, weights, weights_scale,
                  weights_zero_point, *kernel_shape, bias, 0.0f, 0, out_scale,
                  out_zero_point);

          auto node_name = node_arg_get_name(*out_node.node_arg);
          auto& input_c0_arg = vaip::dd::insert_named_tensor_in_graph<int64_t>(
              graph, node_name + "_c0_", C0, std::vector({(int64_t)C0.size()}));
          std::vector<int32_t> input_qdq(16, 0);
          input_qdq[2] = static_cast<int32_t>(C1);
          input_qdq[3] = static_cast<int32_t>(C2);
          input_qdq[8] = static_cast<int32_t>(shft_c2);
          input_qdq[9] = static_cast<int32_t>(conv_shift);
          input_qdq[10] = 1;
          auto& input_qdq_arg = vaip::dd::insert_named_tensor_in_graph<int32_t>(
              graph, node_name + "_qdq_", input_qdq,
              std::vector({(int64_t)input_qdq.size()}));

          auto [new_input_arg, extra_names] =
              qconv2matmul::find_new_input(in_node.node);
          if (!new_input_arg) {
            MY_LOG(1) << "new input to skip 3 parents not found, using "
                         "original input";
            new_input_arg = in_node.node_arg;
          }
          ns.insert(ns.end(), extra_names.begin(), extra_names.end());

          // hard code for mzdk5, may need to change
          std::vector<std::string> input_types{"uint16", "uint8", "int64",
                                               "int32"};
          std::vector<std::string> output_types{"uint16"};

          auto [nchw_shape, nhwc_shape] =
              qconv2matmul::get_NCHW_NHWC(*act_in_shape);

          NodeBuilder(*graph, *self)
              .set_input_node_args({new_input_arg, wt_node.node_arg,
                                    &input_c0_arg, &input_qdq_arg})
              .set_op_type("QConv2MatMul", "com.xilinx")
              .clone_attrs(*conv_node.node)
              .add("nodes", ns)
              .add("orig_input_shape", *act_in_shape)
              .add("input_shape", *act_in_shape)
              .add("weight_shape", *kernel_shape)
              .add("output_shape", *out_shape)
              .add("zero_point", int64_t(in_zero_point))
              .add("wt_name", wt_name)
              .add("orig_output_shape", *orig_output_shape)
              .add("transpose_perm", transpose_perm)
              .add("in_dtypes", input_types)
              .add("out_dtypes", output_types)
              .add("input_format", "NHWC")
              .add("input_q_params", input_q_params)
              .add("output_q_params", output_q_params)
              .add("C1", std::to_string(C1))
              .add("C2", std::to_string(C2))
              .add("qconv_pattern", "2")
              .add("shift_conv", std::to_string(conv_shift))
              .add("shift_final", std::to_string(shft_c2))
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

DEFINE_VAIP_PASS(Dd_merge_qconv2matmul_2, vaip_pass_dd_merge_qconv2matmul_2)
