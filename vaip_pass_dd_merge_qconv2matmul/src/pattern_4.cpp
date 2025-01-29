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

DEF_ENV_PARAM(DEBUG_DD_MERGE_QCONV2MATMUL, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_DD_MERGE_QCONV2MATMUL) >= n)

/**
 *  pattern:
 *  dq --> conv --> q --> dq --> trans -> q
 *
 */

// add the following line in your vaip_config.json
/*
    { "name": "vaip_pass_dd_merge_qconv2matmul_4",
       "plugin": "vaip-pass_dd_merge_qconv2matmul_4",
       "disabled": false
    }
*/
namespace {
using namespace vaip_core;
struct Dd_merge_qconv2matmul_4 {
  Dd_merge_qconv2matmul_4(IPass& self) : self_{self} {}
  std::unique_ptr<Rule> create_rule(IPass* self) {
    auto conv2matmul = vaip::pattern_zoo::get_pattern("m_qconv2matmul_4");
    // std::cout << "Reached the pass" << std::endl;
    CHECK(conv2matmul != nullptr) << "Pattern returned is null";

    return Rule::create_rule(

        conv2matmul, [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
          auto in_node = binder["input_0"];
          auto in_scale_node = binder["constant_0"];
          auto in_zp_node = binder["constant_1"];
          auto out_node = binder["com_microsoft_QuantizeLinear_1"];
          auto wts_node = binder["constant_2"];
          auto wts_scale_node = binder["constant_3"];
          auto wts_zp_node = binder["constant_4"];
          auto out_scale_node = binder["constant_9"];
          auto out_zp_node = binder["constant_10"];

          std::vector<std::string> ns = vaip::dd::get_node_names(graph, binder);

          auto node_name = node_arg_get_name(*out_node.node_arg);

          MY_LOG(1) << "found match at " << ns.front();
          // Extracting the scales and zero points for inputs and weights
          auto in_scale =
              node_arg_get_const_data_as_float(*graph, *in_scale_node.node_arg);
          auto in_zero_point =
              vaip::dd::get_zp_from_node(*graph, *in_zp_node.node_arg);
          auto out_shape = node_arg_get_shape_i64(*out_node.node_arg);
          auto o_s = *out_shape;
          auto in0_shape = node_arg_get_shape_i64(*in_node.node_arg);

          auto w_shape = node_arg_get_shape_i64(*wts_node.node_arg);
          auto wts_shape = *w_shape;
          auto w_sc_shape = node_arg_get_shape_i64(*wts_scale_node.node_arg);
          auto weight_data_type = node_arg_get_element_type(*wts_node.node_arg);
          auto w_zp_shape = node_arg_get_shape_i64(*wts_zp_node.node_arg);
          auto wt_name = node_arg_get_name(*wts_node.node_arg);
          auto weight_zp_type =
              node_arg_get_element_type(*wts_zp_node.node_arg);
          auto out_scale = node_arg_get_const_data_as_float(
              *graph, *out_scale_node.node_arg);
          auto out_zero_point =
              node_arg_get_const_data_as_u16(*graph, *out_zp_node.node_arg);

          // Initializing the weights and scales and zero points to add them as
          // tensors in node builder

          gsl::span<const int8_t> weights;
          gsl::span<const float> weights_scale;
          float weights_sc;
          gsl::span<const int8_t> weights_zero_point;
          int block_size = 128;
          std::vector<float> wts_sc_vec;
          std::vector<float> wts_sc_vec_1;
          std::vector<float> wts_sc_vec_bs_1;
          std::vector<int8_t> wts_zp_vec_1;
          std::vector<float> wts_sc_vec_bs;
          int8_t weights_zp;
          std::vector<int8_t> wts_zp_vec;

          std::vector<float> input_q_params{in_scale, float(in_zero_point)};
          std::vector<float> output_q_params{out_scale, float(out_zero_point)};

          // Weight DataType = 3 is for int8 weights and else part takes care of
          // int4 weights
          if (weight_data_type == 3) {
            return false;
            weights =
                node_arg_get_const_data_as_i8s(*graph, *wts_node.node_arg);
            weights_sc = node_arg_get_const_data_as_float(
                *graph, *wts_scale_node.node_arg);
            weights_zp =
                node_arg_get_const_data_as_i8(*graph, *wts_zp_node.node_arg);

            gsl::span<const int32_t> bias;

            auto [C0, C1, C2, conv_shift, shft_c2] =
                vaip::dd::qmatmulcalc::dq_uint16A_int8W_conv_q_param_gen(
                    in_scale, in_zero_point, weights, weights_sc, weights_zp,
                    wts_shape, bias, 0.0f, 0, out_scale, out_zero_point);

            auto node_name = node_arg_get_name(*out_node.node_arg);
            auto& input_c0_arg =
                vaip::dd::insert_named_tensor_in_graph<int64_t>(
                    graph, node_name + "_c0_", C0,
                    std::vector({(int64_t)C0.size()}));
            std::vector<int32_t> input_qdq(16, 0);
            input_qdq[2] = static_cast<int32_t>(C1);
            input_qdq[3] = static_cast<int32_t>(C2);
            input_qdq[8] = static_cast<int32_t>(shft_c2);
            input_qdq[9] = static_cast<int32_t>(conv_shift);
            auto& input_qdq_arg =
                vaip::dd::insert_named_tensor_in_graph<int32_t>(
                    graph, node_name + "_qdq_", input_qdq,
                    std::vector({(int64_t)input_qdq.size()}));

            std::vector<std::string> input_types{"uint16", "int8", "int64",
                                                 "int32"};
            std::vector<std::string> output_types{"uint16"};

            NodeBuilder(*graph, *self)
                .set_input_node_args({in_node.node_arg, wts_node.node_arg,
                                      &input_c0_arg, &input_qdq_arg})
                .set_op_type("QConv2MatMul", "com.xilinx")
                //.clone_attrs(*conv_node.node)
                .add("nodes", ns)
                .add("orig_input_shape", *in0_shape)
                .add("input_shape", *in0_shape)
                .add("weight_shape", *w_shape)
                .add("output_shape", *out_shape)
                .add("zero_point", int64_t(in_zero_point))
                .add("wt_name", wt_name)
                .add("orig_output_shape", *out_shape)
                //.add("transpose_perm", transpose_perm)
                .add("in_dtypes", input_types)
                .add("out_dtypes", output_types)
                .add("input_format", "NCHW")
                .add("design_param", "4x4PSU")
                .add("input_q_params", input_q_params)
                .add("output_q_params", output_q_params)
                .add("C1", std::to_string(C1))
                .add("C2", std::to_string(C2))
                .add("qconv_pattern", "4")
                .add("shift_conv", std::to_string(conv_shift))
                .add("shift_final", std::to_string(shft_c2))
                .set_anchor_point1(*out_node.node)
                .build();
          }

          else { // uint16xint4
            weights =
                node_arg_get_const_data_as_i4s(*graph, *wts_node.node_arg);
            weights_scale = node_arg_get_const_data_as_floats(
                *graph, *wts_scale_node.node_arg);
            weights_zero_point =
                node_arg_get_const_data_as_i4s(*graph, *wts_zp_node.node_arg);
            size_t num_of_weights =
                std::accumulate(wts_shape.begin(), wts_shape.end(), (size_t)1,
                                std::multiplies<int64_t>());
            std::vector<int8_t> wts_vec;
            if (weight_data_type != 3) {
              wts_vec = vaip::dd::unpack(weights, num_of_weights);
            }

            size_t num_of_weights_zp = wts_shape[0];
            std::vector<int8_t> wts_zp_vec_orig;
            if (weight_zp_type != 3) {
              wts_zp_vec_orig =
                  vaip::dd::unpack(weights_zero_point, num_of_weights_zp);
            }

            std::string wts_zp_initializer_name =
                node_arg_get_name(*wts_zp_node.node_arg) + "0";
            const std::vector<int64_t> wts_zp_initializer_shape = {
                (int64_t)wts_shape[0]};
            const std::vector<int64_t> wts_zp_initializer_shape_i8 = {
                (int64_t)wts_shape[0]};
            NodeArg& wts_zp_arg =
                (weight_data_type == 3)
                    ? vaip::dd::insert_named_tensor_in_graph<int8_t>(
                          graph, wts_zp_initializer_name, wts_zp_vec,
                          wts_zp_initializer_shape_i8)
                    : vaip::dd::insert_named_tensor_in_graph<int8_t>(
                          graph, wts_zp_initializer_name, wts_zp_vec_1,
                          wts_zp_initializer_shape);

            std::string wts_initializer_name =
                node_arg_get_name(*wts_node.node_arg) + "0";
            NodeArg& wts_arg =
                (weight_data_type == 3)
                    ? vaip::dd::insert_named_tensor_in_graph<int8_t>(
                          graph, wts_initializer_name, wts_vec, wts_shape)
                    : vaip::dd::insert_named_tensor_in_graph<int8_t>(
                          graph, wts_initializer_name, wts_vec, wts_shape);

            std::vector<std::string> input_types{"uint16", "int4",  "int64",
                                                 "int32",  "int32", "int32"};
            std::vector<std::string> output_types{"uint16"};
            gsl::span<const int32_t> bias;
            auto [C0, C1, C2, conv_shift, shft_c2] =
                vaip::dd::qmatmulcalc::dq_uint16A_int4W_conv_chwise_q_param_gen(
                    in_scale, in_zero_point, wts_vec, weights_scale,
                    wts_zp_vec_orig, wts_shape, bias, 0.0f, 0, out_scale,
                    out_zero_point);

            auto& input_c0_arg =
                vaip::dd::insert_named_tensor_in_graph<int64_t>(
                    graph, node_name + "_c0_", C0,
                    std::vector({(int64_t)C0.size()}));
            auto& input_c1_arg =
                vaip::dd::insert_named_tensor_in_graph<int32_t>(
                    graph, node_name + "_c1_", C1,
                    std::vector({(int64_t)C1.size()}));
            auto& input_c2_arg =
                vaip::dd::insert_named_tensor_in_graph<int32_t>(
                    graph, node_name + "_c2", C2,
                    std::vector({(int64_t)C2.size()}));

            std::vector<int32_t> input_qdq(16, 0);
            input_qdq[8] = static_cast<int32_t>(shft_c2);
            input_qdq[9] = static_cast<int32_t>(conv_shift);
            input_qdq[10] = 4; // 0 for int8 , greater than 1 for int4
            auto& input_qdq_arg =
                vaip::dd::insert_named_tensor_in_graph<int32_t>(
                    graph, node_name + "_qdq_", input_qdq,
                    std::vector({(int64_t)input_qdq.size()}));

            NodeBuilder(*graph, *self)
                .set_input_node_args({in_node.node_arg, wts_node.node_arg,
                                      &input_c0_arg, &input_qdq_arg,
                                      &input_c1_arg, &input_c2_arg})
                .set_op_type("QConv2MatMul", "com.xilinx")
                .add("nodes", ns)
                .add("orig_input_shape", *in0_shape)
                .add("input_shape", *in0_shape)
                .add("weight_shape", *w_shape)
                .add("output_shape", *out_shape)
                .add("zero_point", int64_t(in_zero_point))
                .add("wt_name", wt_name)
                .add("input_format", "NCHW")
                .add("design_param", "4x4PSU")
                .add("in_dtypes", input_types)
                .add("out_dtypes", output_types)
                .add("orig_output_shape", *out_shape)
                .add("input_q_params", input_q_params)
                .add("qconv_pattern", "4")
                .add("output_q_params", output_q_params)
                .set_anchor_point1(*out_node.node)
                .build();
          }
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

DEFINE_VAIP_PASS(Dd_merge_qconv2matmul_4, vaip_pass_dd_merge_qconv2matmul_4)
