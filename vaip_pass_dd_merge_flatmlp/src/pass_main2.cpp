/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */
#include "vaip/vaip.hpp"

#ifdef __GNUC__
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wpedantic"
#  pragma GCC diagnostic ignored "-Wconversion"
#endif

#include "vaip/dd/coeffs.hpp"
#include "vaip/dd/dd_utils.hpp"
#include "vaip/pattern_zoo.hpp"

#include "vitis/ai/env_config.hpp"
#include <glog/logging.h>
#include <numeric>

namespace {
using namespace vaip_core;
struct MergeFlatMLP2 {
  MergeFlatMLP2(IPass& self) : self_{self} {}

  static std::string shape_as_dd_string(const std::vector<int64_t>& shape) {
    std::stringstream ss;
    for (const auto& item : shape) {
      ss << item << " ";
    }
    return ss.str();
  }

  ////////////////// Pattern includes Input DQLs
  static std::unique_ptr<Rule> create_rule(IPass* self) {
    auto result = vaip::pattern_zoo::get_pattern("FlatMLP_2");
    CHECK(result != nullptr) << "Pattern returned is null";
    return Rule::create_rule(
        result, [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
          std::vector<std::string> ns = vaip::dd::get_node_names(graph, binder);
          //  Node and Data Access
          auto input_0 = binder["/model/layers.31/post_attention_layernorm/"
                                "Mul_1_output_0_QuantizeLinear_Output"];
          auto flat_mlp_node = binder
              ["/model/layers.31/mlp/Mul_2_output_0_QuantizeLinear_Output"];
          auto flat_mlp_shape = node_arg_get_shape_i64(*flat_mlp_node.node_arg);
          auto silu_node = binder["/model/layers.31/mlp/activation_fn/"
                                  "Mul_output_0_QuantizeLinear_Output"];
          auto in_node = binder["/model/layers.31/post_attention_layernorm/"
                                "Mul_1_output_0_QuantizeLinear_Output"];
          auto in0_shape = node_arg_get_shape_i64(*in_node.node_arg);

          auto in_scale_node =
              binder["/model/layers.31/post_attention_layernorm/"
                     "Mul_1_output_0_scale"];
          auto in_zp_node = binder["/model/layers.31/post_attention_layernorm/"
                                   "Mul_1_output_0_zero_point"];
          auto in_scale =
              node_arg_get_const_data_as_float(*graph, *in_scale_node.node_arg);
          auto in_zero_point =
              vaip::dd::get_zp_from_node(*graph, *in_zp_node.node_arg);

          auto weights_node = binder["onnx::MatMul_7734_quantized"];
          auto weights_data =
              node_arg_get_const_data_as_i4s(*graph, *weights_node.node_arg);
          auto w_shape = node_arg_get_shape_i64(*weights_node.node_arg);
          auto weights_shape = *w_shape;
          size_t num_weights =
              std::accumulate(weights_shape.begin(), weights_shape.end(),
                              (size_t)1, std::multiplies<int64_t>());

          auto wts_scale_node = binder["onnx::MatMul_7734_scale"];
          auto wts_zp_node = binder["onnx::MatMul_7734_zero_point"];
          auto weights_scale = node_arg_get_const_data_as_floats(
              *graph, *wts_scale_node.node_arg);
          auto weights_zero_point =
              node_arg_get_const_data_as_i4s(*graph, *wts_zp_node.node_arg);

          auto out_matmul_node =
              binder["/model/layers.31/mlp/gate_up_proj/"
                     "MatMul_output_0_QuantizeLinear_Output"];
          auto out_shape = node_arg_get_shape_i64(*out_matmul_node.node_arg);
          std::vector<int64_t> op_shape2(*out_shape);
          op_shape2[3] = op_shape2[3] / 2;
          auto node_name = node_arg_get_name(*out_matmul_node.node_arg);
          auto out_matmul_sc_node = binder
              ["/model/layers.31/mlp/gate_up_proj/MatMul_output_0nchw_scale"];
          auto out_matmul_zp_node = binder["/model/layers.31/mlp/gate_up_proj/"
                                           "MatMul_output_0nchw_zero_point"];

          auto out_matmul_sc_node1 =
              binder["/model/layers.31/mlp/Slice_1_output_0_scale"];
          auto out_matmul_zp_node1 =
              binder["/model/layers.31/mlp/Slice_1_output_0_zero_point"];

          auto out_sc_mm = node_arg_get_const_data_as_float(
              *graph, *out_matmul_sc_node.node_arg);
          auto out_zp_mm = node_arg_get_const_data_as_u16(
              *graph, *out_matmul_zp_node.node_arg);

          auto out_sc_mm1 = node_arg_get_const_data_as_float(
              *graph, *out_matmul_sc_node1.node_arg);
          auto out_zp_mm1 = node_arg_get_const_data_as_u16(
              *graph, *out_matmul_zp_node1.node_arg);
          // Input + output Q params
          std::vector<float> input_q_params{in_scale, float(in_zero_point)};
          std::vector<float> output_q_params{out_sc_mm, float(out_zp_mm)};
          std::vector<std::string> dtypes = {"uint16"};

          // Split into 2 parts. TODO. This actually needs to reflect Slice

          auto wts_vec =
              vaip::dd::unpack(weights_data, weights_data.size() * 2);

          std::vector<int8_t> wts_data_silu(weights_data.data(),
                                            weights_data.data() +
                                                weights_data.size() / 2);
          std::vector<int8_t> wts_data_matmm(
              weights_data.data() + weights_data.size() / 2,
              weights_data.data() + weights_data.size());

          auto w_shape2 = std::vector<int64_t>(weights_shape);
          w_shape2[0] = w_shape2[0] / 2; // TODO Assume half

          std::vector<int8_t> wts_vec_silu(wts_vec.begin(),
                                           wts_vec.begin() + num_weights / 2);
          std::vector<int8_t> wts_vec_matm(wts_vec.begin() + num_weights / 2,
                                           wts_vec.end());

          auto w_zp = vaip::dd::unpack(weights_zero_point,
                                       weights_zero_point.size() * 2);

          std::vector<int8_t> w_zp_silu(w_zp.begin(),
                                        w_zp.begin() + w_zp.size() / 2);
          std::vector<int8_t> w_zp_matm(w_zp.begin() + w_zp.size() / 2,
                                        w_zp.end());

          gsl::span<const float> w_sc_silu(weights_scale.data(),
                                           weights_scale.data() +
                                               weights_scale.size() / 2);
          gsl::span<const float> w_sc_matm(
              weights_scale.data() + weights_scale.size() / 2,
              weights_scale.data() + weights_scale.size());

          LOG(INFO) << vaip::dd::shape_as_string(weights_shape)
                    << weights_data.size();
          LOG(INFO) << "Output shape:" << vaip::dd::shape_as_string(*out_shape);

          auto wts_first_half = std::vector<int8_t>(
              weights_data.data(),
              weights_data.data() + weights_data.size() / 2);
          LOG(INFO) << "First half size " << wts_first_half.size();
          auto wts_second_half =
              std::vector<int8_t>(weights_data.data() + weights_data.size() / 2,
                                  weights_data.data() + weights_data.size());
          LOG(INFO) << "Second half size " << wts_second_half.size();

          gsl::span<const int32_t> bias;
          LOG(INFO) << "Bias size:" << bias.size();
          auto [C0, C1, C2, conv_shift, shft_c2] =
              vaip::dd::qmatmulcalc::dq_uint16A_int4W_conv_chwise_q_param_gen(
                  in_scale, in_zero_point, wts_vec, weights_scale, w_zp,
                  weights_shape, bias, 0.0f, 0, out_sc_mm, out_zp_mm);

          // auto [C0_mm_silu, C1_mm_silu, C2_mm_silu, conv_shift_mm_silu,
          //       shft_c2_mm_silu] =
          //     vaip::dd::qmatmulcalc::dq_uint16A_int4W_conv_chwise_q_param_gen(
          //         in_scale, in_zero_point, wts_vec_silu, w_sc_silu,
          //         w_zp_silu, w_shape2, bias, 0.0f, 0, out_sc_mm, out_zp_mm);
          std::vector<int64_t> C0_mm_silu(C0.data(), C0.data() + C0.size() / 2);
          std::vector<int64_t> C0_mm(C0.data() + C0.size() / 2,
                                     C0.data() + C0.size());

          std::vector<int32_t> C1_mm_silu(C1.data(), C1.data() + C1.size() / 2);
          std::vector<int32_t> C1_mm(C1.data() + C1.size() / 2,
                                     C1.data() + C1.size());

          std::vector<int32_t> C2_mm_silu(C2.data(), C2.data() + C2.size() / 2);
          std::vector<int32_t> C2_mm(C2.data() + C2.size() / 2,
                                     C2.data() + C2.size());

          std::vector<int32_t> input_qdq(16, 0);
          input_qdq[8] = static_cast<int32_t>(shft_c2);
          input_qdq[9] = static_cast<int32_t>(conv_shift);
          input_qdq[10] = 4;
          auto& input_qdq_arg = vaip::dd::insert_named_tensor_in_graph<int32_t>(
              graph, node_name + "_qdq_", input_qdq,
              std::vector({(int64_t)input_qdq.size()}));

          std::vector<int32_t> silu_qdq(16, 0);
          // TODO Silu
          auto silu_ip_sc_node =
              binder["/model/layers.31/mlp/Slice_output_0_scale"];
          auto silu_ip_sc = node_arg_get_const_data_as_float(
              *graph, *silu_ip_sc_node.node_arg);
          auto silu_ip_zp_node =
              binder["/model/layers.31/mlp/Slice_output_0_zero_point"];
          auto silu_ip_zp =
              vaip::dd::get_zp_from_node(*graph, *silu_ip_zp_node.node_arg);
          auto silu_op_sc_node =
              binder["/model/layers.31/mlp/activation_fn/Mul_output_0_scale"];
          auto silu_op_sc = node_arg_get_const_data_as_float(
              *graph, *silu_op_sc_node.node_arg);
          auto silu_op_zp_node = binder
              ["/model/layers.31/mlp/activation_fn/Mul_output_0_zero_point"];
          auto silu_op_zp =
              vaip::dd::get_zp_from_node(*graph, *silu_op_zp_node.node_arg);
          silu_qdq[0] = (int32_t)out_zp_mm;
          silu_qdq[1] =
              (int32_t)vaip::dd::qmatmulcalc::float_to_bfloat16(out_sc_mm);
          silu_qdq[2] = (int32_t)silu_op_zp;
          silu_qdq[3] =
              (int32_t)vaip::dd::qmatmulcalc::float_to_bfloat16(1 / silu_op_sc);

          auto& input_qdq_silu_arg =
              vaip::dd::insert_named_tensor_in_graph<int32_t>(
                  graph, node_name + "_qdq_silu", silu_qdq,
                  std::vector({(int64_t)silu_qdq.size()}));

          auto& input_c0_arg = vaip::dd::insert_named_tensor_in_graph<int64_t>(
              graph, node_name + "_c0_silu", C0_mm_silu,
              std::vector({(int64_t)C0_mm_silu.size()}));
          auto& input_c1_arg = vaip::dd::insert_named_tensor_in_graph<int32_t>(
              graph, node_name + "_c1_silu", C1_mm_silu,
              std::vector({(int64_t)C1_mm_silu.size()}));
          auto& input_c2_arg = vaip::dd::insert_named_tensor_in_graph<int32_t>(
              graph, node_name + "_c2_silu", C2_mm_silu,
              std::vector({(int64_t)C2_mm_silu.size()}));

          std::string wts_initializer_name =
              node_arg_get_name(*weights_node.node_arg) + "0";
          NodeArg& wts_arg_silu =
              vaip::dd::insert_named_tensor_in_graph<int8_t>(
                  graph, wts_initializer_name, wts_data_silu, w_shape2, true);
          std::vector<std::string> input_types{
              "uint16", "int4", "int64", "int32", "int32", "int32", "int32"};
          std::vector<std::string> output_types_silu{"bfloat16"};
          std::vector<std::string> output_types{"uint16"};
          auto matmulsilu_builder = NodeBuilder(*graph, *self);
          matmulsilu_builder.set_input_node_args(
              {input_0.node_arg, &wts_arg_silu, &input_c0_arg, &input_qdq_arg,
               &input_qdq_silu_arg, &input_c1_arg, &input_c2_arg});
          matmulsilu_builder.set_op_type("QConv2MatMulSilu", "com.xilinx");
          matmulsilu_builder.clone_data_type(*silu_node.node);
          matmulsilu_builder.clone_attrs(*silu_node.node);
          matmulsilu_builder.set_anchor_point1(*silu_node.node);
          matmulsilu_builder.add("nodes", ns);
          matmulsilu_builder.add("orig_input_shape", *in0_shape);
          matmulsilu_builder.add("input_shape", *in0_shape);
          matmulsilu_builder.add("weight_shape", w_shape2);
          matmulsilu_builder.add("output_shape", op_shape2);
          matmulsilu_builder.add("zero_point", int64_t(in_zero_point));
          // matmulsilu_builder.add("wt_name", wt_name);
          matmulsilu_builder.add("input_format", "NHWC");
          matmulsilu_builder.add("design_param", "4x4PSU");
          matmulsilu_builder.add("in_dtypes", input_types);
          matmulsilu_builder.add("out_dtypes", output_types_silu);
          matmulsilu_builder.add("input_q_params", input_q_params);
          matmulsilu_builder.add("output_q_params", output_q_params);
          auto& mm_silu_node = matmulsilu_builder.build();

          // auto [C0_mm, C1_mm, C2_mm, conv_shift_mm, shft_c2_mm] =
          //     vaip::dd::qmatmulcalc::dq_uint16A_int4W_conv_chwise_q_param_gen(
          //         in_scale, in_zero_point, wts_vec_matm, w_sc_matm,
          //         w_zp_matm, w_shape2, bias, 0.0f, 0, out_sc_mm, out_zp_mm);

          std::vector<int32_t> input_qdq_matm(16, 0);
          input_qdq_matm[8] = static_cast<int32_t>(shft_c2);
          input_qdq_matm[9] = static_cast<int32_t>(conv_shift);
          input_qdq_matm[10] = 4;
          auto& input_qdq_mm_arg =
              vaip::dd::insert_named_tensor_in_graph<int32_t>(
                  graph, node_name + "_qdq_mm", input_qdq_matm,
                  std::vector({(int64_t)input_qdq_matm.size()}));

          auto& input_c0_mm_arg =
              vaip::dd::insert_named_tensor_in_graph<int64_t>(
                  graph, node_name + "_c0_matm", C0_mm,
                  std::vector({(int64_t)C0_mm.size()}));
          auto& input_c1_mm_arg =
              vaip::dd::insert_named_tensor_in_graph<int32_t>(
                  graph, node_name + "_c1_matm", C1_mm,
                  std::vector({(int64_t)C1_mm.size()}));
          auto& input_c2_mm_arg =
              vaip::dd::insert_named_tensor_in_graph<int32_t>(
                  graph, node_name + "_c2_matm", C2_mm,
                  std::vector({(int64_t)C2_mm.size()}));

          std::string wts_initializer_name_mm =
              node_arg_get_name(*weights_node.node_arg) + "1";
          NodeArg& wts_arg_matm =
              vaip::dd::insert_named_tensor_in_graph<int8_t>(
                  graph, wts_initializer_name_mm, wts_data_matmm, w_shape2,
                  true);

          auto slice_node = binder
              ["/model/layers.31/mlp/Slice_1_output_0_QuantizeLinear_Output"];

          auto matmul_builder = NodeBuilder(*graph, *self);
          matmul_builder.set_input_node_args(
              {input_0.node_arg, &wts_arg_matm, &input_c0_mm_arg,
               &input_qdq_mm_arg, &input_c1_mm_arg, &input_c2_mm_arg});
          std::vector<std::string> input_types_mm{"uint16", "int4",  "int64",
                                                  "int32",  "int32", "int32"};
          matmul_builder.set_op_type("QConv2MatMul", "com.xilinx");
          matmul_builder.clone_data_type(*slice_node.node);
          matmul_builder.clone_attrs(*slice_node.node);
          matmul_builder.set_anchor_point1(*slice_node.node);
          matmul_builder.add("nodes", ns);
          matmul_builder.add("orig_input_shape", *in0_shape);
          matmul_builder.add("input_shape", *in0_shape);
          matmul_builder.add("weight_shape", w_shape2);
          matmul_builder.add("output_shape", op_shape2);
          matmul_builder.add("zero_point", int64_t(in_zero_point));
          // matmlu_builder.add("wt_name", wt_name);
          matmul_builder.add("input_format", "NHWC");
          matmul_builder.add("design_param", "4x4PSU");
          matmul_builder.add("in_dtypes", input_types_mm);
          matmul_builder.add("out_dtypes", output_types);
          matmul_builder.add("input_q_params", input_q_params);
          matmul_builder.add("output_q_params", output_q_params);

          auto mul_in0_sc_node =
              binder["/model/layers.31/mlp/activation_fn/Mul_output_0_scale"];
          auto mul_in0_sc = node_arg_get_const_data_as_float(
              *graph, *mul_in0_sc_node.node_arg);
          auto mul_in0_zp_node = binder
              ["/model/layers.31/mlp/activation_fn/Mul_output_0_zero_point"];
          auto mul_in0_zp =
              vaip::dd::get_zp_from_node(*graph, *mul_in0_zp_node.node_arg);
          auto [mul_in0_sc_q, mul_in0_zp_q] =
              vaip::dd::qmatmulcalc::calc_lrn_coeff(mul_in0_sc, mul_in0_zp);

          auto mul_in1_sc_node =
              binder["/model/layers.31/mlp/Slice_1_output_0_scale"];
          auto mul_in1_sc = node_arg_get_const_data_as_float(
              *graph, *mul_in1_sc_node.node_arg);
          auto mul_in1_zp_node =
              binder["/model/layers.31/mlp/Slice_1_output_0_zero_point"];
          auto mul_in1_zp =
              vaip::dd::get_zp_from_node(*graph, *mul_in1_zp_node.node_arg);

          auto [mul_in1_sc_q, mul_in1_zp_q] =
              vaip::dd::qmatmulcalc::calc_lrn_coeff(out_sc_mm, out_zp_mm);

          auto mul_op_sc_node =
              binder["/model/layers.31/mlp/Mul_2_output_0_scale"];
          auto mul_op_sc = node_arg_get_const_data_as_float(
              *graph, *mul_op_sc_node.node_arg);
          auto mul_op_zp_node =
              binder["/model/layers.31/mlp/Mul_2_output_0_zero_point"];
          auto mul_op_zp =
              vaip::dd::get_zp_from_node(*graph, *mul_op_zp_node.node_arg);
          auto [mul_final_sc, mul_final_zp] =
              vaip::dd::qmatmulcalc::calc_lrn_coeff(1 / mul_op_sc, mul_op_zp);

          std::vector<float> input_q_params1{mul_in0_sc, float(mul_in0_zp),
                                             mul_in1_sc, float(mul_in1_zp)};
          std::vector<float> output_q_params1{mul_op_sc, float(mul_op_zp)};
          std::vector<int32_t> elt_coeffs(16, 0);
          elt_coeffs[0] = mul_in0_sc_q;
          elt_coeffs[1] = mul_in0_zp_q;
          elt_coeffs[2] = mul_in1_sc_q;
          elt_coeffs[3] = mul_in1_zp_q;
          elt_coeffs[4] = mul_final_sc;
          elt_coeffs[5] = mul_final_zp;
          elt_coeffs[6] = 0;
          elt_coeffs[7] = 1;
          std::string elt_coeff_name = std::string(node_name + "_mul_qdq_");
          auto& elt_arg = vaip::dd::insert_named_tensor_in_graph<int32_t>(
              graph, elt_coeff_name, elt_coeffs,
              std::vector({(int64_t)elt_coeffs.size()}));
          std::vector<std::string> mul_in_types{"bfloat16", "uint16", "int32"};
          auto& mm_node = matmul_builder.build();
          int64_t mul_op_sz =
              std::accumulate(op_shape2.begin(), op_shape2.end(), (size_t)1,
                              std::multiplies<int64_t>());
          std::vector<int64_t> shape_mul = {mul_op_sz, 1};
          std::vector<std::string> i_shapes_mul = {
              shape_as_dd_string(shape_mul), shape_as_dd_string(shape_mul)};
          std::vector<std::string> o_shapes_mul = {
              shape_as_dd_string(shape_mul)};
          auto elemul_builder = NodeBuilder(*graph, *self);
          elemul_builder.set_input_node_args(
              {node_get_output_node_args(mm_silu_node)[0],
               node_get_output_node_args(mm_node)[0], &elt_arg});
          elemul_builder.set_op_type("QELWEMUL_qdq", "com.xilinx");
          elemul_builder.clone_data_type(*flat_mlp_node.node);
          elemul_builder.clone_attrs(*flat_mlp_node.node);
          elemul_builder.set_anchor_point1(*flat_mlp_node.node);
          elemul_builder.add("nodes", ns);
          elemul_builder.add("orig_output_shape", *flat_mlp_shape);
          // elemul_builder.add("dd_op_in_shape", i_shapes_mul);
          // elemul_builder.add("dd_op_out_shape", o_shapes_mul);
          elemul_builder.add("design_param", "4x4PSU");
          elemul_builder.add("input_q_params", input_q_params1);
          elemul_builder.add("output_q_params", output_q_params1);
          elemul_builder.add("in_dtypes", mul_in_types);
          elemul_builder.add("out_dtypes", dtypes);
          elemul_builder.build();
          return true;
        });
  }
  void process(IPass& self, Graph& graph) { create_rule(&self)->apply(&graph); }

public:
  IPass& self_;
};
} // namespace

#ifdef __GNUC__
#  pragma GCC diagnostic pop
#endif

DEFINE_VAIP_PASS(MergeFlatMLP2, vaip_pass_dd_merge_flatmlp2)
