/*
 *     The Xilinx Vitis AI Vaip in this distribution are provided under the
 * following free and permissive binary-only license, but are not provided in
 * source code form.  While the following free and permissive license is similar
 * to the BSD open source license, it is NOT the BSD open source license nor
 * other OSI-approved open source license.
 *
 *      Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights
 * reserved.
 *
 *      Redistribution and use in binary form only, without modification, is
 * permitted provided that the following conditions are met:
 *
 *      1. Redistributions must reproduce the above copyright notice, this list
 * of conditions and the following disclaimer in the documentation and/or other
 * materials provided with the distribution.
 *
 *      2. The name of Xilinx, Inc. may not be used to endorse or promote
 * products redistributed with this software without specific prior written
 * permission.
 *
 *      THIS SOFTWARE IS PROVIDED BY XILINX, INC. "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL XILINX, INC. BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 *      PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE
 */

#include "vaip/dd/coeffs.hpp"
#include "vaip/dd/dd_utils.hpp"
#include "vaip/pattern_zoo.hpp"
#include "vaip/vaip.hpp"

namespace {
using namespace vaip_core;

std::tuple<std::vector<int64_t>, int32_t, int64_t, int64_t, int64_t, int64_t>
dq_uint16A_uint8W_bias_matmul_q_param_gen(
    float a_dq_xscale, uint16_t a_dq_xzero_pt,
    const std::vector<std::vector<uint8_t>>& weights, float w_dq_xscale,
    uint16_t w_dq_xzero_pt, const std::vector<int32_t>& bias, float b_dq_xscale,
    int32_t b_dq_xzero_pt, float a_q_yscale, uint16_t a_q_yzero_pt) {

  int64_t a_dq_xzero_pt_int64 = static_cast<int64_t>(a_dq_xzero_pt);
  int64_t w_dq_xzero_pt_int64 = static_cast<int64_t>(w_dq_xzero_pt);
  int64_t a_q_yzero_pt_int64 = static_cast<int64_t>(a_q_yzero_pt);

  // assert(weights.size() > 0 && weights[0].size() > 0);  // weights shape
  // should be 2 dims

  int64_t weights_in_ch = static_cast<int64_t>(weights.size());

  int64_t matmul_shift = std::min(
      std::max(static_cast<int64_t>(std::ceil(std::log2(weights_in_ch))) - 7,
               int64_t(0)),
      int64_t(7));

  std::vector<std::vector<int64_t>> weights_int64(
      weights.size(), std::vector<int64_t>(weights[0].size()));
  for (size_t i = 0; i < weights.size(); ++i) {
    for (size_t j = 0; j < weights[i].size(); ++j) {
      weights_int64[i][j] = static_cast<int64_t>(weights[i][j]);
    }
  }

  std::vector<int64_t> bias_min_zp(bias.size());
  std::transform(bias.begin(), bias.end(), bias_min_zp.begin(),
                 [b_dq_xzero_pt](int32_t b) {
                   return static_cast<int64_t>(b) -
                          static_cast<int64_t>(b_dq_xzero_pt);
                 });

  double c2_coeff = (a_dq_xscale * w_dq_xscale) / a_q_yscale;
  double c4_coeff = b_dq_xscale / a_q_yscale;
  auto [c2_coeff_prime, shft_c2] =
      vaip::dd::qmatmulcalc::find_closest_shifted_int32(c2_coeff, 8388607);
  auto [_c4_coeff_prime, shft_c4] =
      vaip::dd::qmatmulcalc::find_closest_shifted_int32(c4_coeff, 8388607);
  int64_t c4_coeff_prime = _c4_coeff_prime;

  if (shft_c2 != shft_c4) {
    int64_t diff_shft_c2_c4 = shft_c2 - shft_c4;
    int64_t abs_diff_shft_c2_c4 = std::abs(diff_shft_c2_c4);
    if (diff_shft_c2_c4 > 0) {
      c4_coeff_prime = c4_coeff_prime << abs_diff_shft_c2_c4;
    } else if (diff_shft_c2_c4 < 0) {
      c4_coeff_prime = c4_coeff_prime >> abs_diff_shft_c2_c4;
    }
  }

  c2_coeff_prime = static_cast<int64_t>(c2_coeff_prime);

  std::vector<int64_t> c1_coeff(weights[0].size());
  for (size_t i = 0; i < weights[0].size(); ++i) {
    int64_t weights_sum = 0;
    for (size_t j = 0; j < weights.size(); ++j) {
      weights_sum += weights_int64[j][i];
    }
    c1_coeff[i] = (-a_dq_xzero_pt_int64) * c2_coeff_prime * weights_sum +
                  (a_q_yzero_pt_int64 << shft_c2) +
                  (bias_min_zp[i] * c4_coeff_prime);
  }

  int64_t num_weights_unrolled = weights_in_ch;
  int32_t c3_coeff_offset =
      static_cast<int32_t>(-a_dq_xzero_pt_int64 * num_weights_unrolled);
  int64_t c3_coeff_scale = -c2_coeff_prime * w_dq_xzero_pt_int64;
  int64_t c3_coeff_scale_shift = 0;

  if (std::abs(c3_coeff_scale) > 2147483647) { // Max int32 number
    c3_coeff_scale_shift = static_cast<int64_t>(
        std::ceil(std::log2(std::abs(c3_coeff_scale))) - 31);
  } else {
    c3_coeff_scale_shift = 0;
  }

  c3_coeff_scale = static_cast<int32_t>(c3_coeff_scale >> c3_coeff_scale_shift);
  int64_t temp = c3_coeff_scale * c3_coeff_offset << c3_coeff_scale_shift;
  std::transform(c1_coeff.begin(), c1_coeff.end(), c1_coeff.begin(),
                 [temp](int64_t c) { return c + temp; });

  // std::tuple<
  //     std::vector<int64_t>,
  //     int32_t,
  //     int64_t,
  //     int64_t,
  //     int64_t,
  //     int64_t
  // >

  // std::tuple<std::vector<int64_t>, int32_t, int64_t, int64_t, int64_t,
  // int64_t>

  int32_t C2 = c2_coeff_prime << matmul_shift;
  // int32_t C1 = c3_coeff_scale;
  // int64_t C0 = static_cast<int64_t>(c3_coeff_scale*c3_coeff_offset) <<
  // c3_coeff_scale_shift + c1_coeff;

  return std::make_tuple(c1_coeff, (int32_t)c3_coeff_scale, C2,
                         (int64_t)c3_coeff_scale_shift, (int64_t)shft_c2,
                         (int64_t)matmul_shift);
}

static void add_common_attr(NodeBuilder& building_node) {
  building_node.add("Node_dtype", "<class 'numpy.uint16'>");
  building_node.add("input_format", "NHWC");
}

static void add_node_attr_qgemmv(NodeBuilder& building_node,
                                 onnxruntime::Graph* graph, binder_t* binder) {
  std::vector<std::string> nodes;
  for (auto& ni : *binder) {
    if ((*node_arg_is_constant)(*graph, *ni.second.node_arg)) {
      continue;
    }
    nodes.push_back(node_arg_get_name(*ni.second.node_arg));
  }
  building_node.add("nodes", nodes);
}

static void modify_inputs(NodeBuilder& building_node, const NodeInput& gemm,
                          onnxruntime::Graph* graph, float op_scale,
                          uint16_t op_zp) {
  auto new_inputs = std::vector<const NodeArg*>();
  auto inputs = node_get_inputs(*gemm.node);
  auto dequant_0 = inputs[0].node;
  auto input_0 = node_get_input_node_args(*dequant_0)[0];

  new_inputs.push_back(input_0);
  auto dequant_1 = inputs[1].node;
  auto input_1 = node_get_input_node_args(*dequant_1)[0];
  new_inputs.push_back(input_1);
  auto dequant_2 = inputs[2].node;

  auto a_dq_scale = node_arg_get_const_data_as_float(
      *graph, *node_get_input_node_args(*dequant_0)[1]);
  auto a_dq_zp = node_arg_get_const_data_as_u16(
      *graph, *node_get_input_node_args(*dequant_0)[2]);

  auto weight = node_get_input_node_args(*dequant_1)[0];
  auto weight_shape = node_arg_get_shape_i64(*weight);
  auto weight_tensor_untransposed = vaip::dd::fold2D<uint8_t>(
      node_arg_get_const_data_as_u8s(*graph, *weight), *weight_shape.get());
  decltype(weight_tensor_untransposed) weight_tensor;
  int transpose_size_0 = static_cast<int>(weight_tensor_untransposed[0].size());
  int transpose_size_1 = static_cast<int>(weight_tensor_untransposed.size());
  weight_tensor.resize(transpose_size_0);
  for (int i = 0; i < transpose_size_1; ++i) {
    for (int j = 0; j < transpose_size_0; ++j) {
      weight_tensor[j].push_back(weight_tensor_untransposed[i][j]);
    }
  }
  auto w_dq_scale = node_arg_get_const_data_as_float(
      *graph, *node_get_input_node_args(*dequant_1)[1]);
  auto w_dq_zp = node_arg_get_const_data_as_u8(
      *graph, *node_get_input_node_args(*dequant_1)[2]);

  auto bias = node_get_input_node_args(*dequant_2)[0];
  auto bias_shape = node_arg_get_shape_i64(*bias);
  auto bias_tensor = vaip::dd::fold1D<int32_t>(
      node_arg_get_const_data_as_i32s(*graph, *bias), *(bias_shape.get()));

  auto b_dq_scale = node_arg_get_const_data_as_floats(
      *graph, *node_get_input_node_args(*dequant_2)[1])[0];
  auto b_dq_zp = node_arg_get_const_data_as_i32(
      *graph, *node_get_input_node_args(*dequant_2)[2]);

  // auto quant_node_input = node_get_inputs(*output.node)[0];
  // auto quant = node_get_input_node_args(*quant_node_input.node);

  auto a_q_scale = op_scale;
  auto a_q_zp = op_zp;
  std::vector<int64_t> c0_tensor_value;
  int32_t c1 = 0;
  int64_t c2 = 0;
  int64_t shift_qb = 0;
  int64_t shift_out = 0;
  int64_t matmul_shift = 0;

  std::tie(c0_tensor_value, c1, c2, shift_qb, shift_out, matmul_shift) =
      dq_uint16A_uint8W_bias_matmul_q_param_gen(
          a_dq_scale, a_dq_zp, weight_tensor, w_dq_scale, w_dq_zp, bias_tensor,
          b_dq_scale, b_dq_zp, a_q_scale, a_q_zp);

  auto c0_name = node_arg_get_name(*input_1) + "_c0_";
  std::vector<int64_t> c0_shape = {
      (static_cast<int64_t>(c0_tensor_value.size()))};
  auto c0_tensor = tensor_proto_new_i64(c0_name, c0_shape, c0_tensor_value);
  VAIP_ORT_API(graph_add_initialized_tensor)(*graph, *c0_tensor);
  const NodeArg* c0 = &VAIP_ORT_API(node_arg_new)(
      *graph, c0_name, &c0_shape, ONNX_NAMESPACE::TensorProto_DataType_INT64);
  new_inputs.push_back(c0);

  auto qdq_name = node_arg_get_name(*input_1) + "_qdq";
  auto qdq_value = std::vector<int32_t>(16, 0);

  qdq_value[2] = c1;
  qdq_value[3] = static_cast<int32_t>(c2);
  qdq_value[5] = 64;
  qdq_value[6] = 64;
  qdq_value[7] = static_cast<int32_t>(shift_qb);
  qdq_value[8] = static_cast<int32_t>(shift_out);
  qdq_value[9] = static_cast<int32_t>(matmul_shift);
  qdq_value[10] = 1;

  auto qdq_shape = std::vector<int64_t>({16});
  auto qdq_tensor = tensor_proto_new_i32(qdq_name, qdq_shape, qdq_value);
  VAIP_ORT_API(graph_add_initialized_tensor)(*graph, *qdq_tensor);
  const NodeArg* qdq = &VAIP_ORT_API(node_arg_new)(
      *graph, qdq_name, &qdq_shape, ONNX_NAMESPACE::TensorProto_DataType_INT32);
  new_inputs.push_back(qdq);

  building_node.set_input_node_args(new_inputs);
}
struct MergeQGemmv {
  MergeQGemmv(IPass& self) : self_{self} {}
  std::unique_ptr<Rule> create_rule(IPass* self) {
    auto pattern_ = vaip::pattern_zoo::get_pattern("m_qgemmv");
    CHECK(pattern_ != nullptr) << "Pattern returned is null";
    return Rule::create_rule(
        pattern_, [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
          auto output = binder["com_microsoft_QuantizeLinear_1"];
          auto output_scale = node_arg_get_const_data_as_float(
              *graph, *(binder["constant_3"]).node_arg);
          auto output_zp = node_arg_get_const_data_as_u16(
              *graph, *(binder["constant_4"]).node_arg);
          auto name = node_arg_get_name(*output.node_arg);
          auto new_node = NodeBuilder(*graph, self_);
          auto act_node = binder["input_0"];
          auto act_in_shape = node_arg_get_shape_i64(*act_node.node_arg);
          if (act_in_shape.get()->size() == 2) {
            act_in_shape.get()->push_back(1);
            act_in_shape.get()->push_back(1);
          }
          auto out_node = binder["com_microsoft_QuantizeLinear_1"];
          auto out_shape = node_arg_get_shape_i64(*out_node.node_arg);
          if (out_shape.get()->size() == 2) {
            out_shape.get()->push_back(1);
            out_shape.get()->push_back(1);
          }
          add_common_attr(new_node);
          add_node_attr_qgemmv(new_node, graph, &binder);
          modify_inputs(new_node, binder["Gemm_0"], graph, output_scale,
                        output_zp);
          new_node.set_op_type("QConv2MatMul", "com.xilinx");
          new_node.set_anchor_point1(*output.node);
          std::vector<std::string> in_dtypes = {"uint16", "uint8", "int64",
                                                "int32"};
          std::vector<std::string> out_dtypes = {"uint16"};
          new_node.add("in_dtypes", in_dtypes);
          new_node.add("out_dtypes", out_dtypes);
          new_node.add("input_shape", *act_in_shape);
          // new_node.add("weight_shape", *kernel_shape);
          new_node.add("output_shape", *out_shape);
          new_node.add("from_gemmv", "true");
          new_node.build();
          return true;
        });
  }
  void process(IPass& self, Graph& graph) { create_rule(&self)->apply(&graph); }

public:
  IPass& self_;
};
} // namespace

DEFINE_VAIP_PASS(MergeQGemmv, vaip_pass_dd_merge_qgemmv)