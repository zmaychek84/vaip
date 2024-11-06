/*
 *     The Xilinx Vitis AI Vaip in this distribution are provided under the
 * following free and permissive binary-only license, but are not provided in
 * source code form.  While the following free and permissive license is similar
 * to the BSD open source license, it is NOT the BSD open source license nor
 * other OSI-approved open source license.
 *
 *      Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc.
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
#include <glog/logging.h>

#include "vaip/dd/coeffs.hpp"
#include "vaip/dd/dd_utils.hpp"
#include "vaip/vaip.hpp"
#include "vitis/ai/env_config.hpp"

#include "vaip/dd/coeffs.hpp"
#include "vaip/dd/dd_utils.hpp"
#include "vaip/pattern_zoo.hpp"

DEF_ENV_PARAM(DEBUG_DD_MERGE_mzdk5MHA, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_DD_MERGE_mzdk5MHA) >= n)

namespace {
using namespace vaip_core;

// if next set of nodes contain unfused concat pattern, i.e if concat op is a
// child node of final_quant_node vsm_sc and vsm_zp (output scale and zeropoint
// of calling op) are updated with  scale and zeropoint of QuantizeLinear after
// concat (concat op's output scale and zeropoint)
static std::pair<float, uint16_t>
get_concat_qparams(onnxruntime::Graph* graph, const NodeInput& final_quant_node,
                   float vsm_sc, uint16_t vsm_zp) {
  // Check if these's only one consumer and get it's name and nodearg
  if (graph_get_consumer_nodes(*graph,
                               node_arg_get_name(*final_quant_node.node_arg))
          .size() == 1) {
    std::vector<const Node*> final_quant_node_nextnodes =
        graph_get_consumer_nodes(*graph,
                                 node_arg_get_name(*final_quant_node.node_arg));
    std::string dq_before_concat_node_name =
        node_get_first_output_name(*final_quant_node_nextnodes[0]);
    auto dq_before_concat_node_arg =
        VAIP_ORT_API(graph_get_node_arg)(*graph, dq_before_concat_node_name);

    // Check if these's only one consumer and get it's name and nodearg
    if (graph_get_consumer_nodes(*graph,
                                 node_arg_get_name(*dq_before_concat_node_arg))
            .size() == 1) {
      std::vector<const Node*> dq_before_concat_next_nodes =
          graph_get_consumer_nodes(
              *graph, node_arg_get_name(*dq_before_concat_node_arg));
      std::string concat_node_name =
          node_get_first_output_name(*dq_before_concat_next_nodes[0]);
      auto concat_node_arg =
          VAIP_ORT_API(graph_get_node_arg)(*graph, concat_node_name);
      // Get the op_type of 2nd level consumer and check if it is concat
      auto concat_node_op_type =
          VAIP_ORT_API(node_op_type)(*dq_before_concat_next_nodes[0]);

      if (concat_node_op_type == "Concat") {
        // if 2nd level consumer is Concat op, get it's child QuantizeLinear
        // node's scale and zero point and update vsm_sc and vsm_zp
        if (graph_get_consumer_nodes(*graph,
                                     node_arg_get_name(*concat_node_arg))
                .size() == 1) {
          std::vector<const Node*> concat_node_nextnodes =
              graph_get_consumer_nodes(*graph,
                                       node_arg_get_name(*concat_node_arg));
          std::string Q_node_after_concat_name =
              node_get_first_output_name(*concat_node_nextnodes[0]);
          auto Q_node_input_node_args =
              node_get_input_node_args(*concat_node_nextnodes[0]);
          vsm_sc = node_arg_get_const_data_as_float(*graph,
                                                    *Q_node_input_node_args[1]);
          vsm_zp =
              vaip::dd::get_zp_from_node(*graph, *Q_node_input_node_args[2]);
        }
      }
    }
  }
  return std::make_pair(vsm_sc, vsm_zp);
}

std::tuple<std::vector<NodeArg*>, std::vector<float>, std::vector<float>,
           std::vector<std::string>, std::vector<int32_t>>
get_QMMDy_node_args(onnxruntime::Graph* graph_, binder_t& binder_,
                    bool is_qkt = true) {
  std::vector<int32_t> qdq_params(16, 0);
  std::string initializer_name = "_qdq_";
  std::vector<float> in_q_params;
  std::vector<float> out_q_params;
  std::vector<std::string> nodes;

  if (is_qkt) {
    // query
    auto input_q_node =
        binder_["/down_blocks.0/attentions.0/transformer_blocks.0/attn2/"
                "Transpose_2_output_0_QuantizeLinear_Output"];
    auto q_sc_node = binder_["/down_blocks.0/attentions.0/transformer_blocks.0/"
                             "attn2/to_q_convs.0/Conv_output_0_scale"];
    auto q_zp_node = binder_["/down_blocks.0/attentions.0/transformer_blocks.0/"
                             "attn2/to_q_convs.0/Conv_output_0_zero_point"];
    auto q_shape_back =
        (*node_arg_get_shape_i64(*input_q_node.node_arg).get()).back();
    float q_sc = node_arg_get_const_data_as_float(*graph_, *q_sc_node.node_arg);
    uint16_t q_zp = vaip::dd::get_zp_from_node(*graph_, *q_zp_node.node_arg);

    // key
    auto k_sc_node = binder_["/down_blocks.0/attentions.0/transformer_blocks.0/"
                             "attn2/to_k_convs.0/Conv_output_0_scale"];
    auto k_zp_node = binder_["/down_blocks.0/attentions.0/transformer_blocks.0/"
                             "attn2/to_k_convs.0/Conv_output_0_zero_point"];
    float k_sc = node_arg_get_const_data_as_float(*graph_, *k_sc_node.node_arg);
    uint16_t k_zp = vaip::dd::get_zp_from_node(*graph_, *k_zp_node.node_arg);

    // qkt
    auto qkt_sc_node =
        binder_["/down_blocks.0/attentions.0/transformer_blocks.0/attn2/"
                "matmul_1.0/MatMul_output_0_scale"];
    auto qkt_zp_node =
        binder_["/down_blocks.0/attentions.0/transformer_blocks.0/attn2/"
                "matmul_1.0/MatMul_output_0_zero_point"];
    float qkt_sc =
        node_arg_get_const_data_as_float(*graph_, *qkt_sc_node.node_arg);
    uint16_t qkt_zp =
        vaip::dd::get_zp_from_node(*graph_, *qkt_zp_node.node_arg);

    auto coeff_qkt = vaip::dd::qmatmulcalc::qdq_act_matmul_uint16_uint16_cstm(
        q_sc, q_zp, q_shape_back, k_sc, k_zp, qkt_sc, qkt_zp);

    reinterpret_cast<int64_t*>(qdq_params.data())[0] = std::get<0>(coeff_qkt);
    qdq_params[2] = std::get<1>(coeff_qkt);
    qdq_params[3] = static_cast<int32_t>(std::get<2>(coeff_qkt));
    qdq_params[4] = std::get<3>(coeff_qkt);
    qdq_params[5] = 16;
    qdq_params[6] = 64;
    qdq_params[7] = std::get<4>(coeff_qkt);
    qdq_params[8] = std::get<5>(coeff_qkt);
    qdq_params[9] = std::get<6>(coeff_qkt);

    initializer_name = node_arg_get_name(*q_zp_node.node_arg) + "_qdq_qkt_";

    in_q_params = {q_sc, float(q_zp), k_sc, float(k_zp)};
    out_q_params = {qkt_sc, float(qkt_zp)};

    in_q_params = {q_sc, float(q_zp), k_sc, float(k_zp)};
    out_q_params = {qkt_sc, float(qkt_zp)};

    auto dq1 = binder_["/down_blocks.0/attentions.0/transformer_blocks.0/attn2/"
                       "Transpose_2_output_0_DequantizeLinear_Output"];
    auto dq2 = binder_["/down_blocks.0/attentions.0/transformer_blocks.0/attn2/"
                       "Unsqueeze_output_0_DequantizeLinear_Output"];
    auto mm1 = binder_["/down_blocks.0/attentions.0/transformer_blocks.0/attn2/"
                       "matmul_1.0/MatMul_output_0"];
    auto q1 = binder_["/down_blocks.0/attentions.0/transformer_blocks.0/attn2/"
                      "matmul_1.0/MatMul_output_0_QuantizeLinear_Output"];

    nodes = {node_arg_get_name(*dq1.node_arg), node_arg_get_name(*dq2.node_arg),
             node_arg_get_name(*mm1.node_arg), node_arg_get_name(*q1.node_arg)};

  } else {

    auto v_sc_node = binder_["/down_blocks.0/attentions.0/transformer_blocks.0/"
                             "attn2/to_v_convs.0/Conv_output_0_scale"];
    auto v_zp_node = binder_["/down_blocks.0/attentions.0/transformer_blocks.0/"
                             "attn2/to_v_convs.0/Conv_output_0_zero_point"];
    float v_sc = node_arg_get_const_data_as_float(*graph_, *v_sc_node.node_arg);
    uint16_t v_zp = vaip::dd::get_zp_from_node(*graph_, *v_zp_node.node_arg);

    // sm
    auto sm_sc_node =
        binder_["/down_blocks.0/attentions.0/transformer_blocks.0/attn2/"
                "softmax_1.0/Softmax_output_0_scale"];
    auto sm_zp_node =
        binder_["/down_blocks.0/attentions.0/transformer_blocks.0/attn2/"
                "softmax_1.0/Softmax_output_0_zero_point"];
    float sm_sc =
        node_arg_get_const_data_as_float(*graph_, *sm_sc_node.node_arg);
    uint16_t sm_zp = vaip::dd::get_zp_from_node(*graph_, *sm_zp_node.node_arg);

    // vsm
    auto vsm_sc_node =
        binder_["/down_blocks.0/attentions.0/transformer_blocks.0/attn2/"
                "matmul_2.0/MatMul_output_0_scale"];
    auto vsm_zp_node =
        binder_["/down_blocks.0/attentions.0/transformer_blocks.0/attn2/"
                "matmul_2.0/MatMul_output_0_zero_point"];
    auto input_k_node =
        binder_["/down_blocks.0/attentions.0/transformer_blocks.0/attn2/"
                "Unsqueeze_output_0_QuantizeLinear_Output"];
    auto k_shape_back =
        (*node_arg_get_shape_i64(*input_k_node.node_arg).get()).back();
    float vsm_sc =
        node_arg_get_const_data_as_float(*graph_, *vsm_sc_node.node_arg);
    uint16_t vsm_zp =
        vaip::dd::get_zp_from_node(*graph_, *vsm_zp_node.node_arg);

    // insert the concat qdq update here,
    // If concat is  next op, the output scale and zp are updated here else same
    // params are retained
    auto final_quant_node = binder_
        ["/down_blocks.0/attentions.0/transformer_blocks.0/attn2/matmul_2.0/"
         "MatMul_output_0_QuantizeLinear_Output"]; // TODO: Remove this hard
                                                   // coding, get the final node
                                                   // in the pattern directly
    auto concat_params =
        get_concat_qparams(graph_, final_quant_node, vsm_sc, vsm_zp);
    vsm_sc = concat_params.first;
    vsm_zp = concat_params.second;
    auto coeff_smv = vaip::dd::qmatmulcalc::qdq_act_matmul_uint16_uint16_cstm(
        sm_sc, sm_zp, k_shape_back, v_sc, v_zp, vsm_sc, vsm_zp);

    reinterpret_cast<int64_t*>(qdq_params.data())[0] = std::get<0>(coeff_smv);
    qdq_params[2] = std::get<1>(coeff_smv);
    qdq_params[3] = static_cast<int32_t>(std::get<2>(coeff_smv));
    qdq_params[4] = std::get<3>(coeff_smv);
    qdq_params[5] = 16;
    qdq_params[6] = 128;
    qdq_params[7] = std::get<4>(coeff_smv);
    qdq_params[8] = std::get<5>(coeff_smv);
    qdq_params[9] = std::get<6>(coeff_smv);

    initializer_name = node_arg_get_name(*v_sc_node.node_arg) + "_qdq_smv_";

    in_q_params = {v_sc, float(v_zp), sm_sc, float(sm_zp)};
    out_q_params = {vsm_sc, float(vsm_zp)};

    auto dq1 = binder_["/down_blocks.0/attentions.0/transformer_blocks.0/attn2/"
                       "softmax_1.0/Softmax_output_0_DequantizeLinear_Output"];
    auto dq2 = binder_["/down_blocks.0/attentions.0/transformer_blocks.0/attn2/"
                       "Unsqueeze_1_output_0_DequantizeLinear_Output"];
    auto mm1 = binder_["/down_blocks.0/attentions.0/transformer_blocks.0/attn2/"
                       "matmul_2.0/MatMul_output_0"];
    auto q1 = binder_["/down_blocks.0/attentions.0/transformer_blocks.0/attn2/"
                      "matmul_2.0/MatMul_output_0_QuantizeLinear_Output"];

    nodes = {node_arg_get_name(*dq1.node_arg), node_arg_get_name(*dq2.node_arg),
             node_arg_get_name(*mm1.node_arg), node_arg_get_name(*q1.node_arg)};
  }

  auto& qdq_arg = vaip::dd::insert_named_tensor_in_graph<int32_t>(
      graph_, initializer_name, qdq_params, {1, 16});

  std::vector<NodeArg*> ret;
  ret.push_back(&qdq_arg);
  return std::make_tuple(ret, in_q_params, out_q_params, nodes, qdq_params);
}

std::tuple<std::vector<NodeArg*>, std::vector<float>, std::vector<float>>
get_mzdk5MHA_node_args(onnxruntime::Graph* graph_, binder_t& binder_) {

  // query
  auto input_q_node =
      binder_["/down_blocks.0/attentions.0/transformer_blocks.0/attn2/"
              "Transpose_2_output_0_QuantizeLinear_Output"];
  auto q_sc_node = binder_["/down_blocks.0/attentions.0/transformer_blocks.0/"
                           "attn2/to_q_convs.0/Conv_output_0_scale"];
  auto q_zp_node = binder_["/down_blocks.0/attentions.0/transformer_blocks.0/"
                           "attn2/to_q_convs.0/Conv_output_0_zero_point"];
  auto q_shape_back =
      (*node_arg_get_shape_i64(*input_q_node.node_arg).get()).back();
  float q_sc = node_arg_get_const_data_as_float(*graph_, *q_sc_node.node_arg);
  uint16_t q_zp = vaip::dd::get_zp_from_node(*graph_, *q_zp_node.node_arg);

  // key
  auto input_k_node =
      binder_["/down_blocks.0/attentions.0/transformer_blocks.0/attn2/"
              "Unsqueeze_output_0_QuantizeLinear_Output"];
  auto k_sc_node = binder_["/down_blocks.0/attentions.0/transformer_blocks.0/"
                           "attn2/to_k_convs.0/Conv_output_0_scale"];
  auto k_zp_node = binder_["/down_blocks.0/attentions.0/transformer_blocks.0/"
                           "attn2/to_k_convs.0/Conv_output_0_zero_point"];
  auto k_shape_back =
      (*node_arg_get_shape_i64(*input_k_node.node_arg).get()).back();
  float k_sc = node_arg_get_const_data_as_float(*graph_, *k_sc_node.node_arg);
  uint16_t k_zp = vaip::dd::get_zp_from_node(*graph_, *k_zp_node.node_arg);

  // value
  auto v_sc_node = binder_["/down_blocks.0/attentions.0/transformer_blocks.0/"
                           "attn2/to_v_convs.0/Conv_output_0_scale"];
  auto v_zp_node = binder_["/down_blocks.0/attentions.0/transformer_blocks.0/"
                           "attn2/to_v_convs.0/Conv_output_0_zero_point"];
  float v_sc = node_arg_get_const_data_as_float(*graph_, *v_sc_node.node_arg);
  uint16_t v_zp = vaip::dd::get_zp_from_node(*graph_, *v_zp_node.node_arg);

  // qkt
  auto qkt_sc_node = binder_["/down_blocks.0/attentions.0/transformer_blocks.0/"
                             "attn2/matmul_1.0/MatMul_output_0_scale"];
  auto qkt_zp_node = binder_["/down_blocks.0/attentions.0/transformer_blocks.0/"
                             "attn2/matmul_1.0/MatMul_output_0_zero_point"];
  float qkt_sc =
      node_arg_get_const_data_as_float(*graph_, *qkt_sc_node.node_arg);
  uint16_t qkt_zp = vaip::dd::get_zp_from_node(*graph_, *qkt_zp_node.node_arg);

  // sm
  auto sm_sc_node = binder_["/down_blocks.0/attentions.0/transformer_blocks.0/"
                            "attn2/softmax_1.0/Softmax_output_0_scale"];
  auto sm_zp_node = binder_["/down_blocks.0/attentions.0/transformer_blocks.0/"
                            "attn2/softmax_1.0/Softmax_output_0_zero_point"];
  float sm_sc = node_arg_get_const_data_as_float(*graph_, *sm_sc_node.node_arg);
  uint16_t sm_zp = vaip::dd::get_zp_from_node(*graph_, *sm_zp_node.node_arg);

  // vsm
  auto vsm_sc_node = binder_["/down_blocks.0/attentions.0/transformer_blocks.0/"
                             "attn2/matmul_2.0/MatMul_output_0_scale"];
  auto vsm_zp_node = binder_["/down_blocks.0/attentions.0/transformer_blocks.0/"
                             "attn2/matmul_2.0/MatMul_output_0_zero_point"];
  float vsm_sc =
      node_arg_get_const_data_as_float(*graph_, *vsm_sc_node.node_arg);
  uint16_t vsm_zp = vaip::dd::get_zp_from_node(*graph_, *vsm_zp_node.node_arg);
  auto final_quant_node =
      binder_["/down_blocks.0/attentions.0/transformer_blocks.0/attn2/"
              "matmul_2.0/MatMul_output_0_QuantizeLinear_Output"];
  auto concat_params =
      get_concat_qparams(graph_, final_quant_node, vsm_sc, vsm_zp);
  vsm_sc = concat_params.first;
  vsm_zp = concat_params.second;
  // mul
  auto mul_w_node = binder_["/down_blocks.0/attentions.0/transformer_blocks.0/"
                            "attn1/Constant_10_output_0_quantized"];
  auto mul_sc_node = binder_["/down_blocks.0/attentions.0/transformer_blocks.0/"
                             "attn1/Constant_10_output_0_scale"];
  auto mul_zp_node = binder_["/down_blocks.0/attentions.0/transformer_blocks.0/"
                             "attn1/Constant_10_output_0_zero_point"];
  float mul_sc =
      node_arg_get_const_data_as_float(*graph_, *mul_sc_node.node_arg);
  uint16_t mul_zp = vaip::dd::get_zp_from_node(*graph_, *mul_zp_node.node_arg);
  float mul_w =
      static_cast<float>(
          vaip::dd::get_zp_from_node(*graph_, *mul_w_node.node_arg) - mul_zp) *
      mul_sc;

  auto coeff_qkt = vaip::dd::qmatmulcalc::qdq_act_matmul_uint16_uint16_cstm(
      q_sc, q_zp, q_shape_back, k_sc, k_zp, qkt_sc, qkt_zp);

  auto coeff_smv = vaip::dd::qmatmulcalc::qdq_act_matmul_uint16_uint16_cstm(
      sm_sc, sm_zp, k_shape_back, v_sc, v_zp, vsm_sc, vsm_zp);

  std::vector<int32_t> qdq_params =
      vaip::dd::qmatmulcalc::mha_channel_qdq_params_fill( // in32_t * 96
          coeff_qkt, coeff_smv,
          std::make_tuple(vaip::dd::qmatmulcalc::float_to_bfloat16(
                              qkt_sc * mul_w * 1.442695041f),
                          (int)qkt_zp),
          std::make_tuple(
              vaip::dd::qmatmulcalc::float_to_bfloat16(1.0f / sm_sc),
              (int)sm_zp),
          std::make_tuple(0, 0), std::make_tuple(0, 0), 1, 0);

  std::string initializer_name =
      node_arg_get_name(*q_sc_node.node_arg) + "_qdq_";
  const std::vector<int64_t> shape = {6, 16};

  auto& qdq_arg = vaip::dd::insert_named_tensor_in_graph<int32_t>(
      graph_, initializer_name, qdq_params, shape);

  std::vector<NodeArg*> ret;
  ret.push_back(&qdq_arg);

  std::vector<float> in_q_params = {q_sc,        float(q_zp), k_sc,
                                    float(k_zp), v_sc,        float(v_zp)};

  std::vector<float> out_q_params = {vsm_sc, float(vsm_zp)};

  return std::make_tuple(ret, in_q_params, out_q_params);
}
std::tuple<std::vector<NodeArg*>, std::vector<std::string>>
get_QMulSoftmax_node_args(onnxruntime::Graph* graph, binder_t& binder,
                          std::vector<int32_t> qdq_qkt) {

  auto mul_val = binder["/down_blocks.0/attentions.0/transformer_blocks.0/"
                        "attn1/Constant_10_output_0_quantized"];
  auto sc0_node = binder["/down_blocks.0/attentions.0/transformer_blocks.0/"
                         "attn1/Constant_10_output_0_scale"];
  auto zp0_node = binder["/down_blocks.0/attentions.0/transformer_blocks.0/"
                         "attn1/Constant_10_output_0_zero_point"];

  auto sc1_node = binder["/down_blocks.0/attentions.0/transformer_blocks.0/"
                         "attn2/matmul_1.0/MatMul_output_0_scale"];
  auto zp1_node = binder["/down_blocks.0/attentions.0/transformer_blocks.0/"
                         "attn2/matmul_1.0/MatMul_output_0_zero_point"];

  auto outsc1_node = binder["/down_blocks.0/attentions.0/transformer_blocks.0/"
                            "attn2/softmax_1.0/Softmax_output_0_scale"];
  auto outzp1_node = binder["/down_blocks.0/attentions.0/transformer_blocks.0/"
                            "attn2/softmax_1.0/Softmax_output_0_zero_point"];

  float mul_value =
      (float)node_arg_get_const_data_as_u16(*graph, *mul_val.node_arg);
  float mul_scale =
      node_arg_get_const_data_as_float(*graph, *sc0_node.node_arg);
  float mul_zp = (float)vaip::dd::get_zp_from_node(*graph, *zp0_node.node_arg);
  float sm_in_scale =
      node_arg_get_const_data_as_float(*graph, *sc1_node.node_arg);
  float sm_in_zp =
      (float)vaip::dd::get_zp_from_node(*graph, *zp1_node.node_arg);
  float sm_out_scale =
      node_arg_get_const_data_as_float(*graph, *outsc1_node.node_arg);
  float sm_out_zp =
      (float)vaip::dd::get_zp_from_node(*graph, *outzp1_node.node_arg);
  float multiplier = (mul_value - mul_zp) * mul_scale;

  auto qdq_sm_in = vaip::dd::qmatmulcalc::calc_lrn_coeff(
      sm_in_scale * multiplier * 1.442695041f, static_cast<uint16_t>(sm_in_zp));
  auto qdq_sm_out = vaip::dd::qmatmulcalc::calc_lrn_coeff(
      1 / sm_out_scale, static_cast<uint16_t>(sm_out_zp));

  std::vector<int32_t> qdq_params(96, 0);
  // Copy qkt qdq params
  memcpy(qdq_params.data() + 32, qdq_qkt.data(), 16 * sizeof(int32_t));

  qdq_params[64] = std::get<1>(qdq_sm_in);
  qdq_params[65] = std::get<0>(qdq_sm_in);
  qdq_params[80] = std::get<1>(qdq_sm_out);
  qdq_params[81] = std::get<0>(qdq_sm_out);

  auto node_arg_qdq_params_name =
      std::string(node_arg_get_name(*outsc1_node.node_arg)) + "_qdq_";
  auto node_arg_qdq_params_shape = std::vector<int64_t>{6, 16};
  auto& node_arg_qdq_params = vaip::dd::insert_named_tensor_in_graph<int32_t>(
      graph, node_arg_qdq_params_name, qdq_params, node_arg_qdq_params_shape);

  std::vector<NodeArg*> ret;
  ret.push_back(&node_arg_qdq_params);

  auto dq1 = binder["/down_blocks.0/attentions.0/transformer_blocks.0/attn2/"
                    "matmul_1.0/MatMul_output_0_DequantizeLinear_Output"];
  auto dq2 = binder["/down_blocks.0/attentions.0/transformer_blocks.0/attn1/"
                    "Constant_10_output_0_DequantizeLinear_Output"];
  auto m1 = binder
      ["/down_blocks.0/attentions.0/transformer_blocks.0/attn2/Mul_output_0"];
  auto q1 = binder["/down_blocks.0/attentions.0/transformer_blocks.0/attn2/"
                   "Mul_output_0_QuantizeLinear_Output"];
  auto dq3 = binder["/down_blocks.0/attentions.0/transformer_blocks.0/attn2/"
                    "Mul_output_0_DequantizeLinear_Output"];
  auto sfm1 = binder["/down_blocks.0/attentions.0/transformer_blocks.0/attn2/"
                     "softmax_1.0/Softmax_output_0"];
  auto q2 = binder["/down_blocks.0/attentions.0/transformer_blocks.0/attn2/"
                   "softmax_1.0/Softmax_output_0_QuantizeLinear_Output"];

  std::vector<std::string> nodes = {
      node_arg_get_name(*dq1.node_arg), node_arg_get_name(*dq2.node_arg),
      node_arg_get_name(*m1.node_arg),  node_arg_get_name(*q1.node_arg),
      node_arg_get_name(*dq3.node_arg), node_arg_get_name(*sfm1.node_arg),
      node_arg_get_name(*q2.node_arg)};
  return std::make_tuple(ret, nodes);
}

struct Dd_merge_mzdk5mha {
  Dd_merge_mzdk5mha(IPass& self) : self_{self} {}
  std::unique_ptr<Rule> create_rule(IPass* self) {
    auto com_microsoft_QuantizeLinear_3 =
        vaip::pattern_zoo::get_pattern("m_mzdk5mha");
    CHECK(com_microsoft_QuantizeLinear_3 != nullptr)
        << "Pattern returned is null";
    return Rule::create_rule(
        com_microsoft_QuantizeLinear_3,
        [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
          MY_LOG(1)
              << "mzdk5 out node : "
              << node_arg_get_name(
                     *binder["/down_blocks.0/attentions.0/transformer_blocks.0/"
                             "attn2/"
                             "matmul_2.0/MatMul_output_0_QuantizeLinear_Output"]
                          .node_arg);

          // Input nodes
          auto input_q_node =
              binder["/down_blocks.0/attentions.0/transformer_blocks.0/attn2/"
                     "Transpose_2_output_0_QuantizeLinear_Output"];
          auto input_k_node =
              binder["/down_blocks.0/attentions.0/transformer_blocks.0/attn2/"
                     "Unsqueeze_output_0_QuantizeLinear_Output"];
          auto input_v_node =
              binder["/down_blocks.0/attentions.0/transformer_blocks.0/attn2/"
                     "Unsqueeze_1_output_0_QuantizeLinear_Output"];

          // Output nodes
          auto output_node =
              binder["/down_blocks.0/attentions.0/transformer_blocks.0/attn2/"
                     "matmul_2.0/MatMul_output_0_QuantizeLinear_Output"];

          // Anchor Points
          // auto mm1q_node = binder[com_microsoft_QuantizeLinear_0->get_id()];
          auto smq_node =
              binder["/down_blocks.0/attentions.0/transformer_blocks.0/attn2/"
                     "softmax_1.0/Softmax_output_0_QuantizeLinear_Output"];
          auto mm2q_node =
              binder["/down_blocks.0/attentions.0/transformer_blocks.0/attn2/"
                     "matmul_2.0/MatMul_output_0_QuantizeLinear_Output"];

          auto q_shape = *node_arg_get_shape_i64(*input_q_node.node_arg).get();
          auto kt_shape = *node_arg_get_shape_i64(*input_k_node.node_arg).get();

          CHECK(q_shape.size() == 4)
              << "q_shape not compatible. dimensions of query : "
              << q_shape.size();

          int64_t M = q_shape[0] * q_shape[1] * q_shape[2];
          int64_t K = q_shape.back();
          int64_t N = kt_shape.back();

          std::vector<int64_t> qkt_mm_shape = {M, K, N};
          bool offload_to_mha = true;
          std::vector<std::vector<int64_t>> unsupported_MKN_shapes = {
              {4096, 64, 4096}, {1024, 64, 1024}, {256, 64, 256}}; // QKT
          for (const auto& v : unsupported_MKN_shapes) {
            if (v[0] == qkt_mm_shape[0] && v[1] == qkt_mm_shape[1] &&
                v[2] == qkt_mm_shape[2]) {
              offload_to_mha = false;
            }
          }
          if (offload_to_mha)
            MY_LOG(1) << "Offloading to QMHA" << std::endl;
          else
            MY_LOG(1) << "Not offloading to QMHA" << std::endl;

          if (offload_to_mha) {

            std::vector<std::string> attr_nodes;
            for (auto& ni : binder) {
              if (!(*node_arg_is_constant)(*graph, *ni.second.node_arg)) {
                attr_nodes.push_back(node_arg_get_name(*ni.second.node_arg));
              }
            }

            auto temp = get_mzdk5MHA_node_args(graph, binder);
            auto input_node_args = std::get<0>(temp);
            auto input_q_params = std::get<1>(temp);
            auto output_q_params = std::get<2>(temp);

            std::vector<std::string> in_dtypes = {"uint16", "uint16", "uint16",
                                                  "int32"};
            std::vector<std::string> out_dtypes = {"uint16"};

            NodeBuilder(*graph, *self)
                .set_input_node_args(
                    {input_q_node.node_arg, input_k_node.node_arg,
                     input_v_node.node_arg, input_node_args[0]})
                .set_op_type("mzdk5MHA", "com.xilinx")
                .set_anchor_point1(*output_node.node)
                .add("input_q_params", input_q_params)
                .add("output_q_params", output_q_params)
                .add("nodes", attr_nodes)
                .add("in_dtypes", in_dtypes)
                .add("out_dtypes", out_dtypes)
                .add("output_scale", output_q_params[0])
                .add("output_zp", output_q_params[1])
                .build();

            return true; // return true if graph is modified.
          } else {

            std::vector<std::string> in_dtypes_mm = {"uint16", "uint16",
                                                     "int32"};
            std::vector<std::string> out_dtypes_mm = {"uint16"};

            // First Matmul Dynamic : QKT
            auto temp_1 = get_QMMDy_node_args(graph, binder);
            auto input_node_args_1 = std::get<0>(temp_1);
            auto input_q_params_1 = std::get<1>(temp_1);
            auto output_q_params_1 = std::get<2>(temp_1);
            auto attr_nodes_1 = std::get<3>(temp_1);
            auto qkt_qdq = std::get<4>(temp_1);

            auto temp_3 = get_QMulSoftmax_node_args(graph, binder, qkt_qdq);
            auto node_arg_qdq_params = std::get<0>(temp_3);
            // memcpy(*node_arg_qdq_params[0].data()+32,*node_arg_qdq_params[0].data(),
            // 16 *sizeof(int32_t));

            auto& mm1 = NodeBuilder(*graph, *self)
                            .set_input_node_args({input_q_node.node_arg,
                                                  input_k_node.node_arg,
                                                  node_arg_qdq_params[0]})
                            .set_op_type("QMatMulDynamicSoftmax", "com.xilinx")
                            .set_anchor_point1(*smq_node.node)
                            .add("input_q_params", input_q_params_1)
                            .add("output_q_params", output_q_params_1)
                            .add("nodes", attr_nodes_1)
                            .add("qkt", "yes")
                            .add("in_dtypes", in_dtypes_mm)
                            .add("out_dtypes", out_dtypes_mm)
                            .build();

            // Softmax
            // auto temp_3 = get_QMulSoftmax_node_args(graph, binder);
            // auto node_arg_qdq_params = std::get<0>(temp_3);
            // auto attr_nodes_sfm = std::get<1>(temp_3);
            // auto sfm_node =
            //     binder["/down_blocks.0/attentions.0/transformer_blocks.0/attn2/"
            //            "softmax_1.0/Softmax_output_0"];

            // auto& msm =
            //     NodeBuilder(*graph, *self)
            //         .set_input_node_args({node_get_output_node_args(mm1)[0],
            //                               node_arg_qdq_params[0]})
            //         .add("in_dtypes",
            //              std::vector<std::string>({"uint16", "int32"}))
            //         .add("out_dtypes", std::vector<std::string>({"uint16"}))
            //         .add("nodes", attr_nodes_sfm)
            //         .clone_attrs(*sfm_node.node)
            //         .set_op_type("QMulSoftmax", "com.xilinx")
            //         .set_anchor_point1(*smq_node.node)
            //         .build();

            // Second Matmul : VSM
            auto temp_2 = get_QMMDy_node_args(graph, binder, false);
            auto input_node_args_2 = std::get<0>(temp_2);
            auto input_q_params_2 = std::get<1>(temp_2);
            auto output_q_params_2 = std::get<2>(temp_2);
            // auto input_node_args_mm2 = get_QMMDy_node_args(graph, binder,
            // false);
            NodeBuilder(*graph, *self)
                // .set_input_node_args({input_v_node.node_arg})
                // .set_input_nodes({&msm})
                .set_input_node_args({node_get_output_node_args(mm1)[0],
                                      input_v_node.node_arg,
                                      input_node_args_2[0]})
                .set_op_type("QMatMulDynamic", "com.xilinx")
                .set_anchor_point1(*mm2q_node.node)
                .add("input_q_params", input_q_params_2)
                .add("output_q_params", output_q_params_2)
                .add("vsm", "yes")
                //  .add("nodes", attr_nodes)
                .add("in_dtypes", in_dtypes_mm)
                .add("out_dtypes", out_dtypes_mm)
                .add("output_scale", output_q_params_2[0])
                .add("output_zp", output_q_params_2[1])
                .build();

            return true;
          }
          return false;
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

DEFINE_VAIP_PASS(Dd_merge_mzdk5mha, vaip_pass_dd_merge_mzdk5mha)
