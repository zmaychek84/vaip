/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */
#include <glog/logging.h>

#include "vaip/dd/coeffs.hpp"
#include "vaip/dd/dd_utils.hpp"
#include "vaip/vaip.hpp"
#include "vitis/ai/env_config.hpp"

#include "vaip/dd/coeffs.hpp"
#include "vaip/dd/dd_utils.hpp"
#include "vaip/pattern_zoo.hpp"

DEF_ENV_PARAM(DEBUG_DD_MERGE_DeMHA, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_DD_MERGE_DeMHA) >= n)

namespace {
using namespace vaip_core;

/*
- Below function gets the QDQ params required for the QDeMHA
- We need QKt matmul, SM*V matmul , Softmax, pos_con_add (positinal encoding +
context encoing addition), attention mask add
- Fills all the QDQ params in qdq_tensor
*/
std::tuple<std::vector<NodeArg*>, std::vector<float>, std::vector<float>>
get_DeMHA_node_args(onnxruntime::Graph* graph_, binder_t& binder_) {

  // query
  auto input_q_node = binder_["/deberta/encoder/layer.0/attention/self/"
                              "Reshape_1_output_0_QuantizeLinear_Output"];
  auto q_sc_node = binder_
      ["/deberta/encoder/layer.0/attention/self/query_proj/Add_output_0_scale"];
  auto q_zp_node = binder_["/deberta/encoder/layer.0/attention/self/query_proj/"
                           "Add_output_0_zero_point"];
  auto q_shape_back =
      (*node_arg_get_shape_i64(*input_q_node.node_arg).get()).back();
  float q_sc = node_arg_get_const_data_as_float(*graph_, *q_sc_node.node_arg);
  uint16_t q_zp = vaip::dd::get_zp_from_node(*graph_, *q_zp_node.node_arg);

  // key
  auto input_k_node = binder_["/deberta/encoder/layer.0/attention/self/"
                              "Reshape_3_output_0_QuantizeLinear_Output"];
  auto k_sc_node = binder_
      ["/deberta/encoder/layer.0/attention/self/key_proj/Add_output_0_scale"];
  auto k_zp_node = binder_["/deberta/encoder/layer.0/attention/self/key_proj/"
                           "Add_output_0_zero_point"];
  auto k_shape_back =
      (*node_arg_get_shape_i64(*input_k_node.node_arg).get()).back();
  float k_sc = node_arg_get_const_data_as_float(*graph_, *k_sc_node.node_arg);
  uint16_t k_zp = vaip::dd::get_zp_from_node(*graph_, *k_zp_node.node_arg);

  // value
  auto v_sc_node = binder_
      ["/deberta/encoder/layer.0/attention/self/value_proj/Add_output_0_scale"];
  auto v_zp_node = binder_["/deberta/encoder/layer.0/attention/self/value_proj/"
                           "Add_output_0_zero_point"];
  float v_sc = node_arg_get_const_data_as_float(*graph_, *v_sc_node.node_arg);
  uint16_t v_zp = vaip::dd::get_zp_from_node(*graph_, *v_zp_node.node_arg);

  // qkt
  auto qkt_sc_node =
      binder_["/deberta/encoder/layer.0/attention/self/MatMul_output_0_scale"];
  auto qkt_zp_node = binder_
      ["/deberta/encoder/layer.0/attention/self/MatMul_output_0_zero_point"];
  float qkt_sc =
      node_arg_get_const_data_as_float(*graph_, *qkt_sc_node.node_arg);
  uint16_t qkt_zp = vaip::dd::get_zp_from_node(*graph_, *qkt_zp_node.node_arg);

  // sm
  auto sm_sc_inp_node =
      binder_["/deberta/encoder/layer.0/attention/self/Add_5_output_0_scale"];
  auto sm_zp_inp_node = binder_
      ["/deberta/encoder/layer.0/attention/self/Add_5_output_0_zero_point"];
  float sm_inp_sc =
      node_arg_get_const_data_as_float(*graph_, *sm_sc_inp_node.node_arg);
  uint16_t sm_inp_zp =
      vaip::dd::get_zp_from_node(*graph_, *sm_zp_inp_node.node_arg);

  auto sm_sc_node =
      binder_["/deberta/encoder/layer.0/attention/self/Softmax_output_0_scale"];
  auto sm_zp_node = binder_
      ["/deberta/encoder/layer.0/attention/self/Softmax_output_0_zero_point"];
  float sm_sc = node_arg_get_const_data_as_float(*graph_, *sm_sc_node.node_arg);
  uint16_t sm_zp = vaip::dd::get_zp_from_node(*graph_, *sm_zp_node.node_arg);

  // vsm
  auto vsm_sc_node = binder_
      ["/deberta/encoder/layer.0/attention/self/MatMul_3_output_0_scale"];
  auto vsm_zp_node = binder_
      ["/deberta/encoder/layer.0/attention/self/MatMul_3_output_0_zero_point"];
  float vsm_sc =
      node_arg_get_const_data_as_float(*graph_, *vsm_sc_node.node_arg);
  uint16_t vsm_zp = vaip::dd::get_zp_from_node(*graph_, *vsm_zp_node.node_arg);
  auto final_quant_node = binder_["/deberta/encoder/layer.0/attention/self/"
                                  "Reshape_15_output_0_QuantizeLinear_Output"];

  auto coeff_qkt = vaip::dd::qmatmulcalc::qdq_act_matmul_uint16_uint16_cstm(
      q_sc, q_zp, q_shape_back, k_sc, k_zp, qkt_sc, qkt_zp);

  auto coeff_smv = vaip::dd::qmatmulcalc::qdq_act_matmul_uint16_uint16_cstm(
      sm_sc, sm_zp, k_shape_back, v_sc, v_zp, vsm_sc, vsm_zp);

  // pos_con_add (positional encoding + Contextencoding addition) input QDQ
  // params
  auto pos_con_add_inp_2_sc_node =
      binder_["/deberta/encoder/layer.0/attention/self/Add_3_output_0_scale"];
  auto pos_con_add_inp_2_zp_node = binder_
      ["/deberta/encoder/layer.0/attention/self/Add_3_output_0_zero_point"];
  auto pos_con_add_inp_2_sc = node_arg_get_const_data_as_float(
      *graph_, *pos_con_add_inp_2_sc_node.node_arg);
  auto pos_con_add_inp_2_zp =
      vaip::dd::get_zp_from_node(*graph_, *pos_con_add_inp_2_zp_node.node_arg);

  auto pos_con_add_inp_1_sc_node =
      binder_["/deberta/encoder/layer.0/attention/self/Div_output_0_scale"];
  auto pos_con_add_inp_1_zp_node = binder_
      ["/deberta/encoder/layer.0/attention/self/Div_output_0_zero_point"];
  auto pos_con_add_inp_1_sc = node_arg_get_const_data_as_float(
      *graph_, *pos_con_add_inp_1_sc_node.node_arg);
  auto pos_con_add_inp_1_zp =
      vaip::dd::get_zp_from_node(*graph_, *pos_con_add_inp_1_zp_node.node_arg);

  auto pos_con_add_out_sc_node =
      binder_["/deberta/encoder/layer.0/attention/self/Add_4_output_0_scale"];
  auto pos_con_add_out_zp_node = binder_
      ["/deberta/encoder/layer.0/attention/self/Add_4_output_0_zero_point"];
  auto pos_con_add_out_sc = node_arg_get_const_data_as_float(
      *graph_, *pos_con_add_out_sc_node.node_arg);
  auto pos_con_add_out_zp =
      vaip::dd::get_zp_from_node(*graph_, *pos_con_add_out_zp_node.node_arg);

  auto pos_con_qdq_params = std::make_tuple(
      pos_con_add_inp_1_sc, pos_con_add_inp_1_zp, pos_con_add_inp_2_sc,
      pos_con_add_inp_2_zp, pos_con_add_out_sc, pos_con_add_out_zp);

  // Attention mask add qdq params
  auto att_add_input_1_sc_node =
      binder_["/deberta/encoder/layer.0/attention/self/Add_4_output_0_scale"];
  auto att_add_input_1_zp_node = binder_
      ["/deberta/encoder/layer.0/attention/self/Add_4_output_0_zero_point"];
  auto att_add_input_1_sc = node_arg_get_const_data_as_float(
      *graph_, *att_add_input_1_sc_node.node_arg);
  auto att_add_input_1_zp =
      vaip::dd::get_zp_from_node(*graph_, *att_add_input_1_zp_node.node_arg);

  auto att_add_input_sc_node =
      binder_["/deberta/encoder/layer.0/attention/self/Mul_2_output_0_scale"];
  auto att_add_input_zp_node = binder_
      ["/deberta/encoder/layer.0/attention/self/Mul_2_output_0_zero_point"];
  auto att_add_input_sc = node_arg_get_const_data_as_float(
      *graph_, *att_add_input_sc_node.node_arg);
  auto att_add_input_zp =
      vaip::dd::get_zp_from_node(*graph_, *att_add_input_zp_node.node_arg);

  auto att_add_output_sc_node =
      binder_["/deberta/encoder/layer.0/attention/self/Add_5_output_0_scale"];
  auto att_add_output_zp_node = binder_
      ["/deberta/encoder/layer.0/attention/self/Add_5_output_0_zero_point"];
  auto att_add_output_sc = node_arg_get_const_data_as_float(
      *graph_, *att_add_output_sc_node.node_arg);
  auto att_add_output_zp =
      vaip::dd::get_zp_from_node(*graph_, *att_add_output_zp_node.node_arg);

  auto att_add_qdq_params =
      std::make_tuple(att_add_input_1_sc, att_add_input_1_zp, att_add_input_sc,
                      att_add_input_zp, att_add_output_sc, att_add_output_zp);

  // param fill
  std::vector<int32_t> qdq_params =
      vaip::dd::qmatmulcalc::DeMHA_qdq_params_fill( // in32_t * 96
          coeff_qkt, coeff_smv,
          std::make_tuple(vaip::dd::qmatmulcalc::float_to_bfloat16(sm_inp_sc),
                          (int)sm_inp_zp),
          std::make_tuple(
              vaip::dd::qmatmulcalc::float_to_bfloat16(1.0f / sm_sc),
              (int)sm_zp),
          pos_con_qdq_params, att_add_qdq_params, 1, 0);

  std::string initializer_name =
      node_arg_get_name(*q_sc_node.node_arg) + "_qdq_";
  const std::vector<int64_t> shape = {6, 16};

  auto& qdq_arg = vaip::dd::insert_named_tensor_in_graph<int32_t>(
      graph_, initializer_name, qdq_params, shape);

  std::vector<NodeArg*> ret;
  ret.push_back(&qdq_arg);

  std::vector<float> in_q_params = {q_sc,
                                    float(q_zp),
                                    k_sc,
                                    float(k_zp),
                                    v_sc,
                                    float(v_zp),
                                    float(pos_con_add_inp_2_sc),
                                    float(pos_con_add_inp_2_zp),
                                    float(att_add_input_sc),
                                    float(att_add_output_zp)};

  std::vector<float> out_q_params = {vsm_sc, float(vsm_zp)};

  return std::make_tuple(ret, in_q_params, out_q_params);
}

struct Dd_merge_DeMHA {
  Dd_merge_DeMHA(IPass& self) : self_{self} {}
  std::unique_ptr<Rule> create_rule(IPass* self) {
    auto com_microsoft_QuantizeLinear_11 =
        vaip::pattern_zoo::get_pattern("m_DeMHA");
    CHECK(com_microsoft_QuantizeLinear_11 != nullptr)
        << "Pattern returned is null";
    return Rule::create_rule(
        com_microsoft_QuantizeLinear_11,
        [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
          MY_LOG(1) << "mzdk5 out node : "
                    << node_arg_get_name(
                           *binder["/deberta/encoder/layer.0/attention/self/"
                                   "Reshape_15_output_0_QuantizeLinear_Output"]
                                .node_arg);

          // Input nodes
          //   if (
          //   "/deberta/encoder/layer.0/attention/self/Reshape_15_output_0_QuantizeLinear_Output"
          //   != node_arg_get_name(
          //                    *binder["/deberta/encoder/layer.0/attention/self/"
          //                            "Reshape_15_output_0_QuantizeLinear_Output"]
          //                         .node_arg))
          //                         {
          //                             return false;
          //                         }

          auto input_q_node =
              binder["/deberta/encoder/layer.0/attention/self/"
                     "Reshape_1_output_0_QuantizeLinear_Output"];
          auto input_k_node =
              binder["/deberta/encoder/layer.0/attention/self/"
                     "Reshape_3_output_0_QuantizeLinear_Output"];
          auto input_v_node =
              binder["/deberta/encoder/layer.0/attention/self/"
                     "Reshape_5_output_0_QuantizeLinear_Output"];
          auto input_add_de_node =
              binder["/deberta/encoder/layer.0/attention/self/"
                     "Add_3_output_0_QuantizeLinear_Output"];
          auto input_add_att_mask_node =
              binder["/deberta/encoder/layer.0/attention/self/"
                     "Mul_2_output_0_QuantizeLinear_Output"];

          auto q_shape = *node_arg_get_shape_i64(*input_q_node.node_arg).get();
          auto kt_shape = *node_arg_get_shape_i64(*input_k_node.node_arg).get();

          //   // Output nodes
          auto output_node =
              binder["/deberta/encoder/layer.0/attention/self/"
                     "Reshape_15_output_0_QuantizeLinear_Output"];
          auto node_name = node_arg_get_name(
              *binder["/deberta/encoder/layer.0/attention/self/"
                      "Reshape_15_output_0_QuantizeLinear_Output"]
                   .node_arg);

          //    std::vector<std::string> layers = {"layer.0","layer.2"};
          //     bool temp_1 = false;

          //     for (const auto& layer : layers) {
          //         if (node_name.find(layer) != std::string::npos) {
          //             temp_1 = true;
          //             break; // No need to continue once we find a match
          //         }
          //     }
          //     std::cout<<"temp_1="<<temp_1<<std::endl;
          //     if (!temp_1) {
          //         return false;
          //     }

          auto temp = get_DeMHA_node_args(graph, binder);
          auto input_node_args = std::get<0>(temp);
          auto input_q_params = std::get<1>(temp);
          auto output_q_params = std::get<2>(temp);

          auto attr_nodes = vaip::dd::get_node_names(graph, binder);
          MY_LOG(1) << "mzdk5 out node : "
                    << node_arg_get_name(
                           *binder["/deberta/encoder/layer.0/attention/self/"
                                   "Reshape_15_output_0_QuantizeLinear_Output"]
                                .node_arg);

          std::vector<std::string> in_dtypes = {"uint16", "uint16", "uint16",
                                                "uint16", "uint16", "int32"};
          std::vector<std::string> out_dtypes = {"uint16"};
          NodeBuilder(*graph, *self)
              .set_input_node_args(
                  {input_q_node.node_arg, input_k_node.node_arg,
                   input_v_node.node_arg, input_add_de_node.node_arg,
                   input_add_att_mask_node.node_arg, input_node_args[0]})
              .set_op_type("QDeMHA", "com.xilinx")
              .set_anchor_point1(*output_node.node)
              .add("input_q_params", input_q_params)
              .add("output_q_params", output_q_params)
              .add("nodes", attr_nodes)
              .add("in_dtypes", in_dtypes)
              .add("design_param", "4x4")
              .add("out_dtypes", out_dtypes)
              .add("output_scale", output_q_params[0])
              .add("output_zp", output_q_params[1])
              .build();
          return true;

          MY_LOG(1) << "mzdk5 out node : "
                    << node_arg_get_name(
                           *binder["/deberta/encoder/layer.0/attention/self/"
                                   "Reshape_15_output_0_QuantizeLinear_Output"]
                                .node_arg);
          std::cout << "Matched MHA pattern" << std::endl;
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

DEFINE_VAIP_PASS(Dd_merge_DeMHA, vaip_pass_dd_merge_DeMHA)
