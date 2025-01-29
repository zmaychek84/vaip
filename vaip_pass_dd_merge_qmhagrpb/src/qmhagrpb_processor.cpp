/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

#include "./qmhagrpb_processor.hpp"
#include "vaip/dd/coeffs.hpp"
#include "vaip/dd/dd_utils.hpp"
#include "vitis/ai/env_config.hpp"
DEF_ENV_PARAM(DEBUG_DD_MERGE_QMHAGRPB, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_DD_MERGE_QMHAGRPB) >= n)

namespace vaip_dd_merge_qmhagrpb {
DdMergeQmhagrpbProcessor::DdMergeQmhagrpbProcessor(
    IPass& self, onnxruntime::Graph* graph, binder_t* binder,
    const std::unordered_map<std::string, std::string>& binder_params)
    : self_{self}, graph_{graph}, binder_{binder},
      binder_params_{binder_params} {}

NodeInput
DdMergeQmhagrpbProcessor::get_matched_node(const std::string& pat_name) {
  auto ni = (*binder_)[pat_name];
  CHECK(ni.node_arg != nullptr) << "cannot find id " << pat_name;
  return ni;
}

std::vector<NodeArg*> DdMergeQmhagrpbProcessor::process(int output_pat_id) {

  // Query
  float query_sc = node_arg_get_const_data_as_float(
      *graph_, *get_matched_node(binder_params_.at("query_sc")).node_arg);
  uint16_t query_zp = vaip::dd::get_zp_from_node(
      *graph_, *get_matched_node(binder_params_.at("query_zp")).node_arg);

  // Key
  float key_sc = node_arg_get_const_data_as_float(
      *graph_, *get_matched_node(binder_params_.at("key_sc")).node_arg);
  uint16_t key_zp = vaip::dd::get_zp_from_node(
      *graph_, *get_matched_node(binder_params_.at("key_zp")).node_arg);

  // Value
  float v_sc = node_arg_get_const_data_as_float(
      *graph_, *get_matched_node(binder_params_.at("v_sc")).node_arg);
  uint16_t v_zp = vaip::dd::get_zp_from_node(
      *graph_, *get_matched_node(binder_params_.at("v_zp")).node_arg);

  // QKT
  float qkt_sc = node_arg_get_const_data_as_float(
      *graph_, *get_matched_node(binder_params_.at("qkt_sc")).node_arg);
  uint16_t qkt_zp = vaip::dd::get_zp_from_node(
      *graph_, *get_matched_node(binder_params_.at("qkt_zp")).node_arg);

  // Softmax
  float sm_sc = node_arg_get_const_data_as_float(
      *graph_, *get_matched_node(binder_params_.at("sm_sc")).node_arg);
  uint16_t sm_zp = vaip::dd::get_zp_from_node(
      *graph_, *get_matched_node(binder_params_.at("sm_zp")).node_arg);

  // V*sm(QKT(grpb))
  float vsm_sc = node_arg_get_const_data_as_float(
      *graph_, *get_matched_node(binder_params_.at("vsm_sc")).node_arg);
  uint16_t vsm_zp = vaip::dd::get_zp_from_node(
      *graph_, *get_matched_node(binder_params_.at("vsm_zp")).node_arg);

  // GRPB MM weight
  auto grpb_w_node = get_matched_node(binder_params_.at("grpb_w"));
  auto grpb_w_shape = node_arg_get_shape_i64(*grpb_w_node.node_arg);
  auto grpb_w = vaip::dd::fold2D<uint8_t>(
      node_arg_get_const_data_as_u8s(*graph_, *grpb_w_node.node_arg),
      *(grpb_w_shape.get()));
  float grpb_w_sc = node_arg_get_const_data_as_float(
      *graph_, *get_matched_node(binder_params_.at("grpb_w_sc")).node_arg);
  auto grpb_w_zp = vaip::dd::get_zp_from_node(
      *graph_, *get_matched_node(binder_params_.at("grpb_w_zp")).node_arg);

  // GRPB MM
  auto grpb_b_node = get_matched_node(binder_params_.at("grpb_b"));
  auto grpb_b_shape = node_arg_get_shape_i64(*grpb_b_node.node_arg);
  auto grpb_b = vaip::dd::fold1D<uint16_t>(
      vaip::dd::get_const_as_uint16_t(*graph_, *grpb_b_node.node_arg),
      *(grpb_b_shape.get()));
  float grpb_b_sc = node_arg_get_const_data_as_float(
      *graph_, *get_matched_node(binder_params_.at("grpb_b_sc")).node_arg);
  auto grpb_b_zp = vaip::dd::get_zp_from_node(
      *graph_, *get_matched_node(binder_params_.at("grpb_b_zp")).node_arg);

  // GRPB
  float grpb_sc = node_arg_get_const_data_as_float(
      *graph_, *get_matched_node(binder_params_.at("grpb_sc")).node_arg);
  auto grpb_zp = vaip::dd::get_zp_from_node(
      *graph_, *get_matched_node(binder_params_.at("grpb_zp")).node_arg);

  // Div weight
  auto div_w = vaip::dd::get_zp_from_node(
      *graph_, *get_matched_node(binder_params_.at("div_w")).node_arg);
  float div_w_sc = node_arg_get_const_data_as_float(
      *graph_, *get_matched_node(binder_params_.at("div_w_sc")).node_arg);
  auto div_w_zp = vaip::dd::get_zp_from_node(
      *graph_, *get_matched_node(binder_params_.at("div_w_zp")).node_arg);

  // GRPB mul1
  auto mul_1_w_node = get_matched_node(binder_params_.at("mul_1_w"));
  auto mul_1_w_shape = node_arg_get_shape_i64(*mul_1_w_node.node_arg);
  auto mul_1_w = vaip::dd::fold1D<uint16_t>(
      vaip::dd::get_const_as_uint16_t(*graph_, *mul_1_w_node.node_arg),
      *(mul_1_w_shape.get()), false);
  float mul_1_w_sc = node_arg_get_const_data_as_float(
      *graph_, *get_matched_node(binder_params_.at("mul_1_w_sc")).node_arg);
  auto mul_1_w_zp = vaip::dd::get_zp_from_node(
      *graph_, *get_matched_node(binder_params_.at("mul_1_w_zp")).node_arg);

  // GRPB mul3
  auto mul_3_w_node = get_matched_node(binder_params_.at("mul_3_w"));
  auto mul_3_w_shape = *(node_arg_get_shape_i64(*mul_3_w_node.node_arg).get());
  auto mul_3_w =
      vaip::dd::get_const_as_uint16_t(*graph_, *mul_3_w_node.node_arg);
  float mul_3_w_sc = node_arg_get_const_data_as_float(
      *graph_, *get_matched_node(binder_params_.at("mul_3_w_sc")).node_arg);
  auto mul_3_w_zp = vaip::dd::get_zp_from_node(
      *graph_, *get_matched_node(binder_params_.at("mul_3_w_zp")).node_arg);

  // Add weight
  auto add_w = vaip::dd::get_zp_from_node(
      *graph_, *get_matched_node(binder_params_.at("add_w")).node_arg);
  float add_w_sc = node_arg_get_const_data_as_float(
      *graph_, *get_matched_node(binder_params_.at("add_w_sc")).node_arg);
  auto add_w_zp = vaip::dd::get_zp_from_node(
      *graph_, *get_matched_node(binder_params_.at("add_w_zp")).node_arg);

  // sub weight
  auto sub_w = vaip::dd::get_zp_from_node(
      *graph_, *get_matched_node(binder_params_.at("sub_w")).node_arg);
  float sub_w_sc = node_arg_get_const_data_as_float(
      *graph_, *get_matched_node(binder_params_.at("sub_w_sc")).node_arg);
  auto sub_w_zp = vaip::dd::get_zp_from_node(
      *graph_, *get_matched_node(binder_params_.at("sub_w_zp")).node_arg);

  /////////////////////////////// QDQ coeff calculation

  //   auto mhagrpb = get_matched_node(binder_params_.at("out"));
  auto out_zp = get_matched_node(binder_params_.at("out_zp"));

  std::string out_dtype = data_type_to_string(
      VAIP_ORT_API(node_arg_get_element_type)(*out_zp.node_arg));

  int is_qkt_smv_int16 = 0;
  int is_grpb_int16 = 0;
  if (out_dtype == "uint16") {
    is_qkt_smv_int16 = 1;
    is_grpb_int16 = 1;
  } else {
    // if mdsqr or mxgan
    is_qkt_smv_int16 = 0;
    is_grpb_int16 = 0;
  }

  std::tuple<int64_t, int32_t, int64_t, int32_t, int32_t, int32_t, int32_t>
      coeff_qkt, coeff_smv;
  if (out_dtype == "uint16") {
    coeff_qkt = vaip::dd::qmatmulcalc::qdq_act_matmul_uint16_uint16_cstm(
        query_sc, query_zp, 96, key_sc, key_zp, qkt_sc, qkt_zp);

    if (binder_params_.find("hardcode_mxganv1.2") != binder_params_.end()) {
      coeff_smv = vaip::dd::qmatmulcalc::qdq_act_matmul_uint16_uint16_cstm(
          sm_sc, sm_zp, 512, v_sc, v_zp, vsm_sc, vsm_zp);
    } else {
      coeff_smv = vaip::dd::qmatmulcalc::qdq_act_matmul_uint16_uint16_cstm(
          sm_sc, sm_zp, 128, v_sc, v_zp, vsm_sc, vsm_zp);
    }
  } else {
    coeff_qkt = vaip::dd::qmatmulcalc::qdq_act_matmul_uint8_uint8_cstm(
        query_sc, query_zp, 96, key_sc, key_zp, qkt_sc, qkt_zp);

    coeff_smv = vaip::dd::qmatmulcalc::qdq_act_matmul_uint8_uint8_cstm(
        sm_sc, sm_zp, 96, v_sc, v_zp, vsm_sc, vsm_zp);
  }

  auto qdq_params = vaip::dd::qmatmulcalc::mha_qdq_params_fill( // in32_t * 96
      coeff_qkt, coeff_smv,
      std::make_tuple(vaip::dd::qmatmulcalc::float_to_bfloat16(
                          qkt_sc / ((float)(div_w - div_w_zp) * div_w_sc)),
                      (int)qkt_zp),
      std::make_tuple(vaip::dd::qmatmulcalc::float_to_bfloat16(1.0f / sm_sc),
                      (int)sm_zp),
      is_qkt_smv_int16);

  std::vector<int64_t> c0_gate_linear;
  int32_t c1_gate_linear;
  int64_t c2_gate_linear, shift_qb_gate_linear, shift_out_gate_linear,
      matmul_shift_gate_linear;

  if (out_dtype == "uint16") {
    std::tie(c0_gate_linear, c1_gate_linear, c2_gate_linear,
             shift_qb_gate_linear, shift_out_gate_linear,
             matmul_shift_gate_linear) =
        vaip::dd::qmatmulcalc::dq_uint16A_uint8W_bias_matmul_q_param_gen(
            query_sc, query_zp, grpb_w, grpb_w_sc, grpb_w_zp, grpb_b, grpb_b_sc,
            grpb_b_zp, grpb_sc, grpb_zp);

  } else {
    std::tie(c0_gate_linear, c1_gate_linear, c2_gate_linear,
             shift_qb_gate_linear, shift_out_gate_linear,
             matmul_shift_gate_linear) =
        vaip::dd::qmatmulcalc::compute_qdq_coeff_matmul_bias(
            query_sc, (uint8_t)query_zp, grpb_w, grpb_w_sc, (uint8_t)grpb_w_zp,
            grpb_b, grpb_b_sc, (uint8_t)grpb_b_zp, grpb_sc, (uint8_t)grpb_zp);
  }

  auto gprb_vec64 =
      vaip::dd::qmatmulcalc::grpb_qgprb_vec64_fill( // int64_t * 11
          c0_gate_linear, std::get<0>(coeff_qkt), std::get<0>(coeff_smv));

  auto gprb_vec32 = vaip::dd::qmatmulcalc::gprb_vec32_fill( // int32_t * 32
      {
          c1_gate_linear,
          c2_gate_linear,
          shift_qb_gate_linear,
          shift_out_gate_linear,
          matmul_shift_gate_linear,
      },
      grpb_sc, grpb_zp, mul_3_w_sc, mul_3_w_zp, mul_1_w, mul_1_w_sc, mul_1_w_zp,
      sub_w, sub_w_sc, sub_w_zp, add_w, add_w_sc, add_w_zp, is_grpb_int16);

  // List of node arguments
  std::vector<NodeArg*> ret;

  // C4
  std::string c4_initializer_name =
      node_arg_get_name(
          *(get_matched_node(binder_params_.at("query_sc"))).node_arg) +
      "_coeff_4";
  const std::vector<int64_t> c4_initializer_shape = *(grpb_w_shape.get());
  NodeArg& c4_arg = vaip::dd::insert_named_tensor_in_graph<uint8_t>(
      graph_, c4_initializer_name,
      vaip::dd::fold1D<uint8_t>(
          node_arg_get_const_data_as_u8s(*graph_, *grpb_w_node.node_arg),
          *(grpb_w_shape.get()), false),
      c4_initializer_shape);
  ret.push_back(&c4_arg);

  // C5
  std::string c5_initializer_name =
      node_arg_get_name(
          *(get_matched_node(binder_params_.at("query_sc"))).node_arg) +
      "_coeff_5";
  const std::vector<int64_t> c5_initializer_shape = {
      (int64_t)gprb_vec64.size()};

  NodeArg& c5_arg = vaip::dd::insert_named_tensor_in_graph<int64_t>(
      graph_, c5_initializer_name, gprb_vec64, c5_initializer_shape);
  ret.push_back(&c5_arg);

  // C6
  std::string c6_initializer_name =
      node_arg_get_name(
          *(get_matched_node(binder_params_.at("query_sc"))).node_arg) +
      "_coeff_6";
  const std::vector<int64_t> c6_initializer_shape = {
      (int64_t)gprb_vec32.size()};
  NodeArg& c6_arg = vaip::dd::insert_named_tensor_in_graph<int32_t>(
      graph_, c6_initializer_name, gprb_vec32, c6_initializer_shape);
  ret.push_back(&c6_arg);

  // C7
  std::string c7_initializer_name =
      node_arg_get_name(
          *(get_matched_node(binder_params_.at("query_sc"))).node_arg) +
      "_coeff_7";

  const std::vector<int64_t> c7_initializer_shape = {(int64_t)mul_3_w_shape[1],
                                                     (int64_t)mul_3_w_shape[2],
                                                     (int64_t)mul_3_w_shape[3]};

  uint64_t mul3_w_size0 =
      std::accumulate(mul_3_w_shape.cbegin(), mul_3_w_shape.cend(), (int64_t)1,
                      std::multiplies<int64_t>{}) /
      mul_3_w_shape[0];

  if (out_dtype == "uint16") {

    std::vector<uint16_t> vec3d(mul3_w_size0);
    for (uint64_t i = 0; i < mul3_w_size0; i++)
      vec3d[i] = mul_3_w[i];

    NodeArg& c7_arg = vaip::dd::insert_named_tensor_in_graph<uint16_t>(
        graph_, c7_initializer_name, vec3d, c7_initializer_shape);

    ret.push_back(&c7_arg);

  } else {
    std::vector<uint8_t> vec3d(mul3_w_size0);
    for (uint64_t i = 0; i < mul3_w_size0; i++)
      vec3d[i] = (uint8_t)mul_3_w[i];

    NodeArg& c7_arg = vaip::dd::insert_named_tensor_in_graph<uint8_t>(
        graph_, c7_initializer_name, vec3d, c7_initializer_shape);

    ret.push_back(&c7_arg);
  }

  // C8
  std::string c8_initializer_name =
      node_arg_get_name(
          *(get_matched_node(binder_params_.at("query_sc"))).node_arg) +
      "_coeff_8";
  const std::vector<int64_t> c8_initializer_shape = {
      (int64_t)qdq_params.size()};
  NodeArg& c8_arg = vaip::dd::insert_named_tensor_in_graph<int32_t>(
      graph_, c8_initializer_name, qdq_params, c8_initializer_shape);

  ret.push_back(&c8_arg);

  return ret;
}
} // namespace vaip_dd_merge_qmhagrpb
