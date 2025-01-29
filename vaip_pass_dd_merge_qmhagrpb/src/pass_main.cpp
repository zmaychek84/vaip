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

static int64_t reduce(const std::vector<int64_t>& shape) {
  return std::accumulate(shape.cbegin(), shape.cend(), (int64_t)1,
                         std::multiplies<int64_t>{});
}

// template <typename T>
// float dq(T data, float scale, T zp){
//     return ((int64_t)data - zp) * scale;
// }

// uint16_t f2bf(float value) {
//     // Reinterpret the float as an unsigned 32-bit integer
//     uint32_t int_value;
//     std::memcpy(&int_value, &value, sizeof(value));

//     // Extract the least significant bit (lsb) of the original float's lower
//     16 bits uint32_t lsb = (int_value >> 16) & 1;

//     // Calculate the bias
//     uint32_t bias = 0x7FFF + lsb;

//     // Add the bias to the original integer value
//     uint32_t new_value = int_value + bias;

//     // Shift the new value right by 16 bits to get the bfloat16
//     representation uint16_t bfloat16_value = new_value >> 16;

//     return bfloat16_value;
// }

// template <typename T>
// static std::vector<std::vector<T>> fold2D(gsl::span<const T> ws,
//                                           const std::vector<int64_t>& shape)
//                                           {
//   CHECK(ws.size() == (size_t)reduce(shape))
//       << ws.size() << "!=" << (size_t)reduce(shape);
//   CHECK(shape.size() == 2);
//   int32_t rows = (int32_t)shape[0];
//   int32_t cols = (int32_t)shape[1];
//   std::vector<std::vector<T>> ret(rows);
//   for (int i = 0; i < rows; ++i)
//     ret[i].resize(cols);

//   for (size_t i = 0; i < ws.size(); ++i) {
//     int r = (int)i / cols;
//     int c = (int)i % cols;
//     ret[r][c] = ws[i];
//   }
//   return ret;
// }

// template <typename T>
// static std::vector<T> fold1D(gsl::span<const T> ws,
//                              const std::vector<int64_t>& shape, bool
//                              check1d=true) {
//   CHECK(ws.size() == (size_t)reduce(shape))
//       << ws.size() << "!=" << (size_t)reduce(shape);
//   if (check1d) CHECK(shape.size() == 1);
//   int32_t rows = (int32_t)reduce(shape);
//   std::vector<T> ret(rows);
//   for (int i = 0; i < rows; ++i)
//     ret[i] = ws[i];
//   return ret;
// }

// std::vector<int32_t> gprb_vec32_fill(
//     const std::vector<int64_t> &coeff_grpb,
//     float act_scale,
//     int32_t act_zero_point,
//     float wgt_scale,
//     int32_t wgt_zero_point,
//     const std::vector<uint8_t> &model_a,
//     float model_a_scale,
//     int32_t model_a_zp,
//     uint8_t model_b,
//     float model_b_scale,
//     int32_t model_b_zp,
//     uint8_t model_c,
//     float model_c_scale,
//     int32_t model_c_zp,
//     int32_t is_grpb_int16
// ) {
//     std::vector<int32_t> gprb_vec32(32, 0);

//     const int qdq_c0_idx = 0;
//     const int qdq_c1_idx = 2;
//     const int qdq_c2_idx = 3;
//     const int qdq_c3_idx = 4;
//     const int qdq_Mv_idx = 5;
//     const int qdq_Nv_idx = 6;
//     const int qdq_SQb_idx = 7;
//     const int qdq_Sout_idx = 8;
//     const int qdq_Stdm_idx = 9;

//     const int gprb_act_scale_idx = 10;
//     const int gprb_act_zero_idx = 11;
//     const int gprb_wgt_scale_idx = 12;
//     const int gprb_wgt_zero_idx = 13;
//     const int gprb_model_a_idx = 14;
//     const int gprb_model_b_idx = 26;
//     const int gprb_model_c_idx = 27;
//     const int gprb_isint16_idx = 28;

//     const int num_heads = 12;

//     gprb_vec32[qdq_c1_idx] = coeff_grpb[0];
//     gprb_vec32[qdq_c2_idx] = coeff_grpb[1];
//     gprb_vec32[qdq_c3_idx] = 0;
//     gprb_vec32[qdq_Mv_idx] = 32;
//     gprb_vec32[qdq_Nv_idx] = 8;
//     gprb_vec32[qdq_SQb_idx] = coeff_grpb[2];
//     gprb_vec32[qdq_Sout_idx] = coeff_grpb[3];
//     gprb_vec32[qdq_Stdm_idx] = coeff_grpb[4];

//     gprb_vec32[gprb_act_scale_idx] = static_cast<int32_t>(f2bf(act_scale));
//     gprb_vec32[gprb_act_zero_idx] = act_zero_point;
//     gprb_vec32[gprb_wgt_scale_idx] = static_cast<int32_t>(f2bf(wgt_scale));
//     gprb_vec32[gprb_wgt_zero_idx] = wgt_zero_point;
//     gprb_vec32[gprb_isint16_idx] = is_grpb_int16;

//     std::vector<float> model_a_bf(num_heads);
//     for (size_t i = 0; i < num_heads; ++i) {
//         model_a_bf[i] = dq<int32_t>(model_a[i], model_a_scale, model_a_zp);
//     }

//     for (int h = 0; h < num_heads; ++h) {
//         gprb_vec32[gprb_model_a_idx + h] =
//         static_cast<int32_t>(f2bf(model_a_bf[h]));
//     }

//     gprb_vec32[gprb_model_b_idx] =
//     static_cast<int32_t>(f2bf(dq<int32_t>(model_b, model_b_scale,
//     model_b_zp))); gprb_vec32[gprb_model_c_idx] =
//     static_cast<int32_t>(f2bf(dq<int32_t>(model_c, model_c_scale,
//     model_c_zp)));

//     return gprb_vec32;
// }

// std::vector<int64_t> grpb_qgprb_vec64_fill(std::vector<int64_t> bias, int64_t
// qk_qdq_c0, int64_t smv_qdq_c0){
//     std::vector<int64_t> gprb_vec64(11, 0);

//     for(int i=0; i<8; i++) gprb_vec64[i] = bias[i];

//     gprb_vec64[9] = qk_qdq_c0;
//     gprb_vec64[10] = smv_qdq_c0;

//     return gprb_vec64;
// }

std::vector<int32_t>
mha_qdq_params_fill(const std::tuple<int64_t, int32_t, int64_t, int32_t,
                                     int32_t, int32_t, int32_t>& coeff_qkt,
                    const std::tuple<int64_t, int32_t, int64_t, int32_t,
                                     int32_t, int32_t, int32_t>& coeff_smv,
                    const std::tuple<uint16_t, int>& sm_qdq_before,
                    const std::tuple<uint16_t, int>& sm_qdq_after,
                    int32_t is_qkt_smv_int16) {
  std::vector<int32_t> qdq_params(96, 0);

  constexpr int32_t qry_subv_rows = 32;
  //   constexpr int32_t qry_subv_cols = 96;
  constexpr int32_t key_subv_rows = 64;
  //   constexpr int32_t key_subv_rows_int16 = 16;
  //   constexpr int32_t key_subv_cols = 96;
  //   constexpr int32_t val_subv_rows = 64;
  constexpr int32_t val_subv_cols = 64;
  //   constexpr int32_t out_subv_rows = 32;
  //   constexpr int32_t out_subv_cols = 64;

  // QKT
  reinterpret_cast<int64_t*>(qdq_params.data())[0] = std::get<0>(coeff_qkt);
  qdq_params[(16 * 0) + 2] = std::get<1>(coeff_qkt);
  qdq_params[(16 * 0) + 3] = static_cast<int32_t>(std::get<2>(coeff_qkt));
  qdq_params[(16 * 0) + 4] = std::get<3>(coeff_qkt);
  qdq_params[(16 * 0) + 5] = qry_subv_rows;
  qdq_params[(16 * 0) + 6] = key_subv_rows;
  qdq_params[(16 * 0) + 7] = std::get<4>(coeff_qkt);
  qdq_params[(16 * 0) + 8] = std::get<5>(coeff_qkt);
  qdq_params[(16 * 0) + 9] = std::get<6>(coeff_qkt);
  qdq_params[(16 * 0) + 10] = is_qkt_smv_int16;

  // SM *V
  reinterpret_cast<int64_t*>(qdq_params.data())[8] = std::get<0>(coeff_smv);
  qdq_params[(16 * 1) + 2] = std::get<1>(coeff_smv);
  qdq_params[(16 * 1) + 3] = std::get<2>(coeff_smv);
  qdq_params[(16 * 1) + 4] = std::get<3>(coeff_smv);
  qdq_params[(16 * 1) + 5] = qry_subv_rows;
  qdq_params[(16 * 1) + 6] = val_subv_cols;
  qdq_params[(16 * 1) + 7] = std::get<4>(coeff_smv);
  qdq_params[(16 * 1) + 8] = std::get<5>(coeff_smv);
  qdq_params[(16 * 1) + 9] = std::get<6>(coeff_smv);
  qdq_params[(16 * 1) + 10] = is_qkt_smv_int16;

  // DQ before SM
  qdq_params[(16 * 2) + 0] = std::get<1>(sm_qdq_before);
  qdq_params[(16 * 2) + 1] = std::get<0>(sm_qdq_before);

  // Q after SM
  qdq_params[(16 * 3) + 0] = std::get<1>(sm_qdq_after);
  qdq_params[(16 * 3) + 1] = std::get<0>(sm_qdq_after);
  qdq_params[(16 * 3) + 2] = is_qkt_smv_int16;

  return qdq_params;
}

std::pair<int32_t, int16_t> find_closest_shifted_int32(double float_val,
                                                       int32_t max_value) {
  int32_t INT32_MAX_ = max_value; // Typically 2147483647 for int32_t
  double prev_rel_err = 1e9;
  double curr_float_val = float_val;
  int16_t shift_val = 0;
  int32_t best_int = 0;
  int64_t closest_curr_int = 0;
  int16_t best_shift_val = 0;

  while (curr_float_val <= INT32_MAX_) {
    closest_curr_int = static_cast<int64_t>(round(curr_float_val));
    double cur_rel_err =
        std::abs((double)float_val -
                 (closest_curr_int /
                  static_cast<double>((uint64_t)1 << shift_val))) /
        (double)float_val;

    if (cur_rel_err < prev_rel_err) {
      prev_rel_err = cur_rel_err;
      best_shift_val = shift_val;
      best_int = closest_curr_int;
    }

    curr_float_val *= 2;
    shift_val += 1;
  }

  return std::make_pair(best_int, best_shift_val);
}

namespace {
using namespace vaip_core;
struct MergeQMHAGRPB {
  MergeQMHAGRPB(IPass& self) : self_{self} {}
  ////////////////// Pattern includes Input DQLs
  static std::unique_ptr<Rule> create_rule(IPass* self) {
    auto com_microsoft_QuantizeLinear_171 =
        vaip::pattern_zoo::get_pattern("m_qmhagrpb_0");
    CHECK(com_microsoft_QuantizeLinear_171 != nullptr)
        << "Pattern returned is null";
    return Rule::create_rule(
        com_microsoft_QuantizeLinear_171,
        [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
          // Get nodes
          std::vector<std::string> attr_nodes;
          for (auto& ni : binder) {
            if (!(*node_arg_is_constant)(*graph, *ni.second.node_arg)) {
              attr_nodes.push_back(node_arg_get_name(*ni.second.node_arg));
            }
          }

          auto constant_0s_node = binder["constant_0s"];
          auto constant_0z_node = binder["constant_0z"];
          auto constant_1s_node = binder["constant_1s"];
          auto constant_1z_node = binder["constant_1z"];
          auto constant_2s_node = binder["constant_2s"];
          auto constant_2z_node = binder["constant_2z"];
          auto constant_3s_node = binder["constant_3s"];
          auto constant_3z_node = binder["constant_3z"];
          auto constant_162_node = binder["constant_162"];
          auto constant_163_node = binder["constant_163"];
          auto input_0_node = binder["input_0"];
          auto input_1_node = binder["input_104"];
          auto input_2_node = binder["input_131"];
          auto input_3_node = binder["input_151"];
          auto input_4_node = binder["constant_10"];

          auto query_sc_node = binder["constant_3"];
          auto query_zp_node = binder["constant_4"];

          auto key_sc_node = binder["constant_106"];
          auto key_zp_node = binder["constant_107"];

          auto qkt_sc_node = binder["constant_114"];
          auto qkt_zp_node = binder["constant_115"];

          auto sm_sc_node = binder["constant_147"];
          auto sm_zp_node = binder["constant_148"];

          auto v_sc_node = binder["constant_154"];
          auto v_zp_node = binder["constant_155"];

          auto vsm_sc_node = binder["constant_162"];
          auto vsm_zp_node = binder["constant_163"];

          auto grpb_w_node = binder["constant_10"];
          auto grpb_w_sc_node = binder["constant_11"];
          auto grpb_w_zp_node = binder["constant_12"];

          auto grpb_b_node = binder["constant_19"];
          auto grpb_b_sc_node = binder["constant_20"];
          auto grpb_b_zp_node = binder["constant_21"];

          auto grpb_sc_node = binder["constant_24"];
          auto grpb_zp_node = binder["constant_25"];

          auto div_w_node = binder["constant_118"];
          auto div_w_sc_node = binder["constant_119"];
          auto div_w_zp_node = binder["constant_120"];

          auto mul_1_w_node = binder["constant_51"];
          auto mul_1_w_sc_node = binder["constant_52"];
          auto mul_1_w_zp_node = binder["constant_53"];

          auto mul_3_w_node = binder["constant_90"];
          auto mul_3_w_sc_node = binder["constant_91"];
          auto mul_3_w_zp_node = binder["constant_92"];

          auto add_w_node = binder["constant_81"];
          auto add_sc_node = binder["constant_82"];
          auto add_zp_node = binder["constant_83"];

          auto sub_w_node = binder["constant_60"];
          auto sub_sc_node = binder["constant_61"];
          auto sub_zp_node = binder["constant_62"];

          auto input_3_dq_node = binder["input_3_dq"];

          //   auto com_microsoft_DequantizeLinear_150_node =
          //       binder["com_microsoft_DequantizeLinear_150"];

          // Need to
          // create and fill auto input_6_node = binder["constant_10"];
          // // Need to create and fill
          //   auto input_7_node = binder["constant_90"];
          // auto input_8_node = binder["constant_90"]; // Need to
          // create and fill

          auto mhagrpb = binder["com_microsoft_QuantizeLinear_171"];
          // // Fuse
          // std::cout << "-- Inside MHAGRPB create_rule NodeBuilder: start --"
          //           << std::endl;
          // std::cout << node_as_string(*mhagrpb.node) << std::endl;

          float query_sc =
              node_arg_get_const_data_as_float(*graph, *query_sc_node.node_arg);
          int64_t query_zp =
              node_arg_get_const_data_as_u8(*graph, *query_zp_node.node_arg);

          float key_sc =
              node_arg_get_const_data_as_float(*graph, *key_sc_node.node_arg);
          int64_t key_zp =
              node_arg_get_const_data_as_u8(*graph, *key_zp_node.node_arg);

          float qkt_sc =
              node_arg_get_const_data_as_float(*graph, *qkt_sc_node.node_arg);
          int64_t qkt_zp =
              node_arg_get_const_data_as_u8(*graph, *qkt_zp_node.node_arg);

          float sm_sc =
              node_arg_get_const_data_as_float(*graph, *sm_sc_node.node_arg);
          int64_t sm_zp =
              node_arg_get_const_data_as_u8(*graph, *sm_zp_node.node_arg);

          float v_sc =
              node_arg_get_const_data_as_float(*graph, *v_sc_node.node_arg);
          int64_t v_zp =
              node_arg_get_const_data_as_u8(*graph, *v_zp_node.node_arg);

          float vsm_sc =
              node_arg_get_const_data_as_float(*graph, *vsm_sc_node.node_arg);
          int64_t vsm_zp =
              node_arg_get_const_data_as_u8(*graph, *vsm_zp_node.node_arg);

          auto grpb_w_shape = node_arg_get_shape_i64(*grpb_w_node.node_arg);
          auto grpb_w = vaip::dd::fold2D<uint8_t>(
              node_arg_get_const_data_as_u8s(*graph, *grpb_w_node.node_arg),
              *(grpb_w_shape.get()));
          float grpb_w_sc = node_arg_get_const_data_as_float(
              *graph, *grpb_w_sc_node.node_arg);
          int64_t grpb_w_zp =
              node_arg_get_const_data_as_u8(*graph, *grpb_w_zp_node.node_arg);

          auto grpb_b_shape = node_arg_get_shape_i64(*grpb_b_node.node_arg);
          auto grpb_b = vaip::dd::fold1D<uint16_t>(
              vaip::dd::get_const_as_uint16_t(*graph, *grpb_b_node.node_arg),
              *(grpb_b_shape.get()));
          float grpb_b_sc = node_arg_get_const_data_as_float(
              *graph, *grpb_b_sc_node.node_arg);
          int64_t grpb_b_zp =
              node_arg_get_const_data_as_u8(*graph, *grpb_b_zp_node.node_arg);

          float grpb_sc =
              node_arg_get_const_data_as_float(*graph, *grpb_sc_node.node_arg);
          int64_t grpb_zp =
              node_arg_get_const_data_as_u8(*graph, *grpb_zp_node.node_arg);

          int64_t div_w =
              node_arg_get_const_data_as_u8(*graph, *div_w_node.node_arg);
          float div_w_sc =
              node_arg_get_const_data_as_float(*graph, *div_w_sc_node.node_arg);
          int64_t div_w_zp =
              node_arg_get_const_data_as_u8(*graph, *div_w_zp_node.node_arg);

          auto mul_1_w_shape = node_arg_get_shape_i64(*mul_1_w_node.node_arg);
          auto sst = *(mul_1_w_shape.get());
          auto mul_1_w = vaip::dd::fold1D<uint16_t>(
              vaip::dd::get_const_as_uint16_t(*graph, *mul_1_w_node.node_arg),
              *(mul_1_w_shape.get()), false);
          float mul_1_w_sc = node_arg_get_const_data_as_float(
              *graph, *mul_1_w_sc_node.node_arg);
          int64_t mul_1_w_zp =
              node_arg_get_const_data_as_u8(*graph, *mul_1_w_zp_node.node_arg);

          auto mul_3_w_shape = node_arg_get_shape_i64(*mul_3_w_node.node_arg);
          auto mul_3_w =
              node_arg_get_const_data_as_u8s(*graph, *mul_3_w_node.node_arg);
          float mul_3_w_sc = node_arg_get_const_data_as_float(
              *graph, *mul_3_w_sc_node.node_arg);
          int64_t mul_3_w_zp =
              node_arg_get_const_data_as_u8(*graph, *mul_3_w_zp_node.node_arg);

          int64_t add_w =
              node_arg_get_const_data_as_u8(*graph, *add_w_node.node_arg);
          float add_sc =
              node_arg_get_const_data_as_float(*graph, *add_sc_node.node_arg);
          int64_t add_zp =
              node_arg_get_const_data_as_u8(*graph, *add_zp_node.node_arg);

          int64_t sub_w =
              node_arg_get_const_data_as_u8(*graph, *sub_w_node.node_arg);
          float sub_sc =
              node_arg_get_const_data_as_float(*graph, *sub_sc_node.node_arg);
          int64_t sub_zp =
              node_arg_get_const_data_as_u8(*graph, *sub_zp_node.node_arg);

          int is_qkt_smv_int16 = 0;
          int is_grpb_int16 = 0;

          auto coeff_qkt =
              vaip::dd::qmatmulcalc::qdq_act_matmul_uint8_uint8_cstm(
                  query_sc, query_zp, 96, key_sc, key_zp, qkt_sc, qkt_zp);

          auto coeff_smv =
              vaip::dd::qmatmulcalc::qdq_act_matmul_uint8_uint8_cstm(
                  sm_sc, sm_zp, 512, v_sc, v_zp, vsm_sc, vsm_zp);

          auto qdq_params = mha_qdq_params_fill( // in32_t * 96
              coeff_qkt, coeff_smv,
              std::make_tuple(vaip::dd::qmatmulcalc::float_to_bfloat16(
                                  qkt_sc / ((div_w - div_w_zp) * div_w_sc)),
                              (int)qkt_zp),
              std::make_tuple(
                  vaip::dd::qmatmulcalc::float_to_bfloat16(1.0f / sm_sc),
                  (int)sm_zp),
              is_qkt_smv_int16);

          auto mul3_w_shape = *(mul_3_w_shape.get());

          // Extracting index 0 of mul_3_w
          uint64_t mul3_w_size0 = reduce(mul3_w_shape) / mul3_w_shape[0];
          std::vector<uint8_t> vec3d(mul3_w_size0);
          for (uint64_t i = 0; i < mul3_w_size0; i++)
            vec3d[i] = mul_3_w[i];

          auto [c0_gate_linear, c1_gate_linear, c2_gate_linear,
                shift_qb_gate_linear, shift_out_gate_linear,
                matmul_shift_gate_linear] =
              vaip::dd::qmatmulcalc::compute_qdq_coeff_matmul_bias(
                  query_sc, query_zp, grpb_w, grpb_w_sc, grpb_w_zp, grpb_b,
                  grpb_b_sc, grpb_b_zp, grpb_sc, grpb_zp);

          auto gprb_vec64 =
              vaip::dd::qmatmulcalc::grpb_qgprb_vec64_fill( // int64_t * 11
                  c0_gate_linear, std::get<0>(coeff_qkt),
                  std::get<0>(coeff_smv));

          auto gprb_vec32 =
              vaip::dd::qmatmulcalc::gprb_vec32_fill( // int32_t * 32
                  {
                      c1_gate_linear,
                      c2_gate_linear,
                      shift_qb_gate_linear,
                      shift_out_gate_linear,
                      matmul_shift_gate_linear,
                  },
                  grpb_sc, grpb_zp, mul_3_w_sc, mul_3_w_zp, mul_1_w, mul_1_w_sc,
                  mul_1_w_zp, sub_w, sub_sc, sub_zp, add_w, add_sc, add_zp,
                  is_grpb_int16);

          std::string c5_initializer_name =
              node_arg_get_name(*input_3_dq_node.node_arg) + "_mha_np_grpb_vec";
          const std::vector<int64_t> c5_initializer_shape = {
              (int64_t)gprb_vec64.size()};
          auto c5_tensor = tensor_proto_new_i64(
              c5_initializer_name, {(int64_t)gprb_vec64.size()}, gprb_vec64);
          VAIP_ORT_API(graph_add_initialized_tensor)(*graph, *c5_tensor);
          auto& c5_arg = VAIP_ORT_API(node_arg_new)(
              *graph, c5_initializer_name, &c5_initializer_shape,
              ONNX_NAMESPACE::TensorProto_DataType_INT64);

          std::string c6_initializer_name =
              node_arg_get_name(*input_3_dq_node.node_arg) + "_mha_np_grpb_qdq";
          const std::vector<int64_t> c6_initializer_shape = {
              (int64_t)gprb_vec32.size()};
          auto c6_tensor = tensor_proto_new_i32(
              c6_initializer_name, {(int64_t)gprb_vec32.size()}, gprb_vec32);
          VAIP_ORT_API(graph_add_initialized_tensor)(*graph, *c6_tensor);

          auto& c6_arg = VAIP_ORT_API(node_arg_new)(
              *graph, c6_initializer_name, &c6_initializer_shape,
              ONNX_NAMESPACE::TensorProto_DataType_INT32);

          std::string c7_initializer_name =
              node_arg_get_name(*input_3_dq_node.node_arg) +
              "777"; // node_arg_get_name(*mul_3_w_node.node_arg) + "0";
          const std::vector<int64_t> c7_initializer_shape = {
              (int64_t)mul3_w_shape[1], (int64_t)mul3_w_shape[2],
              (int64_t)mul3_w_shape[3]};

          NodeArg& c7_arg = vaip::dd::insert_named_tensor_in_graph<uint8_t>(
              graph, c7_initializer_name, vec3d, c7_initializer_shape);

          std::string c8_initializer_name =
              node_arg_get_name(*input_3_dq_node.node_arg) +
              "_mha_np_grpb_vec888";
          const std::vector<int64_t> c8_initializer_shape = {
              (int64_t)qdq_params.size()};
          auto c8_tensor = tensor_proto_new_i32(
              c8_initializer_name, c8_initializer_shape, qdq_params);
          VAIP_ORT_API(graph_add_initialized_tensor)(*graph, *c8_tensor);

          auto& c8_arg = VAIP_ORT_API(node_arg_new)(
              *graph, c8_initializer_name, &c8_initializer_shape,
              ONNX_NAMESPACE::TensorProto_DataType_INT32);

          //           //qdq_3
          //   std::vector<int64_t> qdq3(16, 34); // gprb_vec64
          //   std::string qdq3_initializer_name =
          //   node_arg_get_name(*mul_1_w_zp_node.node_arg) + "_qdq3_"; const
          //   std::vector<int64_t> qdq3_initializer_shape =
          //   {(int64_t)qdq3.size()}; auto qdq3_tensor =
          //   tensor_proto_new_i64(qdq3_initializer_name,
          //   {(int64_t)qdq3.size()}, qdq3);
          //   VAIP_ORT_API(graph_add_initialized_tensor)(*graph, *qdq3_tensor);
          //   auto& qdq3_arg = VAIP_ORT_API(node_arg_new)(*graph,
          //   qdq3_initializer_name, &qdq3_initializer_shape,
          //   ONNX_NAMESPACE::TensorProto_DataType_INT64);

          // std::cout<< "------------------32"<<std::endl;
          std::vector<float> input_q_params;
          input_q_params.push_back(node_arg_get_const_data_as_float(
              *graph, *constant_0s_node.node_arg));
          // std::cout<<"0z"<<(int)node_arg_get_element_type(*constant_0z_node.node_arg)<<std::endl;

          input_q_params.push_back(float(node_arg_get_const_data_as_u8(
              *graph, *constant_0z_node.node_arg)));
          input_q_params.push_back(node_arg_get_const_data_as_float(
              *graph, *constant_1s_node.node_arg));
          // std::cout<<"1z"<<(int)node_arg_get_element_type(*constant_1z_node.node_arg)<<std::endl;

          input_q_params.push_back(float(node_arg_get_const_data_as_u8(
              *graph, *constant_1z_node.node_arg)));
          input_q_params.push_back(node_arg_get_const_data_as_float(
              *graph, *constant_3s_node.node_arg));
          // std::cout<<"3z"<<(int)node_arg_get_element_type(*constant_3z_node.node_arg)<<std::endl;

          input_q_params.push_back(float(node_arg_get_const_data_as_u8(
              *graph, *constant_3z_node.node_arg)));
          input_q_params.push_back(node_arg_get_const_data_as_float(
              *graph, *constant_2s_node.node_arg));
          // std::cout<<"2z"<<(int)node_arg_get_element_type(*constant_2z_node.node_arg)<<std::endl;

          input_q_params.push_back(float(node_arg_get_const_data_as_u16(
              *graph, *constant_2z_node.node_arg)));
          // input_q_params.push_back(0.0f);

          std::vector<float> output_q_params;
          output_q_params.push_back(node_arg_get_const_data_as_float(
              *graph, *constant_162_node.node_arg));
          output_q_params.push_back(float(node_arg_get_const_data_as_u8(
              *graph, *constant_163_node.node_arg)));
          auto mhagrpb_builder = NodeBuilder(*graph, *self);
          mhagrpb_builder.set_input_node_args({
              input_0_node.node_arg, input_1_node.node_arg,
              input_3_node.node_arg, input_2_node.node_arg,
              input_4_node.node_arg, &c5_arg, &c6_arg, &c7_arg, &c8_arg
              // // input_5_node.node_arg,
              // // input_6_node.node_arg,
              // // input_8_node.node_arg
          });

          // 4: as is
          // 5: node_arg_get_name(*input_3_dq_node.node_arg) +
          // "_mha_np_grpb_vec" 6: node_arg_get_name(*input_3_dq_node.node_arg)
          // + "_mha_np_grpb_qdq" //wrong 7:
          // node_arg_get_name(*mul_3_w_node.node_arg) + "0" 8:
          // node_arg_get_name(*input_3_node.node_arg) + "_qdq_";
          std::vector<std::string> in_dtypes = {"uint8",    "uint8", "uint8",
                                                "bfloat16", "uint8", "int64",
                                                "int32",    "uint8", "int32"};
          // Add conditional code here (Below may only work for mdsqr)
          std::vector<std::string> out_dtypes = {"uint8"};
          mhagrpb_builder.set_op_type("QMHAGRPB", "com.xilinx");
          // mhagrpb_builder.clone_inputs(*mhagrpb.node);
          mhagrpb_builder.clone_data_type(*mhagrpb.node);
          mhagrpb_builder.clone_attrs(*mhagrpb.node);
          mhagrpb_builder.set_anchor_point1(*mhagrpb.node);
          mhagrpb_builder.add("nodes", attr_nodes);
          mhagrpb_builder.add("input_q_params", input_q_params);
          mhagrpb_builder.add("output_q_params", output_q_params);
          mhagrpb_builder.add("in_dtypes", in_dtypes);
          mhagrpb_builder.add("out_dtypes", out_dtypes);
          mhagrpb_builder.build();

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

DEFINE_VAIP_PASS(MergeQMHAGRPB, vaip_pass_dd_merge_qmhagrpb)
