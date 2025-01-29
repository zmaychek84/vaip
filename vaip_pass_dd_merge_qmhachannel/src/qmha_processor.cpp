/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

#include "vaip/my_ort.h"
#ifdef _WIN32
#  pragma warning(push)
#  pragma warning(disable : 4200)
#else
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wconversion"
#endif
#include "vaip/dd/coeffs.hpp"
#ifdef _WIN32
#  pragma warning(pop)
#else
#  pragma GCC diagnostic pop
#endif

#include "qmha/qmha_processor.hpp"
#include "vaip/dd/dd_utils.hpp"
#include <glog/logging.h>
DEF_ENV_PARAM(DEBUG_DD_MERGE_QMHACHANNEL, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_DD_MERGE_QMHACHANNEL) >= n)

namespace vaip_dd_merge_qma {
DdMergeQmhaProcessor::DdMergeQmhaProcessor(
    IPass& self, onnxruntime::Graph* graph, binder_t* binder,
    const std::unordered_map<std::string, std::vector<std::string>>&
        binder_params)
    : self_{self}, graph_{graph}, binder_{binder},
      binder_params_{binder_params} {}

float DdMergeQmhaProcessor::node_arg_get_const_data_as_float(
    const std::string& name, size_t index) {
  auto ni = get_node_input(name, index);
  return vaip_core::node_arg_get_const_data_as_float(*graph_, *ni.node_arg);
}

uint16_t
DdMergeQmhaProcessor::node_arg_get_const_data_as_u16(const std::string& name,
                                                     size_t index) {
  auto ni = get_node_input(name, index);
  return vaip_core::node_arg_get_const_data_as_u16(*graph_, *ni.node_arg);
}

NodeInput DdMergeQmhaProcessor::get_node_input(const std::string& name,
                                               size_t index) const {
  auto it = binder_params_.find(name);
  CHECK(it != binder_params_.end()) << "cannot find " << name;
  auto& v = it->second;

  CHECK_LT(index, v.size()) << "out of range: " << name;
  const auto& pat_name = v[index];
  auto ni = (*binder_)[pat_name];
  CHECK(ni.node_arg != nullptr)
      << "cannot find id " << pat_name << " name=" << name;
  return ni;
}

int64_t DdMergeQmhaProcessor::get_k_dim(NodeInput ni1, NodeInput ni2) {
  auto& arg1 = *ni1.node_arg;
  auto& arg2 = *ni2.node_arg;
  auto shape1 = node_arg_get_shape_i64(arg1);
  auto shape2 = node_arg_get_shape_i64(arg2);
  CHECK(shape1 != nullptr);
  CHECK(shape2 != nullptr);
  auto common = std::vector<int64_t>{};
  for (auto i : *shape1) {
    if (std::find(shape2->begin(), shape2->end(), i) != shape2->end()) {
      common.push_back(i);
    }
  }
  MY_LOG(1) << "i1=" << node_arg_as_string(arg1) << " "
            << "i2=" << node_arg_as_string(arg2) << " ";
  CHECK(!common.empty());
  return common.back();
}

const NodeArg& DdMergeQmhaProcessor::create_node_arg(
    onnxruntime::Graph& graph, const std::string& name,
    const std::vector<int64_t>& shape, const std::vector<int64_t>& value) {
  auto tensor = tensor_proto_new_i64(name, shape, value);
  VAIP_ORT_API(graph_add_initialized_tensor)(graph, *tensor);
  return VAIP_ORT_API(node_arg_new)(graph, name, &shape,
                                    ONNX_NAMESPACE::TensorProto_DataType_INT64);
}

const NodeArg& DdMergeQmhaProcessor::process(int output_pat_id) {
  auto QKT_input_1_scale =
      node_arg_get_const_data_as_float("QKT_input_qparams", 0);
  auto QKT_input_1_zp = node_arg_get_const_data_as_u16("QKT_input_qparams", 1);

  auto QKT_input_2_scale =
      node_arg_get_const_data_as_float("QKT_input_qparams", 2);

  auto QKT_input_2_zp = node_arg_get_const_data_as_u16("QKT_input_qparams", 3);

  auto QKT_output_scale =
      node_arg_get_const_data_as_float("QKT_output_qparams", 0);
  auto QKT_output_zp = node_arg_get_const_data_as_u16("QKT_output_qparams", 1);

  auto VSQKT_input_1_scale =
      node_arg_get_const_data_as_float("VSQKT_input_qparams", 0);
  auto VSQKT_input_1_zp =
      node_arg_get_const_data_as_u16("VSQKT_input_qparams", 1);

  auto VSQKT_input_2_scale =
      node_arg_get_const_data_as_float("VSQKT_input_qparams", 2);
  auto VSQKT_input_2_zp =
      node_arg_get_const_data_as_u16("VSQKT_input_qparams", 3);

  auto VSQKT_output_scale =
      node_arg_get_const_data_as_float("VSQKT_output_qparams", 0);
  auto VSQKT_output_zp =
      node_arg_get_const_data_as_u16("VSQKT_output_qparams", 1);
  auto QKT_k_dim = get_k_dim(get_node_input("MATMUL_input", 0),
                             get_node_input("MATMUL_input", 1));
  auto VSQKT_k_dim = get_k_dim(get_node_input("VSMATMUL_input", 0),
                               get_node_input("VSMATMUL_input", 1));

  MY_LOG(1) << "QKT_input_1_scale " << QKT_input_1_scale << " "     //
            << "QKT_input_1_zp " << QKT_input_1_zp << " "           //
            << "QKT_input_2_scale " << QKT_input_2_scale << " "     //
            << "QKT_input_2_zp " << QKT_input_2_zp << " "           //
            << "QKT_output_scale " << QKT_output_scale << " "       //
            << "QKT_output_zp " << QKT_output_zp << " "             //
            << "VSQKT_input_1_scale " << VSQKT_input_1_scale << " " //
            << "VSQKT_input_1_zp " << VSQKT_input_1_zp << " "       //
            << "VSQKT_input_2_scale " << VSQKT_input_2_scale << " " //
            << "VSQKT_input_2_zp " << VSQKT_input_2_zp << " "       //
            << "VSQKT_output_scale " << VSQKT_output_scale << " "   //
            << "VSQKT_output_zp " << VSQKT_output_zp << " "         //
            << "QKT_k_dim " << QKT_k_dim << " "                     //
            << "VSQKT_k_dim " << VSQKT_k_dim << " "                 //
      ;
  auto coeff_qkt = vaip::dd::qmatmulcalc::qdq_act_matmul_uint16_uint16_cstm(
      QKT_input_1_scale, QKT_input_1_zp, QKT_k_dim, QKT_input_2_scale,
      QKT_input_2_zp, QKT_output_scale, QKT_output_zp);
  auto coeff_smv = vaip::dd::qmatmulcalc::qdq_act_matmul_uint16_uint16_cstm(
      VSQKT_input_1_scale, VSQKT_input_1_zp, VSQKT_k_dim, VSQKT_input_2_scale,
      VSQKT_input_2_zp, VSQKT_output_scale, VSQKT_output_zp);
  MY_LOG(1) << "coeff_qkt "                         //
            << " " << std::get<0>(coeff_qkt) << " " //
            << " " << std::get<1>(coeff_qkt) << " " //
            << " " << std::get<2>(coeff_qkt) << " " //
            << " " << std::get<3>(coeff_qkt) << " " //
            << " " << std::get<4>(coeff_qkt) << " " //
            << " " << std::get<5>(coeff_qkt) << " " //
      ;
  MY_LOG(1) << "coeff_smv "                         //
            << " " << std::get<0>(coeff_smv) << " " //
            << " " << std::get<1>(coeff_smv) << " " //
            << " " << std::get<2>(coeff_smv) << " " //
            << " " << std::get<3>(coeff_smv) << " " //
            << " " << std::get<4>(coeff_smv) << " " //
            << " " << std::get<5>(coeff_smv) << " " //
      ;

  auto qdq_sm_in = vaip::dd::qmatmulcalc::calc_lrn_coeff(
      QKT_output_scale * 1.442695041f, QKT_output_zp);

  auto softmax_output_scale =
      node_arg_get_const_data_as_float("softmax_output_qparams", 0);
  auto softmax_output_zp =
      node_arg_get_const_data_as_u16("softmax_output_qparams", 1);

  MY_LOG(1) << "softmax_output_scale " << softmax_output_scale << " " //
            << "softmax_output_zp " << softmax_output_zp << " "       //
      ;
  auto qdq_sm_out = vaip::dd::qmatmulcalc::calc_lrn_coeff(
      1.0f / softmax_output_scale, softmax_output_zp);

  MY_LOG(1) << "qdq_sm_in (" << std::get<0>(qdq_sm_in) << ","
            << std::get<1>(qdq_sm_in) << ") "  //
            << "qdq_sm_in (" << std::get<0>(qdq_sm_out) << ","
            << std::get<1>(qdq_sm_out) << ") " //
      ;
  auto MUL_weight_value =
      (float)node_arg_get_const_data_as_u16("MUL_weight_qparams", 0);
  auto MUL_weight_scale =
      node_arg_get_const_data_as_float("MUL_weight_qparams", 1);
  auto MUL_weight_zp =
      (float)node_arg_get_const_data_as_u16("MUL_weight_qparams", 2);
  MY_LOG(1) << "MUL_weight_value " << MUL_weight_value << " " //
            << "MUL_weight_scale " << MUL_weight_scale << " " //
            << "MUL_weight_zp " << MUL_weight_zp << " "       //
      ;
  auto mul_float_value =
      (MUL_weight_value - MUL_weight_zp) * (MUL_weight_scale);
  MY_LOG(1) << "mul_float_value " << mul_float_value << " " //
      ;

  auto MUL_input_scale =
      node_arg_get_const_data_as_float("MUL_input_qparams", 0);
  auto MUL_input_zp = node_arg_get_const_data_as_u16("MUL_input_qparams", 1);
  MY_LOG(1) << "MUL_input_scale " << MUL_input_scale << " " //
            << "MUL_input_zp " << MUL_input_zp << " "       //
      ;
  auto qdq_mul_in = vaip::dd::qmatmulcalc::calc_lrn_coeff(
      mul_float_value * MUL_input_scale, MUL_input_zp);
  MY_LOG(1) << "qdq_mul_in " << std::get<0>(qdq_mul_in) << ","
            << std::get<1>(qdq_mul_in) << ")"
            << " " //
      ;

  auto MUL_output_scale =
      node_arg_get_const_data_as_float("MUL_output_qparams", 0);
  auto MUL_output_zp = node_arg_get_const_data_as_u16("MUL_output_qparams", 1);
  MY_LOG(1) << "MUL_output_scale " << MUL_output_scale << " " //
            << "MUL_output_zp " << MUL_output_zp << " "       //
      ;
  auto qdq_mul_out = vaip::dd::qmatmulcalc::calc_lrn_coeff(
      1.0f / MUL_output_scale, MUL_output_zp);
  MY_LOG(1) << "qdq_mul_out (" << std::get<0>(qdq_mul_out) << ","
            << std::get<1>(qdq_mul_out) << ")"
            << " " //
      ;

  auto is_qkt_smv_int16 = true;
  // TODO: return type?
  auto qdq_params =
      vaip::dd::qmatmulcalc::mha_channel_qdq_params_fill(coeff_qkt,       //
                                                         coeff_smv,       //
                                                         qdq_sm_in,       //
                                                         qdq_sm_out,      //
                                                         qdq_mul_in,      //
                                                         qdq_mul_out,     //
                                                         is_qkt_smv_int16 //
      );
  auto node_arg_qdeq_params_name =
      std::string(node_arg_get_name(*(*binder_)[output_pat_id].node_arg)) +
      "_mha_np_qdq";
  auto node_arg_qdq_params_shape = std::vector<int64_t>{6, 16};
  auto& node_arg_qdq_params = vaip::dd::insert_named_tensor_in_graph<int32_t>(
      graph_, node_arg_qdeq_params_name, qdq_params, node_arg_qdq_params_shape);
  return node_arg_qdq_params;
}

const NodeArg& DdMergeQmhaProcessor::process_m7h4xjg(int output_pat_id) {
  auto QKT_input_1_scale =
      node_arg_get_const_data_as_float("QKT_input_qparams", 0);
  auto QKT_input_1_zp = node_arg_get_const_data_as_u16("QKT_input_qparams", 1);

  auto QKT_input_2_scale =
      node_arg_get_const_data_as_float("QKT_input_qparams", 2);

  auto QKT_input_2_zp = node_arg_get_const_data_as_u16("QKT_input_qparams", 3);

  auto QKT_output_scale =
      node_arg_get_const_data_as_float("QKT_output_qparams", 0);
  auto QKT_output_zp = node_arg_get_const_data_as_u16("QKT_output_qparams", 1);

  auto VSQKT_input_1_scale =
      node_arg_get_const_data_as_float("VSQKT_input_qparams", 0);
  auto VSQKT_input_1_zp =
      node_arg_get_const_data_as_u16("VSQKT_input_qparams", 1);

  auto VSQKT_input_2_scale =
      node_arg_get_const_data_as_float("VSQKT_input_qparams", 2);
  auto VSQKT_input_2_zp =
      node_arg_get_const_data_as_u16("VSQKT_input_qparams", 3);

  auto VSQKT_output_scale =
      node_arg_get_const_data_as_float("VSQKT_output_qparams", 0);
  auto VSQKT_output_zp =
      node_arg_get_const_data_as_u16("VSQKT_output_qparams", 1);
  auto QKT_k_dim = get_k_dim(get_node_input("MATMUL_input", 0),
                             get_node_input("MATMUL_input", 1));
  auto VSQKT_k_dim = get_k_dim(get_node_input("VSMATMUL_input", 0),
                               get_node_input("VSMATMUL_input", 1));

  MY_LOG(1) << "QKT_input_1_scale " << QKT_input_1_scale << " "     //
            << "QKT_input_1_zp " << QKT_input_1_zp << " "           //
            << "QKT_input_2_scale " << QKT_input_2_scale << " "     //
            << "QKT_input_2_zp " << QKT_input_2_zp << " "           //
            << "QKT_output_scale " << QKT_output_scale << " "       //
            << "QKT_output_zp " << QKT_output_zp << " "             //
            << "VSQKT_input_1_scale " << VSQKT_input_1_scale << " " //
            << "VSQKT_input_1_zp " << VSQKT_input_1_zp << " "       //
            << "VSQKT_input_2_scale " << VSQKT_input_2_scale << " " //
            << "VSQKT_input_2_zp " << VSQKT_input_2_zp << " "       //
            << "VSQKT_output_scale " << VSQKT_output_scale << " "   //
            << "VSQKT_output_zp " << VSQKT_output_zp << " "         //
            << "QKT_k_dim " << QKT_k_dim << " "                     //
            << "VSQKT_k_dim " << VSQKT_k_dim << " "                 //
      ;

  auto qdq_sm_in = vaip::dd::qmatmulcalc::calc_lrn_coeff(
      QKT_output_scale * 1.442695041f, QKT_output_zp);

  auto softmax_output_scale =
      node_arg_get_const_data_as_float("softmax_output_qparams", 0);
  auto softmax_output_zp =
      node_arg_get_const_data_as_u16("softmax_output_qparams", 1);

  MY_LOG(1) << "softmax_output_scale " << softmax_output_scale << " " //
            << "softmax_output_zp " << softmax_output_zp << " "       //
      ;
  auto qdq_sm_out = vaip::dd::qmatmulcalc::calc_lrn_coeff(
      1.0f / softmax_output_scale, softmax_output_zp);

  MY_LOG(1) << "qdq_sm_in (" << std::get<0>(qdq_sm_in) << ","
            << std::get<1>(qdq_sm_in) << ") "  //
            << "qdq_sm_in (" << std::get<0>(qdq_sm_out) << ","
            << std::get<1>(qdq_sm_out) << ") " //
      ;
  auto MUL_weight_value =
      (float)node_arg_get_const_data_as_u16("MUL_weight_qparams", 0);
  auto MUL_weight_scale =
      node_arg_get_const_data_as_float("MUL_weight_qparams", 1);
  auto MUL_weight_zp =
      (float)node_arg_get_const_data_as_u16("MUL_weight_qparams", 2);
  MY_LOG(1) << "MUL_weight_value " << MUL_weight_value << " " //
            << "MUL_weight_scale " << MUL_weight_scale << " " //
            << "MUL_weight_zp " << MUL_weight_zp << " "       //
      ;
  auto mul_float_value =
      (MUL_weight_value - MUL_weight_zp) * (MUL_weight_scale);
  MY_LOG(1) << "mul_float_value " << mul_float_value << " " //
      ;

  auto MUL_input_scale =
      node_arg_get_const_data_as_float("MUL_input_qparams", 0);
  auto MUL_input_zp = node_arg_get_const_data_as_u16("MUL_input_qparams", 1);
  MY_LOG(1) << "MUL_input_scale " << MUL_input_scale << " " //
            << "MUL_input_zp " << MUL_input_zp << " "       //
      ;

  std::tuple<uint16_t, uint16_t> qdq_mul_in = {0, 0};

  MY_LOG(1) << "qdq_mul_in " << std::get<0>(qdq_mul_in) << ","
            << std::get<1>(qdq_mul_in) << ")"
            << " " //
      ;

  auto MUL_output_scale =
      node_arg_get_const_data_as_float("MUL_output_qparams", 0);
  auto MUL_output_zp = node_arg_get_const_data_as_u16("MUL_output_qparams", 1);
  MY_LOG(1) << "MUL_output_scale " << MUL_output_scale << " " //
            << "MUL_output_zp " << MUL_output_zp << " "       //
      ;
  std::tuple<uint16_t, uint16_t> qdq_mul_out = {0, 0};
  MY_LOG(1) << "qdq_mul_out (" << std::get<0>(qdq_mul_out) << ","
            << std::get<1>(qdq_mul_out) << ")"
            << " " //
      ;

  auto is_qkt_smv_int16 = true;

  auto coeff_qkt = vaip::dd::qmatmulcalc::qdq_act_matmul_uint16_uint16_cstm(
      MUL_input_scale * mul_float_value, MUL_input_zp, QKT_k_dim,
      QKT_input_2_scale, QKT_input_2_zp, QKT_output_scale, QKT_output_zp);
  auto coeff_smv = vaip::dd::qmatmulcalc::qdq_act_matmul_uint16_uint16_cstm(
      VSQKT_input_1_scale, VSQKT_input_1_zp, VSQKT_k_dim, VSQKT_input_2_scale,
      VSQKT_input_2_zp, VSQKT_output_scale, VSQKT_output_zp);

  MY_LOG(1) << "coeff_qkt "                         //
            << " " << std::get<0>(coeff_qkt) << " " //
            << " " << std::get<1>(coeff_qkt) << " " //
            << " " << std::get<2>(coeff_qkt) << " " //
            << " " << std::get<3>(coeff_qkt) << " " //
            << " " << std::get<4>(coeff_qkt) << " " //
            << " " << std::get<5>(coeff_qkt) << " " //
      ;
  MY_LOG(1) << "coeff_smv "                         //
            << " " << std::get<0>(coeff_smv) << " " //
            << " " << std::get<1>(coeff_smv) << " " //
            << " " << std::get<2>(coeff_smv) << " " //
            << " " << std::get<3>(coeff_smv) << " " //
            << " " << std::get<4>(coeff_smv) << " " //
            << " " << std::get<5>(coeff_smv) << " " //
      ;

  // TODO: return type?
  auto qdq_params =
      vaip::dd::qmatmulcalc::mha_channel_qdq_params_fill(coeff_qkt,        //
                                                         coeff_smv,        //
                                                         qdq_sm_in,        //
                                                         qdq_sm_out,       //
                                                         qdq_mul_in,       //
                                                         qdq_mul_out,      //
                                                         is_qkt_smv_int16, //
                                                         1);
  auto node_arg_qdeq_params_name =
      std::string(node_arg_get_name(*(*binder_)[output_pat_id].node_arg)) +
      "_mha_np_qdq";
  auto node_arg_qdq_params_shape = std::vector<int64_t>{6, 16};
  auto& node_arg_qdq_params = vaip::dd::insert_named_tensor_in_graph<int32_t>(
      graph_, node_arg_qdeq_params_name, qdq_params, node_arg_qdq_params_shape);
  return node_arg_qdq_params;
}

} // namespace vaip_dd_merge_qma
