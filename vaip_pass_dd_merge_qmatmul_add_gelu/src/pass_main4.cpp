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
#include "vaip/pattern_zoo.hpp"
#ifdef __GNUC__
#  pragma GCC diagnostic pop
#endif
#include "vaip/vaip.hpp"
#include "vitis/ai/env_config.hpp"
#include <filesystem>
#include <fstream>
#include <glog/logging.h>
namespace fs = std::filesystem;
DEF_ENV_PARAM(DEBUG_DD_PATTERN, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_DD_PATTERN) >= n)
using namespace vaip_core;

struct MergeQMatMulAddGelu4 {
  MergeQMatMulAddGelu4(IPass& self) : self_{self} {}

  static std::vector<std::string> change_inputs(const NodeArg& a,
                                                const NodeArg& b) {
    std::vector<std::string> dtypes;
    // Add conditional code here (Below may only work for mdsqr)
    dtypes.emplace_back(vaip::dd::nodearg_dtype_to_string(a));
    dtypes.emplace_back("uint8");
    dtypes.emplace_back("int64");
    dtypes.emplace_back("int32");
    dtypes.emplace_back("int32");
    return dtypes;
  }

  static std::vector<std::string> change_outputs(const NodeArg& a) {
    std::vector<std::string> dtypes;
    // Add conditional code here (Below may only work for mdsqr)
    dtypes.emplace_back(vaip::dd::nodearg_dtype_to_string(a));
    return dtypes;
  }

  static std::unique_ptr<Rule> create_rule(IPass* self) {

    auto q4 = vaip::pattern_zoo::get_pattern("m_gemm_gelu");
    CHECK(q4 != nullptr) << "Pattern returned is null";

    return Rule::create_rule(
        q4, [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
          // Get nodes
          auto attr_nodes = vaip::dd::get_node_names(graph, binder);

          auto a_node = binder["input_0"];
          auto as_node = binder["constant_11"];
          auto az_node = binder["constant_12"];
          auto a_sc =
              node_arg_get_const_data_as_float(*graph, *as_node.node_arg);

          auto a_zp = vaip::dd::get_zp_from_node(*graph, *az_node.node_arg);
          auto a_shape = node_arg_get_shape_i64(*a_node.node_arg);
          auto w_node = binder["constant_3"];
          auto ws_node = binder["constant_4"];
          auto wz_node = binder["constant_5"];
          auto w_shape = node_arg_get_shape_i64(*w_node.node_arg);
          auto w_data =
              node_arg_get_const_data_as_u8s(*graph, *w_node.node_arg);
          std::vector<uint8_t> wts(w_data.begin(), w_data.end());
          auto w_sc =
              node_arg_get_const_data_as_float(*graph, *ws_node.node_arg);
          auto w_zp = vaip::dd::get_zp_from_node(*graph, *wz_node.node_arg);
          // auto q1_node = binder["q1"];
          // auto q1s_node = binder["q1_s"];
          // auto q1z_node = binder["q1_z"];

          auto b_node = binder["constant_0"];
          auto bs_node = binder["constant_1"];
          auto bz_node = binder["constant_2"];
          auto b_data =
              vaip::dd::get_const_as_int32_t(*graph, *b_node.node_arg);
          auto b_shape = node_arg_get_shape_i64(*b_node.node_arg);
          auto b_sc =
              node_arg_get_const_data_as_float(*graph, *bs_node.node_arg);
          auto b_zp = vaip::dd::get_zp_from_node(*graph, *bz_node.node_arg);
          // auto q2_node = binder["q2"];
          auto q2s_node = binder["constant_13"];
          auto q2z_node = binder["constant_14"];
          auto q2_sc =
              node_arg_get_const_data_as_float(*graph, *q2s_node.node_arg);
          auto q2_zp = vaip::dd::get_zp_from_node(*graph, *q2z_node.node_arg);
          auto q3_node = binder["com_microsoft_QuantizeLinear_2"];
          auto q3s_node = binder["constant_17"];
          auto q3z_node = binder["constant_18"];
          auto q3_sc =
              node_arg_get_const_data_as_float(*graph, *q3s_node.node_arg);
          auto q3_zp = vaip::dd::get_zp_from_node(*graph, *q3z_node.node_arg);
          auto q3_shape = node_arg_get_shape_i64(*q3_node.node_arg);
          auto node_name = node_arg_get_name(*q3_node.node_arg);
          // CHECK Data Type
          auto out_dtype = node_arg_get_element_type(*q3_node.node_arg);
          auto bias_dtype = node_arg_get_element_type(*b_node.node_arg);
          // CHECK(out_dtype == (int)ONNX_NAMESPACE::TensorProto_DataType_UINT8)
          //     << "Currently only uint8 is supported";
          MY_LOG(1) << "QMATMULADDGELU VARIANT 4 " << node_name;
          auto gemm_node = binder["Gemm_0"];
          int64_t transB = node_get_attr_int(*gemm_node.node, "transB");
          auto wts_shape = *w_shape;
          std::vector<std::vector<uint8_t>> vec_wts(
              wts_shape[1], std::vector<uint8_t>(wts_shape[0]));
          for (size_t i = 0; i < wts_shape[1]; ++i) {
            for (size_t j = 0; j < wts_shape[0]; ++j) {
              if (transB) {
                vec_wts[j][i] = w_data[i * wts_shape[0] + j];
              } else {
                vec_wts[i][j] = w_data[i * wts_shape[0] + j];
              }
            }
          }
          std::vector<uint8_t> vec_w(wts_shape[0] * wts_shape[1]);
          for (size_t i = 0; i < wts_shape[1]; ++i) {
            for (size_t j = 0; j < wts_shape[0]; ++j) {
              vec_w[i * wts_shape[0] + j] = vec_wts[i][j];
            }
          }
          vaip::dd::qmatmulcalc::MatmulQDQParams qdq_params;
          if (out_dtype == (int)ONNX_NAMESPACE::TensorProto_DataType_UINT8) {
            if (bias_dtype == (int)ONNX_NAMESPACE::TensorProto_DataType_INT32) {
              qdq_params = vaip::dd::qmatmulcalc::
                  calculate_matmuladd_qdq_params_uint8_uint8_b32(
                      vaip::dd::fold2D<uint8_t>(w_data, *(w_shape.get())),
                      b_data, a_sc, a_zp, w_sc, w_zp, b_sc, b_zp, q2_sc, q2_zp);
            } else {
              // qdq_params = vaip::dd::qmatmulcalc::
              //     calculate_matmuladd_qdq_params_uint8_uint8(
              //         vaip::dd::fold2D<uint8_t>(w_data, *(w_shape.get())),
              //         b_data, a_sc, a_zp, w_sc, w_zp, b_sc, b_zp, q2_sc,
              //         q2_zp);
            }

          } else if (out_dtype ==
                     (int)ONNX_NAMESPACE::TensorProto_DataType_UINT16) {
            if (bias_dtype == (int)ONNX_NAMESPACE::TensorProto_DataType_INT32) {
              qdq_params = vaip::dd::qmatmulcalc::
                  calculate_matmuladd_qdq_params_uint16_uint8_b32(
                      vec_wts, b_data, a_sc, a_zp, w_sc, w_zp, b_sc, b_zp,
                      q2_sc, q2_zp);
            } else {
              // qdq_params = vaip::dd::qmatmulcalc::
              //     calculate_matmuladd_qdq_params_uint16_uint8(
              //         vaip::dd::fold2D<uint8_t>(w_data, *(w_shape.get())),
              //         b_data, a_sc, a_zp, w_sc, w_zp, b_sc, b_zp, q2_sc,
              //         q2_zp);
            }
          } else {
            LOG(FATAL) << "Unknown Data Type";
          }
          auto c0_coeffs = qdq_params.c0_coeffs; // get_initialized_vector(768);
          std::string c0_initializer_name = std::string(node_name + "_c0_");
          auto& c0_arg = vaip::dd::insert_named_tensor_in_graph<int64_t>(
              graph, c0_initializer_name, c0_coeffs,
              std::vector({(int64_t)c0_coeffs.size()}));

          auto qdq_coeffs =
              qdq_params.qdq_params; // get_initialized_vector(768);
          std::string qdq_name = std::string(node_name + "_qdq_");
          auto& qdq_arg = vaip::dd::insert_named_tensor_in_graph<int32_t>(
              graph, qdq_name, qdq_coeffs,
              std::vector({(int64_t)qdq_coeffs.size()}));
          auto [c0_sc_a, c0_zp_a, c0_sc_b, c0_zp_b] =
              vaip::dd::qmatmulcalc::calc_eltwise_coeff(
                  q2_sc, q2_zp, (float)1.0 / q3_sc, q3_zp);
          // GELU CALC
          std::vector<int32_t> gelu_coeffs(16, 0);
          gelu_coeffs[0] = c0_zp_a;
          gelu_coeffs[1] = c0_sc_a;
          gelu_coeffs[2] = c0_zp_b;
          gelu_coeffs[3] = c0_sc_b;
          gelu_coeffs[4] =
              out_dtype == (int)ONNX_NAMESPACE::TensorProto_DataType_UINT8 ? 0
                                                                           : 1;
          std::string gelu_coeff_name = std::string(node_name + "gelu_qdq");
          auto& gelu_arg = vaip::dd::insert_named_tensor_in_graph<int32_t>(
              graph, gelu_coeff_name, gelu_coeffs,
              std::vector({(int64_t)gelu_coeffs.size()}));

          // auto wts_shape = node_arg_get_shape_i64(*w_node.node_arg);

          std::string wts_initializer_name =
              node_arg_get_name(*w_node.node_arg) + "0";
          const std::vector<int64_t> wts_initializer_shape = {
              (int64_t)wts_shape[1], (int64_t)wts_shape[0]};
          NodeArg& wts_arg = vaip::dd::insert_named_tensor_in_graph<uint8_t>(
              graph, wts_initializer_name, vec_w, wts_initializer_shape);

          auto matmul_add_gelu = NodeBuilder(*graph, *self);
          matmul_add_gelu.set_input_node_args(
              {a_node.node_arg, &wts_arg, &c0_arg, &qdq_arg, &gelu_arg});
          matmul_add_gelu.set_op_type("QGemmvGelu", "com.xilinx");
          matmul_add_gelu.set_anchor_point1(*q3_node.node);
          matmul_add_gelu.add("nodes", attr_nodes);
          matmul_add_gelu.add(
              "in_dtypes", change_inputs(*a_node.node_arg, *w_node.node_arg));
          matmul_add_gelu.add("out_dtypes", change_outputs(*q3_node.node_arg));
          matmul_add_gelu.add("input_shape", *(a_shape.get()));
          matmul_add_gelu.add("output_shape", *(q3_shape.get()));
          matmul_add_gelu.add("design_param", "4x4");
          matmul_add_gelu.build();
          return true;
        });
  }

  void process(IPass& self, Graph& graph) { create_rule(&self)->apply(&graph); }

public:
  IPass& self_;
};

DEFINE_VAIP_PASS(MergeQMatMulAddGelu4, vaip_pass_dd_merge_qmatmul_add_gelu4)
