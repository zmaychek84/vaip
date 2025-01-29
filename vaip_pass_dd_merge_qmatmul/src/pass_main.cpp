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
#include <functional>
#include <glog/logging.h>
#include <numeric>
DEF_ENV_PARAM(DEBUG_DD_PATTERN, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_DD_PATTERN) >= n)
using namespace vaip_core;

struct MergeQMatMul {
  MergeQMatMul(IPass& self) : self_{self} {}

  static std::vector<std::string> change_inputs(const NodeArg& a,
                                                const NodeArg& b) {
    std::vector<std::string> dtypes;
    // Add conditional code here (Below may only work for mdsqr)
    dtypes.emplace_back(vaip::dd::nodearg_dtype_to_string(a));
    dtypes.emplace_back("uint8");
    dtypes.emplace_back("int64");
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
    auto q = vaip::pattern_zoo::get_pattern("m_qmatmul_0");
    CHECK(q != nullptr) << "Pattern returned is null";
    return Rule::create_rule(
        q, [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
          // Activation Node & Data
          auto a_node = binder["a"];
          auto as_node = binder["a_s"];
          auto az_node = binder["a_z"];
          auto a_sc =
              node_arg_get_const_data_as_float(*graph, *as_node.node_arg);
          auto a_zp = vaip::dd::get_zp_from_node(*graph, *az_node.node_arg);
          auto a_shape = node_arg_get_shape_i64(*a_node.node_arg);
          // Weights Node and Data
          auto w_node = binder["w"];
          auto ws_node = binder["w_s"];
          auto wz_node = binder["w_z"];
          auto w_shape = node_arg_get_shape_i64(*w_node.node_arg);
          auto w_data =
              node_arg_get_const_data_as_u8s(*graph, *w_node.node_arg);
          auto w_sc =
              node_arg_get_const_data_as_float(*graph, *ws_node.node_arg);
          auto w_zp = vaip::dd::get_zp_from_node(*graph, *wz_node.node_arg);

          if (nullptr != w_shape)
            MY_LOG(1) << "PQR "
                      << vaip::dd::shape_as_string(
                             *(w_shape.get())); //.c_str();
          else
            LOG(FATAL) << "Shape could not be determined";

          auto q_node = binder["q"];
          auto qs_node = binder["q_s"];
          auto qz_node = binder["q_z"];
          auto q_sc =
              node_arg_get_const_data_as_float(*graph, *qs_node.node_arg);
          auto q_zp = vaip::dd::get_zp_from_node(*graph, *qz_node.node_arg);
          auto q_shape = node_arg_get_shape_i64(*q_node.node_arg);
          // CHECK Data Type
          auto out_dtype = node_arg_get_element_type(*q_node.node_arg);
          MY_LOG(1) << "QMATMUL " << out_dtype;
          std::vector<std::string> ns = vaip::dd::get_node_names(graph, binder);

          auto node_name = node_arg_get_name(*q_node.node_arg);

          if ((*(w_shape.get())).size() == 2) {
            vaip::dd::qmatmulcalc::MatmulQDQParams qdq_params;
            if (out_dtype == (int)ONNX_NAMESPACE::TensorProto_DataType_UINT8) {
              qdq_params = vaip::dd::qmatmulcalc::
                  calculate_matmul_qdq_params_uint8_uint8(
                      vaip::dd::fold2D<uint8_t>(w_data, *(w_shape.get())), a_sc,
                      a_zp, w_sc, w_zp, q_sc, q_zp);
            } else if (out_dtype ==
                       (int)ONNX_NAMESPACE::TensorProto_DataType_UINT16) {
              qdq_params = vaip::dd::qmatmulcalc::
                  calculate_matmul_qdq_params_uint16_uint8(
                      vaip::dd::fold2D<uint8_t>(w_data, *(w_shape.get())), a_sc,
                      a_zp, w_sc, w_zp, q_sc, q_zp);
            } else {
              LOG(FATAL) << "Unknown Data Type";
            }

            // TODO auto activation_zp = node_arg_get_const_data_as_u8(*graph,
            // *az_node.node_arg);

            auto c0_coeffs =
                qdq_params.c0_coeffs; // get_initialized_vector(768);
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

            NodeBuilder(*graph, *self)
                .set_input_node_args(
                    {a_node.node_arg, w_node.node_arg, &c0_arg, &qdq_arg})
                .set_op_type("QMatMul", "com.xilinx")
                .clone_attrs(*q_node.node)
                .add("nodes", ns)
                .set_anchor_point1(*q_node.node)
                .add("in_dtypes",
                     change_inputs(*a_node.node_arg, *w_node.node_arg))
                .add("out_dtypes", change_outputs(*q_node.node_arg))
                .add("input_shape", *(a_shape.get()))
                .add("output_shape", *(q_shape.get()))
                .build();

          } else if ((*(w_shape.get())).size() == 3) {
            vaip::dd::qmatmulcalc::MatmulQDQParams_3d qdq_params;

            if (out_dtype == (int)ONNX_NAMESPACE::TensorProto_DataType_UINT16) {
              qdq_params = vaip::dd::qmatmulcalc::
                  calculate_matmul_3d_qdq_params_uint16_uint8(
                      vaip::dd::fold3D<uint8_t>(w_data, *(w_shape.get())), a_sc,
                      a_zp, w_sc, w_zp, q_sc, q_zp);
            } else {
              LOG(FATAL) << "Unknown Data Type";
            }

            // TODO auto activation_zp = node_arg_get_const_data_as_u8(*graph,
            // *az_node.node_arg);
            int c0_rows = qdq_params.c0_coeffs.size();
            int c0_cols = qdq_params.c0_coeffs[0].size();
            std::vector<int64_t> c0_coeffs_vec(c0_rows * c0_cols);

            for (auto m = 0; m < c0_rows; m++) {
              for (auto n = 0; n < c0_cols; n++) {
                c0_coeffs_vec[m * c0_cols + n] = qdq_params.c0_coeffs[m][n];
              }
            }
            std::string c0_initializer_name = std::string(node_name + "_c0_");
            auto& c0_arg = vaip::dd::insert_named_tensor_in_graph<int64_t>(
                graph, c0_initializer_name, c0_coeffs_vec,
                std::vector({(int64_t)c0_rows, (int64_t)c0_cols}));

            auto qdq_coeffs =
                qdq_params.qdq_params; // get_initialized_vector(768);
            std::string qdq_name = std::string(node_name + "_qdq_");
            auto& qdq_arg = vaip::dd::insert_named_tensor_in_graph<int32_t>(
                graph, qdq_name, qdq_coeffs,
                std::vector({(int64_t)qdq_coeffs.size()}));

            NodeBuilder(*graph, *self)
                .set_input_node_args(
                    {a_node.node_arg, w_node.node_arg, &c0_arg, &qdq_arg})
                .set_op_type("QBatchMatMul", "com.xilinx")
                .clone_attrs(*q_node.node)
                .add("nodes", ns)
                .set_anchor_point1(*q_node.node)
                .add("in_dtypes",
                     change_inputs(*a_node.node_arg, *w_node.node_arg))
                .add("out_dtypes", change_outputs(*q_node.node_arg))
                .add("input_shape", *(a_shape.get()))
                .add("output_shape", *(q_shape.get()))
                .build();
          }
          return true;
        });
  }

  void process(IPass& self, Graph& graph) { create_rule(&self)->apply(&graph); }

public:
  IPass& self_;
};

DEFINE_VAIP_PASS(MergeQMatMul, vaip_pass_dd_merge_qmatmul)
