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

#include <utility> // for std::pair
DEF_ENV_PARAM(DEBUG_DD_PATTERN, "1")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_DD_PATTERN) >= n)
// #include "calc_coeffs.hpp"
using namespace vaip_core;

struct MergeQLayerNorm_1 {
  MergeQLayerNorm_1(IPass& self) : self_{self} {}
  static bool check_conv_matmuladd_or_pool_in_parent(const Node* in_0) {
    if (in_0 && (node_is_op(*in_0, "IConv", "com.xilinx") ||
                 node_is_op(*in_0, "QGlobalAvgPool", "com.xilinx") ||
                 node_is_op(*in_0, "QMatMulAdd", "com.xilinx"))) {
      return true;
    }
    return false;
  }

  static std::unique_ptr<Rule> create_rule(IPass* self) {

    auto q2 = vaip::pattern_zoo::get_pattern("m_qlayernorm_1");
    CHECK(q2 != nullptr) << "Pattern returned is null";

    return Rule::create_rule(
        q2, [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
          // Get nodes
          std::vector<std::string> attr_nodes;
          for (auto& ni : binder) {
            if (!(*node_arg_is_constant)(*graph, *ni.second.node_arg)) {
              attr_nodes.push_back(node_arg_get_name(*ni.second.node_arg));
            }
          }
          auto a_node = binder["a"];
          auto b_node = binder["b"];
          auto c_node = binder["c"];
          auto q2_node = binder["q2"];
          auto a_s_node = binder["a_s"];
          auto a_z_node = binder["a_z"];
          auto b_s_node = binder["b_s"];
          auto b_z_node = binder["b_z"];
          auto c_s_node = binder["c_s"];
          auto c_z_node = binder["c_z"];
          auto y_s_node = binder["y_s"];
          auto y_z_node = binder["y_z"];

          //   auto dq1_node = binder["dq1"];

          //   std::cout<< "----IN LRN1 : " <<
          //   node_arg_as_string(*b_node.node_arg)<<std::endl;

          float y_sc =
              node_arg_get_const_data_as_float(*graph, *y_s_node.node_arg);
          uint16_t y_zp =
              vaip::dd::get_zp_from_node(*graph, *y_z_node.node_arg);
          // node_arg_get_const_data_as_u8(*graph, *y_z_node.node_arg);

          // qdq1
          auto alpha_shape = node_arg_get_shape_i64(*b_node.node_arg);
          auto alpha_data =
              node_arg_get_const_data_as_u8s(*graph, *b_node.node_arg);
          std::vector<uint8_t> alpha;
          for (auto e : alpha_data)
            alpha.push_back(
                e); // node_arg_get_const_data_as_u8s(*graph, *b_node.node_arg);
          auto alpha_sc =
              node_arg_get_const_data_as_floats(*graph, *b_s_node.node_arg)[0];
          auto alpha_zp = static_cast<uint8_t>(
              vaip::dd::get_zp_from_node(*graph, *b_z_node.node_arg));
          auto qdq1 =
              vaip::dd::qmatmulcalc::dq_vec_to_bf16(alpha, alpha_sc, alpha_zp);

          std::string qdq1_initializer_name =
              node_arg_get_name(*y_z_node.node_arg) +
              "_qdq1_"; // TODO: this name is based on last q node after lrn
                        // because in mxgan lrn2 and lrn pattern are matching to
                        // same lrn node of original graph due to replication
                        // done while fusion.
          const std::vector<int64_t> qdq1_initializer_shape = {
              (int64_t)qdq1.size()};

          auto& qdq1_arg = vaip::dd::insert_named_tensor_in_graph<uint16_t>(
              graph, qdq1_initializer_name, qdq1, qdq1_initializer_shape);

          // qdq2
          // params extraction
          auto beta_shape = node_arg_get_shape_i64(*c_node.node_arg);
          auto beta_data =
              node_arg_get_const_data_as_i32s(*graph, *c_node.node_arg);
          std::vector<int32_t> beta;
          for (auto e : beta_data)
            beta.push_back(
                e); // node_arg_get_const_data_as_u8s(*graph, *b_node.node_arg);

          auto beta_sc =
              node_arg_get_const_data_as_floats(*graph, *c_s_node.node_arg)[0];

          auto beta_zp =
              node_arg_get_const_data_as_i32(*graph, *c_z_node.node_arg);

          // calculation
          auto qdq2 =
              vaip::dd::qmatmulcalc::dq_vec_to_bf16(beta, beta_sc, beta_zp);

          // initialization
          std::string qdq2_initializer_name =
              node_arg_get_name(*y_z_node.node_arg) +
              "_qdq2_"; // TODO: remove tmp
          const std::vector<int64_t> qdq2_initializer_shape = {
              (int64_t)qdq1.size()};
          auto& qdq2_arg = vaip::dd::insert_named_tensor_in_graph<uint16_t>(
              graph, qdq2_initializer_name, qdq2, qdq2_initializer_shape);

          // qdq_3
          auto act_sc =
              node_arg_get_const_data_as_floats(*graph, *a_s_node.node_arg)[0];
          auto act_zp = vaip::dd::get_zp_from_node(*graph, *a_z_node.node_arg);
          auto in_dtype = node_arg_get_element_type(*a_node.node_arg);
          auto lrn_is_uint16 =
              check_conv_matmuladd_or_pool_in_parent(a_node.node);
          bool act_dtype =
              (in_dtype == (int)ONNX_NAMESPACE::TensorProto_DataType_UINT8)
                  ? true
                  : false;
          std::vector<int32_t> qdq3(16, 0);
          qdq3[0] = vaip::dd::qmatmulcalc::float_to_bfloat16(1 / y_sc);
          qdq3[1] = y_zp;
          qdq3[2] = act_dtype ? 0 : 1;
          qdq3[3] = vaip::dd::qmatmulcalc::float_to_bfloat16(act_sc);
          qdq3[4] = lrn_is_uint16 ? (int32_t)act_zp
                                  : 0; // TODO this might fail other models
          qdq3[5] = lrn_is_uint16;
          std::string qdq3_initializer_name =
              node_arg_get_name(*y_z_node.node_arg) + "_qdq3_";
          const std::vector<int64_t> qdq3_initializer_shape = {
              (int64_t)qdq3.size()};

          auto& qdq3_arg = vaip::dd::insert_named_tensor_in_graph<int32_t>(
              graph, qdq3_initializer_name, qdq3, qdq3_initializer_shape);
          std::vector<float> input_q_params;
          input_q_params.push_back(
              node_arg_get_const_data_as_float(*graph, *a_s_node.node_arg));
          input_q_params.push_back(
              float(vaip::dd::get_zp_from_node(*graph, *a_z_node.node_arg)));

          std::vector<float> output_q_params;
          output_q_params.push_back(
              node_arg_get_const_data_as_float(*graph, *y_s_node.node_arg));
          output_q_params.push_back(
              float(vaip::dd::get_zp_from_node(*graph, *y_z_node.node_arg)));

          MY_LOG(1) << "- QLayerNorm: Matched variant 1 " << attr_nodes.size()
                    << std::endl;
          // FIXME Hacky way other conditions will arise here
          std::string input_dtype = lrn_is_uint16 ? "uint16" : "bfloat16";
          std::vector<std::string> in_dtypes = {input_dtype, "uint16", "uint16",
                                                "int32"};
          std::vector<std::string> out_dtypes = {
              vaip::dd::nodearg_dtype_to_string(*q2_node.node_arg)};
          NodeBuilder(*graph, *self)
              .set_input_node_args(
                  {a_node.node_arg, &qdq1_arg, &qdq2_arg, &qdq3_arg})
              .set_op_type("QLayerNorm", "com.xilinx")
              .set_anchor_point1(*q2_node.node)
              .add("input_q_params", input_q_params)
              .add("output_q_params", output_q_params)
              .add("nodes", attr_nodes)
              .add("in_dtypes", in_dtypes)
              .add("out_dtypes", out_dtypes)
              .build();
          return true;
        });
  }

  void process(IPass& self, Graph& graph) { create_rule(&self)->apply(&graph); }

public:
  IPass& self_;
};

DEFINE_VAIP_PASS(MergeQLayerNorm_1, vaip_pass_dd_merge_qlayernorm_1)