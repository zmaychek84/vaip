/*
 *     The Xilinx Vitis AI Vaip in this distribution are provided under the
 * following free and permissive binary-only license, but are not provided in
 * source code form.  While the following free and permissive license is similar
 * to the BSD open source license, it is NOT the BSD open source license nor
 * other OSI-approved open source license.
 *
# Copyright (C) 2022 Xilinx, Inc.
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

#include "vaip/pattern_zoo.hpp"
#include "vaip/vaip.hpp"

//
#include "./qmhagrpb_processor.hpp"
#include "vaip/dd/coeffs.hpp"
#include "vaip/dd/dd_utils.hpp"
#include "vitis/ai/env_config.hpp"
DEF_ENV_PARAM(DEBUG_DD_MERGE_QMHAGRPB, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_DD_MERGE_QMHAGRPB) >= n)

#include <glog/logging.h>
#include <numeric>

namespace {
using namespace vaip_core;
struct MergeQMHAGRPB_2 {
  MergeQMHAGRPB_2(IPass& self) : self_{self} {}
  ////////////////// Pattern includes Input DQLs
  static std::unique_ptr<Rule> create_rule(IPass* self) {
    auto com_microsoft_QuantizeLinear_25 =
        vaip::pattern_zoo::get_pattern("m_qmhagrpb_mxpzi");
    CHECK(com_microsoft_QuantizeLinear_25 != nullptr)
        << "Pattern returned is null";
    return Rule::create_rule(
        com_microsoft_QuantizeLinear_25,
        [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
          std::vector<std::string> attr_nodes;
          for (auto& ni : binder) {
            if (!(*node_arg_is_constant)(*graph, *ni.second.node_arg)) {
              attr_nodes.push_back(node_arg_get_name(*ni.second.node_arg));
            }
          }

          auto in_node0 = binder["/lang_encoder/encoder/layer.0/attention/self/"
                                 "query/Add_output_0_QuantizeLinear_Output"];
          auto in_node1 = binder["/lang_encoder/encoder/layer.0/attention/self/"
                                 "key/MatMul_output_0_QuantizeLinear_Output"];
          auto in_node2 =
              binder["/lang_encoder/Mul_output_0_QuantizeLinear_Output"];
          auto in_node3 = binder["/lang_encoder/encoder/layer.0/attention/self/"
                                 "value/Add_output_0_QuantizeLinear_Output"];
          //   auto grpb_w_node = binder[constant_14->get_id()];
          auto mhagrpb = binder["/lang_encoder/encoder/layer.0/attention/self/"
                                "Reshape_4_output_0_QuantizeLinear_Output"];

          // Get nodes
          MY_LOG(1) << "- QMHAGRPB: matched " << node_as_string(*mhagrpb.node);
          auto binder_params = std::unordered_map<std::string, std::string>{
              {"query_sc", "/lang_encoder/encoder/layer.0/attention/self/query/"
                           "Add_output_0_scale"},
              {"query_zp", "/lang_encoder/encoder/layer.0/attention/self/query/"
                           "Add_output_0_zero_point"},

              {"key_sc", "/lang_encoder/encoder/layer.0/attention/self/key/"
                         "MatMul_output_0_scale"},
              {"key_zp", "/lang_encoder/encoder/layer.0/attention/self/key/"
                         "MatMul_output_0_zero_point"},

              {"v_sc", "/lang_encoder/encoder/layer.0/attention/self/value/"
                       "Add_output_0_scale"},
              {"v_zp", "/lang_encoder/encoder/layer.0/attention/self/value/"
                       "Add_output_0_zero_point"},

              {"qkt_sc", "/lang_encoder/encoder/layer.0/attention/self/"
                         "MatMul_output_0_scale"},
              {"qkt_zp", "/lang_encoder/encoder/layer.0/attention/self/"
                         "MatMul_output_0_zero_point"},

              {"sm_sc", "/lang_encoder/encoder/layer.9/attention/self/"
                        "Softmax_output_0_scale"},
              {"sm_zp", "/lang_encoder/encoder/layer.0/attention/self/"
                        "Softmax_output_0_zero_point"},

              {"vsm_sc", "/lang_encoder/encoder/layer.0/attention/self/"
                         "MatMul_1_output_0_scale"},
              {"vsm_zp", "/lang_encoder/encoder/layer.0/attention/self/"
                         "MatMul_1_output_0_zero_point"},

              {"grpb_w", "onnx::MatMul_2457_quantized"},
              {"grpb_w_sc", "onnx::MatMul_2457_scale"},
              {"grpb_w_zp", "onnx::MatMul_2457_zero_point"},

              {"grpb_b", "lang_encoder.encoder.layer.0.attention.self.gate_ur_"
                         "linear.bias_quantized"},
              {"grpb_b_sc", "lang_encoder.encoder.layer.0.attention.self.gate_"
                            "ur_linear.bias_scale"},
              {"grpb_b_zp", "lang_encoder.encoder.layer.0.attention.self.gate_"
                            "ur_linear.bias_zero_point"},

              {"grpb_sc", "/lang_encoder/encoder/layer.0/attention/self/"
                          "gate_ur_linear/Add_output_0_scale"},
              {"grpb_zp", "/lang_encoder/encoder/layer.0/attention/self/"
                          "gate_ur_linear/Add_output_0_zero_point"},

              {"div_w", "/lang_encoder/Constant_17_output_0_quantized"},
              {"div_w_sc", "/lang_encoder/Constant_17_output_0_scale"},
              {"div_w_zp", "/lang_encoder/Constant_17_output_0_zero_point"},

              {"mul_1_w",
               "lang_encoder.encoder.layer.0.attention.self.eco_a_quantized"},
              {"mul_1_w_sc",
               "lang_encoder.encoder.layer.0.attention.self.eco_a_scale"},
              {"mul_1_w_zp",
               "lang_encoder.encoder.layer.0.attention.self.eco_a_zero_point"},

              {"mul_3_w", "/lang_encoder/GatherElements_output_0_quantized"},
              {"mul_3_w_sc", "/lang_encoder/GatherElements_output_0_scale"},
              {"mul_3_w_zp",
               "/lang_encoder/GatherElements_output_0_zero_point"},

              {"add_w", "/lang_encoder/embeddings/LayerNorm/"
                        "Constant_output_0_quantized"},
              {"add_w_sc",
               "/lang_encoder/embeddings/LayerNorm/Constant_output_0_scale"},
              {"add_w_zp", "/lang_encoder/embeddings/LayerNorm/"
                           "Constant_output_0_zero_point"},

              {"sub_w", "/lang_encoder/Constant_3_output_0_quantized"},
              {"sub_w_sc", "/lang_encoder/encoder/layer.9/attention/self/"
                           "Softmax_output_0_scale"},
              {"sub_w_zp", "/lang_encoder/Constant_3_output_0_zero_point"},

              {"out", "/lang_encoder/encoder/layer.0/attention/self/"
                      "Reshape_4_output_0_QuantizeLinear_Output"},

              {"out_zp", "/lang_encoder/encoder/layer.0/attention/self/"
                         "MatMul_1_output_0_zero_point"}

          };

          auto processor = std::make_unique<
              vaip_dd_merge_qmhagrpb::DdMergeQmhagrpbProcessor>(
              *self, graph, &binder, binder_params);
          std::vector<NodeArg*> node_args =
              processor->process(com_microsoft_QuantizeLinear_25->get_id());

          const NodeArg* c4_arg = (node_args[0]);
          const NodeArg* c5_arg = (node_args[1]);
          const NodeArg* c6_arg = (node_args[2]);
          const NodeArg* c7_arg = (node_args[3]);
          const NodeArg* c8_arg = (node_args[4]);

          std::vector<float> output_q_params;

          auto outsc1_node = binder["/lang_encoder/encoder/layer.0/attention/"
                                    "self/MatMul_1_output_0_scale"];
          auto outzp1_node = binder["/lang_encoder/encoder/layer.0/attention/"
                                    "self/MatMul_1_output_0_zero_point"];
          output_q_params.push_back(
              node_arg_get_const_data_as_float(*graph, *outsc1_node.node_arg));
          output_q_params.push_back(
              float(vaip::dd::get_zp_from_node(*graph, *outzp1_node.node_arg)));

          std::vector<float> input_q_params;

          auto sc0_node = binder["/lang_encoder/encoder/layer.0/attention/self/"
                                 "query/Add_output_0_scale"];
          auto zp0_node = binder["/lang_encoder/encoder/layer.0/attention/self/"
                                 "query/Add_output_0_zero_point"];
          input_q_params.push_back(
              node_arg_get_const_data_as_float(*graph, *sc0_node.node_arg));
          input_q_params.push_back(
              float(vaip::dd::get_zp_from_node(*graph, *zp0_node.node_arg)));

          auto sc1_node = binder["/lang_encoder/encoder/layer.0/attention/self/"
                                 "key/MatMul_output_0_scale"];
          auto zp1_node = binder["/lang_encoder/encoder/layer.0/attention/self/"
                                 "key/MatMul_output_0_zero_point"];
          input_q_params.push_back(
              node_arg_get_const_data_as_float(*graph, *sc1_node.node_arg));
          input_q_params.push_back(
              float(vaip::dd::get_zp_from_node(*graph, *zp1_node.node_arg)));

          auto sc3_node = binder["/lang_encoder/encoder/layer.0/attention/self/"
                                 "value/Add_output_0_scale"];
          auto zp3_node = binder["/lang_encoder/encoder/layer.0/attention/self/"
                                 "value/Add_output_0_zero_point"];
          input_q_params.push_back(
              node_arg_get_const_data_as_float(*graph, *sc3_node.node_arg));
          input_q_params.push_back(
              float(vaip::dd::get_zp_from_node(*graph, *zp3_node.node_arg)));

          auto sc2_node = binder["/lang_encoder/Constant_4_output_0_scale"];
          auto zp2_node = binder["/lang_encoder/Mul_output_0_zero_point"];
          input_q_params.push_back(
              node_arg_get_const_data_as_float(*graph, *sc2_node.node_arg));
          input_q_params.push_back(
              float(vaip::dd::get_zp_from_node(*graph, *zp2_node.node_arg)));

          auto mhagrpb_builder = NodeBuilder(*graph, *self);
          mhagrpb_builder.set_input_node_args(
              {in_node0.node_arg, in_node1.node_arg, in_node3.node_arg,
               in_node2.node_arg, c4_arg, c5_arg, c6_arg, c7_arg, c8_arg});

          std::vector<std::string> in_dtypes = {"uint16",   "uint16", "uint16",
                                                "bfloat16", "uint8",  "int64",
                                                "int32",    "uint16", "int32"};
          std::vector<std::string> out_dtypes = {"uint16"};
          mhagrpb_builder.set_op_type("QMHAGRPB", "com.xilinx");
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

DEFINE_VAIP_PASS(MergeQMHAGRPB_2, vaip_pass_dd_merge_qmhagrpb_mxpzi)
