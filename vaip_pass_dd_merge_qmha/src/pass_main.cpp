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
#include "qmha/qmha_processor.hpp"
#include "vaip/dd/dd_utils.hpp"
#include "vaip/pattern_zoo.hpp"
#include "vaip/vaip.hpp"
#include "vitis/ai/env_config.hpp"
#include <glog/logging.h>
DEF_ENV_PARAM(DEBUG_DD_MERGE_QMHA, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_DD_MERGE_QMHA) >= n)

/**
 * test case: m7h4xjg 4e485de54588d95209560c0a29049b68
 *
 *
 * Replace pattern:
 *
 * From: QMHA
 * To  : QMHA node
 */

// add the following line in your vaip_config.json
/*
    { "name": "vaip_pass_dd_merge_qmha",
       "plugin": "vaip-pass_dd_merge_qmha",
       "disabled": false
    }
*/
namespace {
using namespace vaip_core;
struct DdMergeQmha {
  DdMergeQmha(IPass& self) : self_{self} {}
  std::unique_ptr<Rule> create_rule(IPass* self) {
    auto com_microsoft_QuantizeLinear_22 =
        vaip::pattern_zoo::get_pattern("m_qmha_0");
    CHECK(com_microsoft_QuantizeLinear_22 != nullptr)
        << "Pattern returned is null";
    return Rule::create_rule(
        com_microsoft_QuantizeLinear_22,
        [=](onnxruntime::Graph* graph_ptr, binder_t& binder) -> bool {
          graph_ = graph_ptr;
          binder_ = &binder;
          auto ni_input_v = binder["/text_model/encoder/layers.0/self_attn/"
                                   "v_proj/Add_output_0_QuantizeLinear_Output"];
          auto ni_input_q = binder["/text_model/encoder/layers.0/self_attn/"
                                   "q_proj/Add_output_0_QuantizeLinear_Output"];
          auto ni_input_k = binder["/text_model/encoder/layers.0/self_attn/"
                                   "k_proj/Add_output_0_QuantizeLinear_Output"];
          auto ni_output = binder["/text_model/encoder/layers.0/self_attn/"
                                  "Reshape_9_output_0_QuantizeLinear_Output"];

          std::vector<float> output_q_params;

          auto outsc1_node = binder["/text_model/encoder/layers.0/self_attn/"
                                    "softmax/Softmax_output_0_scale"];
          auto outzp1_node = binder["/text_model/encoder/layers.0/self_attn/"
                                    "softmax/Softmax_output_0_zero_point"];
          output_q_params.push_back(node_arg_get_const_data_as_float(
              *graph_ptr, *outsc1_node.node_arg));
          output_q_params.push_back(float(
              vaip::dd::get_zp_from_node(*graph_ptr, *outzp1_node.node_arg)));

          std::vector<float> input_q_params;

          auto sc0_node = binder["/text_model/encoder/layers.0/self_attn/"
                                 "q_proj/Add_output_0_scale"];
          auto zp0_node = binder["/text_model/encoder/layers.0/self_attn/"
                                 "q_proj/Add_output_0_zero_point"];
          input_q_params.push_back(
              node_arg_get_const_data_as_float(*graph_ptr, *sc0_node.node_arg));
          input_q_params.push_back(float(
              vaip::dd::get_zp_from_node(*graph_ptr, *zp0_node.node_arg)));

          auto sc1_node = binder["/text_model/encoder/layers.0/self_attn/"
                                 "k_proj/Add_output_0_scale"];
          auto zp1_node = binder["/text_model/encoder/layers.0/self_attn/"
                                 "k_proj/Add_output_0_zero_point"];
          input_q_params.push_back(
              node_arg_get_const_data_as_float(*graph_ptr, *sc1_node.node_arg));
          input_q_params.push_back(float(
              vaip::dd::get_zp_from_node(*graph_ptr, *zp1_node.node_arg)));

          auto sc2_node = binder["/text_model/encoder/layers.0/self_attn/"
                                 "v_proj/Add_output_0_scale"];
          auto zp2_node = binder["/text_model/encoder/layers.0/self_attn/"
                                 "v_proj/Add_output_0_zero_point"];
          input_q_params.push_back(
              node_arg_get_const_data_as_float(*graph_ptr, *sc2_node.node_arg));
          input_q_params.push_back(float(
              vaip::dd::get_zp_from_node(*graph_ptr, *zp2_node.node_arg)));

          MY_LOG(1) << "found node in QMHA:" << node_as_string(*ni_output.node);
          auto const binder_params = std::unordered_map<
              std::string, std::vector<std::string>>{
              // clang-format off
                {"QKT_input_qparams",{
        "/text_model/encoder/layers.0/self_attn/Mul_output_0_scale",
        "/text_model/encoder/layers.0/self_attn/Mul_output_0_zero_point",
        "/text_model/encoder/layers.0/self_attn/k_proj/Add_output_0_scale",
        "/text_model/encoder/layers.0/self_attn/k_proj/Add_output_0_zero_point",
                    }},
                {"QKT_output_qparams",{
        "/text_model/encoder/layers.0/self_attn/bmm_1/MatMul_output_0_scale",
        "/text_model/encoder/layers.0/self_attn/bmm_1/MatMul_output_0_zero_point",
                    }},
                {"MATMUL_input", {"/text_model/encoder/layers.0/self_attn/Reshape_3_output_0_DequantizeLinear_Output",
                                  "/text_model/encoder/layers.0/self_attn/Transpose_3_output_0_DequantizeLinear_Output",
                    }},
                {"VSQKT_input_qparams",{
        "/text_model/encoder/layers.0/self_attn/v_proj/Add_output_0_scale",
        "/text_model/encoder/layers.0/self_attn/v_proj/Add_output_0_zero_point",
        "/text_model/encoder/layers.0/self_attn/softmax/Softmax_output_0_scale",
        "/text_model/encoder/layers.0/self_attn/softmax/Softmax_output_0_zero_point",
                    }},
                {"VSQKT_output_qparams",{
        "/text_model/encoder/layers.0/self_attn/bmm_2/MatMul_output_0_scale",
        "/text_model/encoder/layers.0/self_attn/bmm_2/MatMul_output_0_zero_point",
                    }},
                {"VSMATMUL_input", {
                        "/text_model/encoder/layers.0/self_attn/Transpose_4_output_0_DequantizeLinear_Output",
                        "/text_model/encoder/layers.0/self_attn/Transpose_5_output_0_DequantizeLinear_Output"
                    }},
                {"softmax_input_qparams",{
        "/text_model/encoder/layers.0/self_attn/Add_output_0_scale",
        "/text_model/encoder/layers.0/self_attn/Add_output_0_zero_point",
                    }},
                {"softmax_output_qparams",{
        "/text_model/encoder/layers.0/self_attn/softmax/Softmax_output_0_scale",
        "/text_model/encoder/layers.0/self_attn/softmax/Softmax_output_0_zero_point",
                    }},
                {"MUL_input_qparams",{
        "/text_model/encoder/layers.0/self_attn/q_proj/Add_output_0_scale",
        "/text_model/encoder/layers.0/self_attn/q_proj/Add_output_0_zero_point",
                    }},
                {"MUL_weight_qparams",{
        "/text_model/encoder/layers.0/self_attn/Constant_output_0_quantized",
        "/text_model/encoder/layers.0/self_attn/Constant_output_0_scale",
        "/text_model/encoder/layers.0/self_attn/Constant_output_0_zero_point",
                    }},
                {"MUL_output_qparams",{
        "/text_model/encoder/layers.0/self_attn/Mul_output_0_scale",
        "/text_model/encoder/layers.0/self_attn/Mul_output_0_zero_point",
                    }},
              // clang-format on
          };
          auto processor =
              std::make_unique<vaip_dd_merge_qma::DdMergeQmhaProcessor>(
                  self_, graph_ptr, &binder, binder_params);
          std::vector<std::string> ns =
              vaip::dd::get_node_names(graph_ptr, binder);
          auto& node_arg_qdq_params = processor->process_m7h4xjg(
              com_microsoft_QuantizeLinear_22->get_id());

          NodeBuilder(*graph_, self_)
              .set_op_type("QMHA", "com.xilinx")
              .set_input_node_args({
                  ni_input_q.node_arg, //
                  ni_input_k.node_arg, //
                  ni_input_v.node_arg,
                  &node_arg_qdq_params //
              })
              .add("nodes", ns)
              .clone_data_type(*ni_output.node_arg)
              .clone_shape(*ni_output.node_arg)
              .add("in_dtypes", std::vector<std::string>(
                                    {"uint16", "uint16", "uint16", "int32"}))
              .add("out_dtypes", std::vector<std::string>({"uint16"}))
              .add("input_q_params", input_q_params)
              .add("output_q_params", output_q_params)
              .set_anchor_point1(*ni_output.node_arg)
              .build();
          return true; // return true if graph is modified.
        });
  }
  // apply the rule
  void process(IPass& self, Graph& graph) { create_rule(&self)->apply(&graph); }

  IPass& self_;
  onnxruntime::Graph* graph_;
  binder_t* binder_;
};
} // namespace

DEFINE_VAIP_PASS(DdMergeQmha, vaip_pass_dd_merge_qmha)
