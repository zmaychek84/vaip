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
struct MergeQMHAGRPB {
  MergeQMHAGRPB(IPass& self) : self_{self} {}
  ////////////////// Pattern includes Input DQLs
  static std::unique_ptr<Rule> create_rule(IPass* self) {
    auto com_microsoft_QuantizeLinear_28 =
        vaip::pattern_zoo::get_pattern("m_qmhagrpb_mxgan");
    CHECK(com_microsoft_QuantizeLinear_28 != nullptr)
        << "Pattern returned is null";
    return Rule::create_rule(

        com_microsoft_QuantizeLinear_28,
        [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
          std::vector<std::string> attr_nodes;
          for (auto& ni : binder) {
            if (!(*node_arg_is_constant)(*graph, *ni.second.node_arg)) {
              attr_nodes.push_back(node_arg_get_name(*ni.second.node_arg));
            }
          }

          auto in_node0 = binder["274_QuantizeLinear_Output"];
          auto in_node1 = binder["276_QuantizeLinear_Output"];
          auto in_node2 = binder["110_convert_QuantizeLinear_Output"];
          auto in_node3 = binder["279_QuantizeLinear_Output"];
          //   auto grpb_w_node = binder[constant_14->get_id()];
          auto mhagrpb = binder["409_QuantizeLinear_Output"];
          // Get nodes
          MY_LOG(1) << "- QMHAGRPB: matched " << node_as_string(*mhagrpb.node);
          auto binder_params = std::unordered_map<std::string, std::string>{
              {"query_sc", "274_scale"},
              {"query_zp", "274_zero_point"},

              {"key_sc", "276_scale"},
              {"key_zp", "276_zero_point"},

              {"v_sc", "279_scale"},
              {"v_zp", "279_zero_point"},

              {"qkt_sc", "337_scale"},
              {"qkt_zp", "337_zero_point"},

              {"sm_sc", "106_scale"},
              {"sm_zp", "392_zero_point_convert"},

              {"vsm_sc", "393_scale"},
              {"vsm_zp", "393_zero_point"},

              {"grpb_w", "1077_quantized"},
              {"grpb_w_sc", "1077_scale"},
              {"grpb_w_zp", "1077_zero_point"},

              {"grpb_b", "roberta_encoder_src.encoder.layer.0.attention.self."
                         "gate_ur_linear.bias_quantized"},
              {"grpb_b_sc", "roberta_encoder_src.encoder.layer.0.attention."
                            "self.gate_ur_linear.bias_scale"},
              {"grpb_b_zp", "roberta_encoder_src.encoder.layer.0.attention."
                            "self.gate_ur_linear.bias_zero_point"},

              {"grpb_sc", "352_scale"},
              {"grpb_zp", "352_zero_point"},

              {"div_w", "1062_quantized"},
              {"div_w_sc", "1062_scale"},
              {"div_w_zp", "1062_zero_point"},

              {"mul_1_w", "roberta_encoder_src.encoder.layer.0.attention.self."
                          "eco_a_quantized"},
              {"mul_1_w_sc", "roberta_encoder_src.encoder.layer.0.attention."
                             "self.eco_a_scale"},
              {"mul_1_w_zp", "roberta_encoder_src.encoder.layer.0.attention."
                             "self.eco_a_zero_point"},

              {"mul_3_w", "271_quantized"},
              {"mul_3_w_sc", "271_scale"},
              {"mul_3_w_zp", "271_zero_point"},

              {"add_w", "130_quantized"},
              {"add_w_sc", "130_scale"},
              {"add_w_zp", "130_zero_point"},

              {"sub_w", "107_quantized"},
              {"sub_w_sc", "106_scale"},
              {"sub_w_zp", "107_zero_point"},

              {"out", "409_QuantizeLinear_Output"},

              {"out_zp", "393_zero_point"}};

          auto processor = std::make_unique<
              vaip_dd_merge_qmhagrpb::DdMergeQmhagrpbProcessor>(
              *self, graph, &binder, binder_params);
          std::vector<NodeArg*> node_args =
              processor->process(com_microsoft_QuantizeLinear_28->get_id());

          const NodeArg* c4_arg = node_args[0];
          const NodeArg* c5_arg = node_args[1];
          const NodeArg* c6_arg = node_args[2];
          const NodeArg* c7_arg = node_args[3];
          const NodeArg* c8_arg = node_args[4];
          // const NodeArg* c8_arg = (node_args[0]);

          std::vector<float> output_q_params;

          auto outsc1_node = binder["393_scale"];
          auto outzp1_node = binder["393_zero_point"];
          output_q_params.push_back(
              node_arg_get_const_data_as_float(*graph, *outsc1_node.node_arg));
          output_q_params.push_back(
              float(vaip::dd::get_zp_from_node(*graph, *outzp1_node.node_arg)));

          std::vector<float> input_q_params;

          auto sc0_node = binder["274_scale"];
          auto zp0_node = binder["274_zero_point"];
          input_q_params.push_back(
              node_arg_get_const_data_as_float(*graph, *sc0_node.node_arg));
          input_q_params.push_back(
              float(vaip::dd::get_zp_from_node(*graph, *zp0_node.node_arg)));

          auto sc1_node = binder["276_scale"];
          auto zp1_node = binder["276_zero_point"];
          input_q_params.push_back(
              node_arg_get_const_data_as_float(*graph, *sc1_node.node_arg));
          input_q_params.push_back(
              float(vaip::dd::get_zp_from_node(*graph, *zp1_node.node_arg)));

          auto sc3_node = binder["279_scale"];
          auto zp3_node = binder["279_zero_point"];
          input_q_params.push_back(
              node_arg_get_const_data_as_float(*graph, *sc3_node.node_arg));
          input_q_params.push_back(
              float(vaip::dd::get_zp_from_node(*graph, *zp3_node.node_arg)));

          auto sc2_node = binder["110_scale_convert"];
          auto zp2_node = binder["110_zero_point_convert"];
          input_q_params.push_back(
              node_arg_get_const_data_as_float(*graph, *sc2_node.node_arg));
          input_q_params.push_back(
              float(vaip::dd::get_zp_from_node(*graph, *zp2_node.node_arg)));

          auto mhagrpb_builder = NodeBuilder(*graph, *self);
          mhagrpb_builder.set_input_node_args(
              {in_node0.node_arg, in_node1.node_arg, in_node3.node_arg,
               in_node2.node_arg, c4_arg, c5_arg, c6_arg, c7_arg, c8_arg});

          std::vector<std::string> in_dtypes = {"uint8",    "uint8", "uint8",
                                                "bfloat16", "uint8", "int64",
                                                "int32",    "uint8", "int32"};
          std::vector<std::string> out_dtypes = {"uint8"};
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

DEFINE_VAIP_PASS(MergeQMHAGRPB, vaip_pass_dd_merge_qmhagrpb_mxgan)
