/*
 *     The Xilinx Vitis AI Vaip in this distribution are provided under the
 * following free and permissive binary-only license, but are not provided in
 * source code form.  While the following free and permissive license is similar
 * to the BSD open source license, it is NOT the BSD open source license nor
 * other OSI-approved open source license.
 *
 *      Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc.
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

#include "vaip/dd/coeffs.hpp"
#include "vaip/dd/dd_utils.hpp"
#include "vaip/pattern_zoo.hpp"
#include "vaip/vaip.hpp"
#include "vitis/ai/env_config.hpp"
#include <glog/logging.h>

DEF_ENV_PARAM(DEBUG_DD_MERGE_ATTENTIONPREPRO, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_DD_MERGE_ATTENTIONPREPRO) >= n)

/**
 * test case: <???>
 *
 *
 * Replace pattern:
 *
 * From: <???>
 * To  : <???>
 */

// add the following line in your vaip_config.json
/*
    { "name": "vaip_pass_dd_merge_attentionprepro_mxpzi",
       "plugin": "vaip-pass_dd_merge_attentionprepro_mxpzi",
       "disabled": false
    }
*/
namespace {
using namespace vaip_core;
struct Dd_merge_attentionprepro_mxpzi {
  static std::vector<std::string> change_inputs(const NodeArg& a,
                                                const NodeArg& b) {
    //     std::cout<<"CHANGE INPUTS\n";
    std::vector<std::string> dtypes;
    // Add conditional code here :TODO
    //     dtypes.emplace_back(vaip::dd::nodearg_dtype_to_string(a));
    dtypes.emplace_back("bfloat16");
    // dtypes.emplace_back(vaip::dd::nodearg_dtype_to_string(b));
    return dtypes;
  }

  static std::vector<std::string> change_outputs(const NodeArg& a) {
    // std::cout<<"CHANGE OUTPUTS\n";
    std::vector<std::string> dtypes;
    // Add conditional code here (Below may only work for mdsqr)
    dtypes.emplace_back("bfloat16");
    // dtypes.emplace_back(vaip::dd::nodearg_dtype_to_string(a));
    return dtypes;
  }
  Dd_merge_attentionprepro_mxpzi(IPass& self) : self_{self} {}
  std::unique_ptr<Rule> create_rule(IPass* self) {
    auto com_microsoft_QuantizeLinear_2 =
        vaip::pattern_zoo::get_pattern("m_AttentionPrePro_2");
    CHECK(com_microsoft_QuantizeLinear_2 != nullptr)
        << "Pattern returned is null";
    return Rule::create_rule(
        com_microsoft_QuantizeLinear_2,
        [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
          // auto input = binder["input_"];
          auto in_node = binder["input_0"];
          auto out_node = binder["com_microsoft_QuantizeLinear_2"];
          auto mul_const_in_node = binder["constant_7"];
          auto mul_const_scale_node = binder["constant_8"];
          auto mul_const_zp_node = binder["constant_9"];
          auto sub_const_in_node = binder["constant_2"];
          auto sub_const_scale_node = binder["constant_0"];
          auto sub_const_zp_node = binder["constant_1"];
          // auto in_scale_node = binder["constant_0"];
          // auto in_zp_node = binder["constant_1"];
          auto out_scale_node = binder["constant_10"];
          auto out_zp_node = binder["constant_11"];
          MY_LOG(1) << "found match at %%%%%%%%%%%%%%%%%%%%%%%";
          auto in_scale = 1.0f;
          // node_arg_get_const_data_as_float(*graph, *in_scale_node.node_arg);
          auto in_zero_point = 0;
          auto in_scale1 = 1.0f;
          // node_arg_get_const_data_as_float(*graph, *in_scale_node.node_arg);
          auto in_zero_point1 = 0;
          // vaip::dd::get_zp_from_node(*graph, *in_zp_node.node_arg);
          std::vector<float> input_q_params;
          input_q_params.push_back(in_scale);
          input_q_params.push_back(float(in_zero_point));
          input_q_params.push_back(in_scale1);
          input_q_params.push_back(float(in_zero_point1));

          auto out_scale = node_arg_get_const_data_as_float(
              *graph, *out_scale_node.node_arg);
          auto out_zero_point =
              vaip::dd::get_zp_from_node(*graph, *out_zp_node.node_arg);
          std::vector<float> output_q_params{out_scale, float(out_zero_point)};

          auto node_name = node_arg_get_name(*out_node.node_arg);
          auto sub_const_in = node_arg_get_const_data_as_u16(
              *graph, *sub_const_in_node.node_arg);

          std::vector<uint16_t> sub_const_vec{sub_const_in};
          auto sub_const_scale = node_arg_get_const_data_as_float(
              *graph, *sub_const_scale_node.node_arg);
          uint16_t sub_const_zp = node_arg_get_const_data_as_u16(
              *graph, *sub_const_zp_node.node_arg);
          auto gamma = vaip::dd::qmatmulcalc::dq_vec_to_bf16(
              sub_const_vec, sub_const_scale, sub_const_zp);
          //   auto gamma_shape =
          //       node_arg_get_shape_i64(*sub_const_in_node.node_arg);
          //   auto& input_gamma_arg = vaip::dd::insert_named_tensor_in_graph(
          //       graph, node_name + "_gamma_", gamma, *gamma_shape);

          uint16_t mul_const_in = node_arg_get_const_data_as_u16(
              *graph, *mul_const_in_node.node_arg);

          auto mul_const_scale = node_arg_get_const_data_as_float(
              *graph, *mul_const_scale_node.node_arg);
          uint16_t mul_const_zp = node_arg_get_const_data_as_u16(
              *graph, *mul_const_zp_node.node_arg);
          std::vector<uint16_t> mul_const_vec{mul_const_zp};
          auto gamma1 = vaip::dd::qmatmulcalc::dq_vec_to_bf16(
              mul_const_vec, mul_const_scale, mul_const_in);
          //   auto gamma1_shape =
          //       node_arg_get_shape_i64(*mul_const_in_node.node_arg);
          //   auto& input_gamma1_arg = vaip::dd::insert_named_tensor_in_graph(
          //       graph, node_name + "_gamma1_", gamma1, *gamma1_shape);

          // qdq
          std::vector<int32_t> lrn_qdq_tensor(16, 0);
          lrn_qdq_tensor[0] = (int32_t)gamma[0];
          lrn_qdq_tensor[1] = (int32_t)gamma1[0];
          // lrn_qdq_tensor[2] = 1;
          MY_LOG(1) << "found match at $$$$$$$$$$$$$$$$$$$$$$$ ";
          std::string qdq_name = std::string(node_name + "_qdq_");
          auto& lrn_qdq_arg = vaip::dd::insert_named_tensor_in_graph<int32_t>(
              graph, qdq_name, lrn_qdq_tensor,
              std::vector({(int64_t)lrn_qdq_tensor.size()}));

          std::vector<std::string> input_types{"bfloat16", "bfloat16"};
          std::vector<std::string> output_types{"bfloat16"};

          std::vector<std::string> ns = vaip::dd::get_node_names(graph, binder);
          MY_LOG(1) << "found match at " << ns.front();
          NodeBuilder(*graph, *self)
              .set_input_node_args({in_node.node_arg, &lrn_qdq_arg})
              .set_op_type("AttentionMaskPrePro", "com.xilinx")
              .clone_attrs(*out_node.node)
              .add("nodes", ns)
              .add("in_dtypes", input_types)
              .add("out_dtypes", output_types)
              .add("input_q_params", input_q_params)
              .add("output_q_params", output_q_params)
              //.add("qdq_params", qdq_params)
              .set_anchor_point1(*out_node.node)
              .build();
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

DEFINE_VAIP_PASS(Dd_merge_attentionprepro_mxpzi,
                 vaip_pass_dd_merge_attentionprepro_mxpzi)
