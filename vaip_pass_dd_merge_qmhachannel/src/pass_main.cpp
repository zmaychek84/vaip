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
#include "qmha/qmha_processor.hpp"
#include "vaip/dd/dd_utils.hpp"
#include "vaip/pattern_zoo.hpp"
#include "vaip/vaip.hpp"
#include "vitis/ai/env_config.hpp"
#include <glog/logging.h>
DEF_ENV_PARAM(DEBUG_DD_MERGE_QMHACHANNEL, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_DD_MERGE_QMHACHANNEL) >= n)

/**
 * test case: Model_m3uec_v1.1_e255659e71c9b17f8b2df740f926e2204c06a2f3
 *
 *
 * Replace pattern:
 *
 * From: a complicated pattern
 * To  : QMHACHANNEL
 */

// add the following line in your vaip_config.json
/*
    { "name": "vaip_pass_dd_merge_qmhachannel",
       "plugin": "vaip-pass_dd_merge_qmhachannel",
       "disabled": false
    }
*/
namespace {
using namespace vaip_core;
// clang-format off
#define INPUT_0 "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/qkv/Add_output_0_QuantizeLinear_Output"
// clang-format on
struct DdMergeQmhachannel {
  DdMergeQmhachannel(IPass& self) : self_{self} {}
  std::unique_ptr<Rule> create_rule(IPass* self) {
    auto pattern_ = vaip::pattern_zoo::get_pattern("QMHACHANNEL_0");
    return Rule::create_rule(
        pattern_, [=](onnxruntime::Graph* graph_ptr, binder_t& binder) -> bool {
          auto ni_input = binder[INPUT_0];
          auto ni_output = binder[pattern_->get_id()];
          MY_LOG(1) << "found node:" << node_as_string(*ni_output.node);
          auto const binder_params =
              std::unordered_map<std::string, std::vector<std::string>>{
                  // clang-format off
                {"QKT_input_qparams",{
        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Mul_2_output_0_scale",
        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Mul_2_output_0_zero_point",
        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/qkv/Add_output_0_scale",
        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/qkv/Add_output_0_zero_point",
                    }},
                {"QKT_output_qparams",{
        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/MatMul_output_0_scale",
        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/MatMul_output_0_zero_point",
                    }},
                {"MATMUL_input", {"/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Transpose_2_output_0_DequantizeLinear_Output",
                                  "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Reshape_4_output_0_DequantizeLinear_Output",
                    }},
                {"VSQKT_input_qparams",{
        "/blocks.3/blocks.3.0/channel_block/channel_attn/fn/Softmax_output_0_scale",
        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Softmax_output_0_zero_point",
        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/qkv/Add_output_0_scale",
        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/qkv/Add_output_0_zero_point",
                    }},
                {"VSQKT_output_qparams",{
        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/MatMul_1_output_0_scale",
        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/MatMul_1_output_0_zero_point",
                    }},
                {"VSMATMUL_input", {
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Softmax_output_0_DequantizeLinear_Output",
                        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Transpose_3_output_0_DequantizeLinear_Output"
                    }},
                {"softmax_input_qparams",{
        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/MatMul_output_0_scale",
        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/MatMul_output_0_zero_point",
                    }},
                {"softmax_output_qparams",{
        "/blocks.3/blocks.3.0/channel_block/channel_attn/fn/Softmax_output_0_scale",
        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Softmax_output_0_zero_point",
                    }},
                {"MUL_input_qparams",{
        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/qkv/Add_output_0_scale",
        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/qkv/Add_output_0_zero_point",
                    }},
                {"MUL_weight_qparams",{
        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Pow_output_0_quantized",
        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Pow_output_0_scale",
        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Pow_output_0_zero_point",
                    }},
                {"MUL_output_qparams",{
        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Mul_2_output_0_scale",
        "/blocks.0/blocks.0.0/channel_block/channel_attn/fn/Mul_2_output_0_zero_point",
                    }},
                  // clang-format on
              };
          auto processor =
              std::make_unique<vaip_dd_merge_qma::DdMergeQmhaProcessor>(
                  self_, graph_ptr, &binder, binder_params);
          std::vector<std::string> ns =
              vaip::dd::get_node_names(graph_ptr, binder);
          auto& node_arg_qdq_params = processor->process(pattern_->get_id());
          NodeBuilder(*graph_ptr, self_)
              .set_op_type("QMHACHANNEL", "com.xilinx")
              .set_input_node_args({ni_input.node_arg, &node_arg_qdq_params})
              .clone_data_type(*ni_output.node_arg)
              .clone_shape(*ni_output.node_arg)
              .add("nodes", ns)
              .add("in_dtypes", std::vector<std::string>({"uint16", "int64"}))
              .add("out_dtypes", std::vector<std::string>({"uint16"}))
              .set_anchor_point1(*ni_output.node_arg)
              .build();
          return true;
        });
  }
  // apply the rule
  void process(IPass& self, Graph& graph) { create_rule(&self)->apply(&graph); }
  IPass& self_;
};
} // namespace

DEFINE_VAIP_PASS(DdMergeQmhachannel, vaip_pass_dd_merge_qmhachannel)
