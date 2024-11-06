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
#include <glog/logging.h>

#ifdef __GNUC__
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wpedantic"
#  pragma GCC diagnostic ignored "-Wconversion"
#endif

#include "vaip/dd/coeffs.hpp"
#include "vaip/dd/dd_utils.hpp"

#ifdef __GNUC__
#  pragma GCC diagnostic pop
#endif

#include "vaip/vaip.hpp"
#include "vitis/ai/env_config.hpp"

DEF_ENV_PARAM(DEBUG_DD_MERGE_MLADFMATMUL, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_DD_MERGE_MLADFMATMUL) >= n)

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
    { "name": "vaip_pass_dd_merge_mladfmatmul",
       "plugin": "vaip-pass_dd_merge_mladfmatmul",
       "disabled": false
    }
*/
namespace {
using namespace vaip_core;
struct Dd_merge_mladfmatmul {
  Dd_merge_mladfmatmul(IPass& self) : self_{self} {}
  std::unique_ptr<Rule> create_rule(IPass* self) {
    auto builder = PatternBuilder();
    auto input_0 = builder.wildcard();
    auto scale_0 = builder.constant();
    auto zp_0 = builder.constant();
    auto dq_0 = builder.node2("com.microsoft:DequantizeLinear",
                              {input_0, scale_0, zp_0});
    auto input_1 = builder.wildcard();
    auto scale_1 = builder.constant();
    auto zp_1 = builder.constant();
    auto dq_1 = builder.node2("com.microsoft:DequantizeLinear",
                              {input_1, scale_1, zp_1});
    auto matmul = builder.node2("MatMul", {dq_0, dq_1});
    auto scale_2 = builder.constant();
    auto zp_2 = builder.constant();
    auto q =
        builder.node2("com.microsoft:QuantizeLinear", {matmul, scale_2, zp_2});

    return Rule::create_rule(
        q, [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
          auto in0_node = binder[input_0->get_id()];
          auto scale0_node = binder[scale_0->get_id()];
          auto zp0_node = binder[zp_0->get_id()];
          auto in1_node = binder[input_1->get_id()];
          auto scale1_node = binder[scale_1->get_id()];
          auto zp1_node = binder[zp_1->get_id()];
          auto scale2_node = binder[scale_2->get_id()];
          auto zp2_node = binder[zp_2->get_id()];
          auto out_node = binder[q->get_id()];

          std::vector<std::string> ns = vaip::dd::get_node_names(graph, binder);
          MY_LOG(1) << "found match at " << ns.front();

          auto out_shape = node_arg_get_shape_i64(*out_node.node_arg);
          auto in0_shape = node_arg_get_shape_i64(*in0_node.node_arg);

          auto in0_scale =
              node_arg_get_const_data_as_float(*graph, *scale0_node.node_arg);
          auto in0_zp =
              node_arg_get_const_data_as_u16(*graph, *zp0_node.node_arg);
          auto in1_scale =
              node_arg_get_const_data_as_float(*graph, *scale1_node.node_arg);
          auto in1_zp =
              node_arg_get_const_data_as_u16(*graph, *zp1_node.node_arg);
          auto out_scale =
              node_arg_get_const_data_as_float(*graph, *scale2_node.node_arg);
          auto out_zp =
              node_arg_get_const_data_as_u16(*graph, *zp2_node.node_arg);

          const int32_t SV_M = 16;
          const int32_t SV_K = 256;
          const int32_t SV_N = 8;
          const int32_t K_dim = static_cast<int32_t>(in0_shape->back());
          const int32_t k_iter = K_dim / SV_K;
          auto coeff_qkt =
              vaip::dd::qmatmulcalc::qdq_act_matmul_uint16_uint16_cstm(
                  in0_scale, in0_zp, K_dim, in1_scale, in1_zp, out_scale,
                  out_zp);
          std::vector<int32_t> kernel_params(16, 0);
          kernel_params[0] = SV_M;
          kernel_params[1] = SV_K;
          kernel_params[2] = SV_N;
          kernel_params[3] = k_iter;
          kernel_params[4] = 0x2000;
          kernel_params[5] = 0x4800;
          kernel_params[6] = 0x3800;
          kernel_params[7] = 0x3C00;
          kernel_params[8] = 0x4000;
          kernel_params[9] = 0x4400;
          int64_t* p_param = reinterpret_cast<int64_t*>(&kernel_params[10]);
          *p_param = std::get<0>(coeff_qkt);
          kernel_params[12] = std::get<3>(coeff_qkt);
          kernel_params[13] = std::get<1>(coeff_qkt);
          kernel_params[14] = static_cast<int32_t>(std::get<2>(coeff_qkt));
          auto matmul_shift = std::get<6>(coeff_qkt);
          auto shift_out = std::get<5>(coeff_qkt);
          kernel_params[15] = matmul_shift | shift_out << 16;
          auto node_name = node_arg_get_name(*out_node.node_arg);
          auto& input_qdq_arg = vaip::dd::insert_named_tensor_in_graph(
              graph, node_name + "_qdq_kparams_", kernel_params,
              std::vector({(int64_t)kernel_params.size()}));

          std::vector<std::string> input_types{"uint16", "uint16", "int32"};
          std::vector<std::string> output_types{"uint16"};

          NodeBuilder(*graph, *self)
              .set_input_node_args(
                  {in0_node.node_arg, in1_node.node_arg, &input_qdq_arg})
              .set_op_type("MLADFMATMULA16A16", "com.xilinx")
              .add("nodes", ns)
              .add("orig_output_shape", *out_shape)
              .add("input_shape", *in0_shape)
              .add("design_param", "'4x2") // hardcoded for now
              .add("in_dtypes", input_types)
              .add("out_dtypes", output_types)
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

DEFINE_VAIP_PASS(Dd_merge_mladfmatmul, vaip_pass_dd_merge_mladfmatmul)
