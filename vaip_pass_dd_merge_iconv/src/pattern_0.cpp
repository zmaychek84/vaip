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

struct MergeIConv_0 {
  MergeIConv_0(IPass& self) : self_{self} {}

  static std::unique_ptr<Rule> create_rule_0(IPass* self) {

    auto ms_QuantizeLinear_29 = vaip::pattern_zoo::get_pattern("m_iconv_0");
    CHECK(ms_QuantizeLinear_29 != nullptr) << "Pattern returned is null";

    return Rule::create_rule(
        ms_QuantizeLinear_29,
        [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
          // transpose -> reshape -> conv -> reshape -> transpose
          auto in_node = binder["input_0"];
          auto in_scale_node = binder["constant_1"];
          auto in_zp_node = binder["constant_2"];
          auto act_node = binder["ms_DequantizeLinear_10"];
          auto kernel_node = binder["ms_DequantizeLinear_14"];
          // weights
          auto wt_node = binder["constant_11"];
          auto wt_scale_node = binder["constant_12"];
          auto wt_zp_node = binder["constant_13"];
          // bias
          auto bias_node = binder["constant_15"];
          auto bias_scale_node = binder["constant_16"];
          auto bias_zp_node = binder["constant_17"];
          // conv out q
          auto conv_outq_scale_node = binder["constant_20"];
          auto conv_outq_zp_node = binder["constant_21"];

          auto conv_node = binder["Conv_19"];
          auto transpose_node_0 = binder["Transpose_4"];
          auto reshape_node_0 = binder["Reshape_8"];
          auto reshape_node_1 = binder["Reshape_25"];
          auto transpose_node_1 = binder["Transpose_28"];
          auto out_node = binder["ms_QuantizeLinear_29"];

          std::vector<std::string> ns = vaip::dd::get_node_names(graph, binder);
          MY_LOG(1) << "found match at " << ns.front();

          auto in_scale =
              node_arg_get_const_data_as_float(*graph, *in_scale_node.node_arg);
          auto in_zero_point =
              node_arg_get_const_data_as_u16(*graph, *in_zp_node.node_arg);
          auto act_in_shape = node_arg_get_shape_i64(*act_node.node_arg);
          auto kernel_shape = node_arg_get_shape_i64(*kernel_node.node_arg);
          auto out_shape = node_arg_get_shape_i64(*conv_node.node_arg);
          auto wt_name = node_arg_get_name(*wt_node.node_arg);
          auto transpose_perm_span_0 =
              node_get_attr_ints(*transpose_node_0.node, "perm");
          std::vector<int64_t> transpose_perm_0(transpose_perm_span_0.begin(),
                                                transpose_perm_span_0.end());
          auto reshape_0_allowzero =
              node_get_attr_int(*reshape_node_0.node, "allowzero");
          auto reshape_1_allowzero =
              node_get_attr_int(*reshape_node_1.node, "allowzero");
          auto transpose_perm_span_1 =
              node_get_attr_ints(*transpose_node_1.node, "perm");
          std::vector<int64_t> transpose_perm_1(transpose_perm_span_1.begin(),
                                                transpose_perm_span_1.end());

          // 10 parameters needed for qdq calc
          // in_scale, in_zero_point
          // weights/weights scale/weights zp
          auto weights =
              node_arg_get_const_data_as_u8s(*graph, *wt_node.node_arg);
          auto weights_scale =
              node_arg_get_const_data_as_float(*graph, *wt_scale_node.node_arg);
          auto weights_zero_point =
              node_arg_get_const_data_as_u8(*graph, *wt_zp_node.node_arg);
          // bias/bias scale/bias zp
          auto bias =
              node_arg_get_const_data_as_i32s(*graph, *bias_node.node_arg);
          // bias scale is array[1]
          auto bias_scale = node_arg_get_const_data_as_floats(
              *graph, *bias_scale_node.node_arg);
          auto bias_zero_point =
              node_arg_get_const_data_as_i32(*graph, *bias_zp_node.node_arg);
          // conv output q scale/zp
          auto conv_outq_scale = node_arg_get_const_data_as_float(
              *graph, *conv_outq_scale_node.node_arg);
          auto conv_outq_zero_point = node_arg_get_const_data_as_u16(
              *graph, *conv_outq_zp_node.node_arg);

          auto [C0, C1, C2, conv_shift, shft_c2] =
              vaip::dd::qmatmulcalc::dq_uint16A_uint8W_conv_q_param_gen(
                  in_scale, in_zero_point, weights, weights_scale,
                  weights_zero_point, *kernel_shape, bias, bias_scale[0],
                  bias_zero_point, conv_outq_scale, conv_outq_zero_point);

          auto node_name = node_arg_get_name(*out_node.node_arg);
          auto& input_c0_arg = vaip::dd::insert_named_tensor_in_graph<int64_t>(
              graph, node_name + "_c0_", C0, std::vector({(int64_t)C0.size()}));
          std::vector<int32_t> input_qdq(16, 0);
          input_qdq[2] = static_cast<int32_t>(C1);
          input_qdq[3] = static_cast<int32_t>(C2);
          input_qdq[8] = static_cast<int32_t>(shft_c2);
          input_qdq[9] = static_cast<int32_t>(conv_shift);
          input_qdq[10] = weights_zero_point;
          input_qdq[11] = in_zero_point;
          auto& input_qdq_arg = vaip::dd::insert_named_tensor_in_graph<int32_t>(
              graph, node_name + "_qdq_", input_qdq,
              std::vector({(int64_t)input_qdq.size()}));
          // hard code for m3uec, may need to change
          std::vector<std::string> input_types{"uint16", "uint8", "int64",
                                               "int32"};
          std::vector<std::string> output_types{"uint16"};

          NodeBuilder(*graph, *self)
              .set_input_node_args({in_node.node_arg, wt_node.node_arg,
                                    &input_c0_arg, &input_qdq_arg})
              .set_op_type("IConv", "com.xilinx")
              .clone_attrs(*conv_node.node)
              .add("nodes", ns)
              .add("input_shape", *act_in_shape)
              .add("weight_shape", *kernel_shape)
              .add("output_shape", *out_shape)
              .add("zero_point", int64_t(in_zero_point))
              .add("wt_name", wt_name)
              .add("reshape_0_allowzero", reshape_0_allowzero)
              .add("reshape_1_allowzero", reshape_1_allowzero)
              .add("transpose_0_perm", transpose_perm_0)
              .add("transpose_1_perm", transpose_perm_1)
              .add("in_dtypes", input_types)
              .add("out_dtypes", output_types)
              .add("input_format", "NHWC")
              .add("C1", std::to_string(C1))
              .add("C2", std::to_string(C2))
              .add("shift_conv", std::to_string(conv_shift))
              .add("shift_final", std::to_string(shft_c2))
              .set_anchor_point1(*out_node.node)
              .build();
          return true;
        });
  }

  void process(IPass& self, Graph& graph) {
    MY_LOG(1) << "try matching IConv pattern 0";
    create_rule_0(&self)->apply(&graph);
  }

public:
  IPass& self_;
};

DEFINE_VAIP_PASS(MergeIConv_0, vaip_pass_dd_merge_iconv_0)
