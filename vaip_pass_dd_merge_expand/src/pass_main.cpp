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

#ifdef __GNUC__
#  pragma GCC diagnostic pop
#endif
#include "vaip/pattern_zoo.hpp"
#include "vaip/vaip.hpp"
#include "vitis/ai/env_config.hpp"
#include <glog/logging.h>

DEF_ENV_PARAM(DEBUG_DD_PATTERN, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_DD_PATTERN) >= n)
using namespace vaip_core;

struct MergeExpand {
  MergeExpand(IPass& self) : self_{self} {}

  std::unique_ptr<Rule> create_rule(IPass* self) {

    auto com_microsoft_QuantizeLinear_2 =
        vaip::pattern_zoo::get_pattern("m_merge_expand");
    CHECK(com_microsoft_QuantizeLinear_2 != nullptr)
        << "Pattern returned is null";

    return Rule::create_rule(
        com_microsoft_QuantizeLinear_2,
        [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
          auto attr_nodes = vaip::dd::get_node_names(graph, binder);

          auto a_node = binder["input_0"];
          auto a_shape = node_arg_get_shape_i64(*a_node.node_arg);

          auto expand_inputs = binder["constant_2"];
          auto shape_gsl =
              node_arg_get_const_data_as_i64s(*graph, *expand_inputs.node_arg);
          std::vector<int64_t> shape_out_i64(shape_gsl.begin(),
                                             shape_gsl.end());
          std::vector<uint16_t> shape_out(shape_out_i64.size(), 0);

          for (long unsigned int i = 0; i < shape_out.size(); i++) {
            shape_out[i] = static_cast<uint16_t>(shape_out_i64[i]);
          }

          auto in_scale_node = binder["constant_0"];
          auto in_zp_node = binder["constant_1"];
          auto in_scale =
              node_arg_get_const_data_as_float(*graph, *in_scale_node.node_arg);
          auto in_zp =
              node_arg_get_const_data_as_u16(*graph, *in_zp_node.node_arg);

          auto q_node = binder["com_microsoft_QuantizeLinear_2"];
          auto q_shape = node_arg_get_shape_i64(*q_node.node_arg);
          auto node_name = node_arg_get_name(*q_node.node_arg);

          auto out_scale_node = binder["constant_5"];
          auto out_zp_node = binder["constant_6"];
          auto out_scale = node_arg_get_const_data_as_float(
              *graph, *out_scale_node.node_arg);
          auto out_zp =
              node_arg_get_const_data_as_u16(*graph, *out_zp_node.node_arg);

          std::vector<std::string> input_types{"bfloat16", "uint16"};
          std::vector<std::string> output_types{"bfloat16"};
          std::vector<float> input_q_params{in_scale, (float)in_zp};
          std::vector<float> output_q_params{out_scale, (float)out_zp};

          std::string shape_out_name = std::string(node_name + "_shape_");
          auto& shape_arg = vaip::dd::insert_named_tensor_in_graph<uint16_t>(
              graph, shape_out_name, shape_out,
              std::vector({(int64_t)shape_out.size()}));

          auto expand = NodeBuilder(*graph, *self);
          expand.set_input_node_args({a_node.node_arg, &shape_arg});
          expand.set_op_type("QExpand", "com.xilinx");
          expand.set_anchor_point1(
              *binder["com_microsoft_QuantizeLinear_2"].node);
          expand.add("nodes", attr_nodes);
          expand.add("in_dtypes", input_types);
          expand.add("out_dtypes", output_types);
          expand.add("input_q_params", input_q_params);
          expand.add("output_q_params", output_q_params);
          expand.add("input1_shape", *(a_shape.get()));
          expand.add("orig_output_shape", *(q_shape.get()));
          expand.build();
          return true;
        });
  }

  void process(IPass& self, Graph& graph) { create_rule(&self)->apply(&graph); }

public:
  IPass& self_;
};

DEFINE_VAIP_PASS(MergeExpand, vaip_pass_dd_merge_expand)
