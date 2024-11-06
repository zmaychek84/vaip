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
#include <functional>
#include <glog/logging.h>
#include <numeric>

DEF_ENV_PARAM(DEBUG_DD_MERGE_REDUCE_SUM, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_DD_MERGE_REDUCE_SUM) >= n)

// add the following line in your vaip_config.json
/*
    { "name": "vaip_pass_dd_merge_reduce_sum",
       "plugin": "vaip-pass_dd_merge_reduce_sum",
       "disabled": false
    }
*/
namespace {
using namespace vaip_core;
struct ReduceSum {
  ReduceSum(IPass& self) : self_{self} {}

  std::unique_ptr<Rule> create_rule(IPass* self) {
    //    std::cout<<"Called create_rule\n";
    auto com_microsoft_QuantizeLinear_0 =
        vaip::pattern_zoo::get_pattern("m_reduce_sum");
    CHECK(com_microsoft_QuantizeLinear_0 != nullptr)
        << "Pattern returned is null";

    return Rule::create_rule(
        com_microsoft_QuantizeLinear_0,
        [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
          std::vector<std::string> ns = vaip::dd::get_node_names(graph, binder);

          // std::cout << "_____________QReduceSum matched!" << std::endl;

          auto a_node = binder["input_0"];
          auto as_node = binder["constant_0"];
          auto az_node = binder["constant_1"];
          auto a_sc =
              node_arg_get_const_data_as_float(*graph, *as_node.node_arg);
          auto a_zp = vaip::dd::get_zp_from_node(*graph, *az_node.node_arg);
          auto q_node = binder["com_microsoft_QuantizeLinear_0"];
          auto qs_node = binder["constant_3"];
          auto qz_node = binder["constant_4"];
          auto q_sc =
              node_arg_get_const_data_as_float(*graph, *qs_node.node_arg);

          auto axes_node = binder["constant_2"];
          auto axes_gsl =
              node_arg_get_const_data_as_i64s(*graph, *axes_node.node_arg);
          std::vector<int64_t> axes(axes_gsl.begin(), axes_gsl.end());
          float axscale = 1.0f;
          int64_t axzp = 0;
          std::vector<uint16_t> axes_bf16 =
              vaip::dd::qmatmulcalc::dq_vec_to_bf16(axes, axscale, axzp);

          auto q_zp = vaip::dd::get_zp_from_node(*graph, *qz_node.node_arg);
          auto q_shape = node_arg_get_shape_i64(*q_node.node_arg);
          auto node_name = node_arg_get_name(*q_node.node_arg);

          std::vector<float> input_q_params = {a_sc, (float)a_zp};
          std::vector<float> output_q_params = {q_sc, (float)q_zp};
          std::string name_ax =
              std::string(node_arg_get_name(*q_node.node_arg) + "_bf16");
          auto& ax_bf16_arg = vaip::dd::insert_named_tensor_in_graph<uint16_t>(
              graph, name_ax, axes_bf16,
              std::vector({(int64_t)axes_bf16.size()}));

          std::vector<std::string> in_dtypes = {"bfloat16", "bfloat16"};
          std::vector<std::string> out_dtypes = {"bfloat16"};

          NodeBuilder(*graph, *self)
              .set_input_node_args({a_node.node_arg, &ax_bf16_arg})
              .set_op_type("QReduceSum", "com.xilinx")
              .clone_attrs(*q_node.node)
              .add("nodes", ns)
              .set_anchor_point1(*q_node.node)
              .add("in_dtypes", in_dtypes)
              .add("out_dtypes", out_dtypes)
              .add("input_q_params", input_q_params)
              .add("output_q_params", output_q_params)
              // .add("orig_output_shape", *(q_shape.get()))
              .build();
          return true; // return true if graph is modified.
        });
  }
  // apply the rule
  void process(IPass& self, Graph& graph) {
    // MY_LOG(1) << self_.get_pass_proto().name() << "[" <<
    // self_.get_pass_proto().plugin() << "] start processing graph";
    create_rule(&self)->apply(&graph);
    // MY_LOG(1) << self_.get_pass_proto().name() << "[" <<
    // self_.get_pass_proto().plugin() << "] finish processing graph";
  }

  IPass& self_;
};
} // namespace

DEFINE_VAIP_PASS(ReduceSum, vaip_pass_dd_merge_reduce_sum)
