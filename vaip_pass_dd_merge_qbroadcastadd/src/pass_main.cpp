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
#include "vaip/pattern_zoo.hpp"

#ifdef __GNUC__
#  pragma GCC diagnostic pop
#endif
#include "vaip/vaip.hpp"
#include "vitis/ai/env_config.hpp"

DEF_ENV_PARAM(DEBUG_DD_MERGE_QBROADCASTADD, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_DD_MERGE_QBROADCASTADD) >= n)

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
    { "name": "vaip_pass_dd_merge_qbroadcastadd",
       "plugin": "vaip-pass_dd_merge_qbroadcastadd",
       "disabled": false
    }
*/
namespace {
using namespace vaip_core;
std::vector<std::string> change_inputs(const NodeArg& a, const NodeArg& b,
                                       const NodeArg& c) {
  std::vector<std::string> dtypes;
  // Add conditional code here (Below may only work for mzdk5)
  auto a_dtype = node_arg_get_element_type(a);
  if (a_dtype == (int)ONNX_NAMESPACE::TensorProto_DataType_UINT16)
    dtypes.emplace_back("uint16");
  else
    dtypes.emplace_back("uint8");

  auto b_dtype = node_arg_get_element_type(b);
  if (b_dtype == (int)ONNX_NAMESPACE::TensorProto_DataType_UINT16)
    dtypes.emplace_back("uint16");
  else
    dtypes.emplace_back("uint8");
  //   dtypes.emplace_back("uint16");
  dtypes.emplace_back(vaip::dd::nodearg_dtype_to_string(c));

  return dtypes;
}

std::vector<std::string> change_outputs(const NodeArg& a) {
  std::vector<std::string> dtypes;
  // Add conditional code here (Below may only work for mzdk5)
  //   dtypes.emplace_back("bfloat16");
  dtypes.emplace_back(vaip::dd::nodearg_dtype_to_string(a));

  return dtypes;
}

struct Dd_merge_qbroadcastadd {
  Dd_merge_qbroadcastadd(IPass& self) : self_{self} {}
  std::unique_ptr<Rule> create_rule(IPass* self) {
    auto com_microsoft_QuantizeLinear_0 =
        vaip::pattern_zoo::get_pattern("m_qbroadcastadd");
    CHECK(com_microsoft_QuantizeLinear_0 != nullptr)
        << "Pattern returned is null";
    return Rule::create_rule(
        com_microsoft_QuantizeLinear_0,
        [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
          auto in_node_0 = binder["input_0"];
          auto in_0_scale_node = binder["constant_0"];
          auto in_0_zp_node = binder["constant_1"];
          auto in_0_scale = node_arg_get_const_data_as_float(
              *graph, *in_0_scale_node.node_arg);
          auto in_0_zp =
              node_arg_get_const_data_as_u16(*graph, *in_0_zp_node.node_arg);
          auto in_0_shape = node_arg_get_shape_i64(*in_node_0.node_arg);

          auto in_node_1 = binder["input_1"];
          auto in_1_scale_node = binder["constant_2"];
          auto in_1_zp_node = binder["constant_3"];
          auto in_1_scale = node_arg_get_const_data_as_float(
              *graph, *in_1_scale_node.node_arg);
          auto in_1_zp =
              node_arg_get_const_data_as_u16(*graph, *in_1_zp_node.node_arg);
          auto in_1_shape = node_arg_get_shape_i64(*in_node_1.node_arg);

          auto out_scale_node = binder["constant_4"];
          auto out_zp_node = binder["constant_5"];
          auto out_node = binder["com_microsoft_QuantizeLinear_0"];
          auto out_scale = node_arg_get_const_data_as_float(
              *graph, *out_scale_node.node_arg);
          auto out_zp =
              node_arg_get_const_data_as_u16(*graph, *out_zp_node.node_arg);
          auto out_shape = node_arg_get_shape_i64(*out_node.node_arg);
          auto output_shape = vaip::dd::shape_as_string(*(out_shape.get()));
          // TODO: check_if_wts_add, for m3uec all matched add is used:
          // fuse = True, fuse_badd = False
          // std::cout<<"Matched nodes before shape check\n";

          if (*(in_0_shape.get()) == *(in_1_shape.get()))
            return false; // Not qbroadcastadd

          // std::vector<int64_t> s = *(in_0_shape.get());
          // for(auto x : s)
          //    std::cout<<x<<"\n";
          // std::cout<<"Matched nodes after shape check\n";
          std::vector<std::string> ns = vaip::dd::get_node_names(graph, binder);
          MY_LOG(1) << "found match at " << ns.front();
          MY_LOG(1) << vaip::dd::shape_as_string(*(in_0_shape.get()));
          MY_LOG(1) << vaip::dd::shape_as_string(*(in_1_shape.get()));
          MY_LOG(1) << vaip::dd::shape_as_string(*(out_shape.get()));

          // INPUT AND OUTPUT _Q_PARAMS
          std::vector<float> input_q_params{in_0_scale, float(in_0_zp),
                                            in_1_scale, float(in_1_zp)};
          std::vector<float> output_q_params{out_scale, float(out_zp)};

          auto s = *(in_1_shape.get());
          if ((s[s.size() - 1] == 1) && (s[s.size() - 2] == 1)) {
            std::swap(input_q_params[0], input_q_params[2]);
            std::swap(input_q_params[1], input_q_params[3]);
          }

          std::vector<uint16_t> elt_qdq_tensor(32, 0);
          elt_qdq_tensor[1] =
              vaip::dd::qmatmulcalc::float_to_bfloat16(input_q_params[0]);
          elt_qdq_tensor[0] = uint16_t(input_q_params[1]);
          elt_qdq_tensor[3] =
              vaip::dd::qmatmulcalc::float_to_bfloat16(input_q_params[2]);
          elt_qdq_tensor[2] = uint16_t(input_q_params[3]);

          elt_qdq_tensor[5] =
              vaip::dd::qmatmulcalc::float_to_bfloat16(1.0f / out_scale);
          elt_qdq_tensor[4] = out_zp;

          auto node_name = node_arg_get_name(*out_node.node_arg);
          std::string elt_coeff_name = std::string(node_name + "_qdq_");
          auto& elt_arg = vaip::dd::insert_named_tensor_in_graph<uint16_t>(
              graph, elt_coeff_name, elt_qdq_tensor,
              std::vector({(int64_t)elt_qdq_tensor.size()}));

          std::vector<const NodeArg*> new_inputs{in_node_0.node_arg,
                                                 in_node_1.node_arg, &elt_arg};

          if ((s[s.size() - 1] == 1) && (s[s.size() - 2] == 1)) {
            std::swap(new_inputs[0], new_inputs[1]);
          }
          NodeBuilder(*graph, *self)
              .set_input_node_args(new_inputs)
              .set_op_type("QBroadcastAdd", "com.xilinx")
              .add("nodes", ns)
              .add("input_q_params", input_q_params)
              .add("output_q_params", output_q_params)
              .add("in_dtypes", change_inputs(*in_node_0.node_arg,
                                              *in_node_1.node_arg, elt_arg))
              .add("out_dtypes", change_outputs(*out_node.node_arg))
              .add("orig_output_shape", output_shape)
              .set_anchor_point1(*out_node.node)
              .build();
          return true; // return true if graph is modified.
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

DEFINE_VAIP_PASS(Dd_merge_qbroadcastadd, vaip_pass_dd_merge_qbroadcastadd)
