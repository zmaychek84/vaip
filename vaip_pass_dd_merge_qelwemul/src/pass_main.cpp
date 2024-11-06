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

#include "vaip/pattern_zoo.hpp"
#include "vaip/vaip.hpp"
#include "vitis/ai/env_config.hpp"

DEF_ENV_PARAM(DEBUG_DD_MERGE_QELWEMUL, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_DD_MERGE_QELWEMUL) >= n)

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
    { "name": "vaip_pass_dd_merge_qelwemul",
       "plugin": "vaip-pass_dd_merge_qelwemul",
       "disabled": false
    }
*/
namespace {
using namespace vaip_core;

// filter out constant mul?
bool check_if_ele_mul(const Graph& graph, const NodeArg* input_0,
                      const NodeArg* input_1) {
  return (!node_arg_is_constant(graph, *input_0)) &&
         (!node_arg_is_constant(graph, *input_1));
}

// find qlinear's scale and zp above QSlice
[[maybe_unused]] std::tuple<bool, float, uint16_t>
get_qlinear_param(const Graph& graph, const Node* qslice_node) {
  std::tuple<bool, float, uint16_t> ret{false, 0.0f, uint16_t(0)};
  if (!qslice_node || node_op_type(*qslice_node) != "QSlice") {
    return ret;
  }
  auto slice_inputs = node_get_inputs(*qslice_node);
  if (slice_inputs.size() > 0 && slice_inputs[0].node &&
      node_op_type(*(slice_inputs[0].node)) == "QuantizeLinear") {
    auto qlinear_inputs = node_get_inputs(*(slice_inputs[0].node));
    auto scale =
        node_arg_get_const_data_as_float(graph, *(qlinear_inputs[1].node_arg));
    auto zp = qlinear_inputs.size() > 2
                  ? node_arg_get_const_data_as_u16(
                        graph, *(qlinear_inputs[2].node_arg))
                  : uint16_t(0);
    ret = {true, scale, zp};
  }
  return ret;
}
static std::pair<float, uint16_t> get_scale_zp_with_ancestor_check(
    onnxruntime::Graph* graph, binder_t& binder, vaip_core::NodeInput& a,
    vaip_core::NodeInput& as_node, vaip_core::NodeInput& az_node) {
  auto a_sc = node_arg_get_const_data_as_float(*graph, *as_node.node_arg);
  auto a_zp = vaip::dd::get_zp_from_node(*graph, *az_node.node_arg);
  auto parent_op_type = VAIP_ORT_API(node_op_type)(*a.node);
  MY_LOG(1) << " " << a_sc << " " << a_zp;
  if (parent_op_type == "QSlice") {
    // Pick Q params from here
    MY_LOG(1) << "Have to get the q_params from here instead";
    auto& attrs = node_get_attributes_ref(*a.node);
    auto attr_proto = node_attributes_get(attrs, "q_scale");
    a_sc = VAIP_ORT_API(attr_proto_get_float)(*attr_proto);
    attr_proto = node_attributes_get(attrs, "q_zp");
    a_zp = (uint16_t)(VAIP_ORT_API(attr_proto_get_int)(*attr_proto));
    MY_LOG(1) << " " << a_sc << " " << a_zp;
  }
  MY_LOG(1) << "DONE";
  auto ret = std::make_pair(a_sc, a_zp);
  return ret;
}

struct Dd_merge_qelwemul {
  Dd_merge_qelwemul(IPass& self) : self_{self} {}
  std::unique_ptr<Rule> create_rule(IPass* self) {

    auto com_microsoft_QuantizeLinear_0 =
        vaip::pattern_zoo::get_pattern("m_qelwemul");
    CHECK(com_microsoft_QuantizeLinear_0 != nullptr)
        << "Pattern returned is null";

    return Rule::create_rule(
        com_microsoft_QuantizeLinear_0,
        [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
          auto in_node_0 = binder["input_1"];
          auto in_0_scale_node = binder["constant_2"];
          auto in_0_zp_node = binder["constant_3"];
          auto r = get_scale_zp_with_ancestor_check(
              graph, binder, in_node_0, in_0_scale_node, in_0_zp_node);
          auto in_0_scale = r.first;
          auto in_0_zp = r.second;

          auto in_node_1 = binder["input_0"];
          auto in_1_scale_node = binder["constant_0"];
          auto in_1_zp_node = binder["constant_1"];
          r = get_scale_zp_with_ancestor_check(graph, binder, in_node_1,
                                               in_1_scale_node, in_1_zp_node);
          auto in_1_scale = r.first;
          auto in_1_zp = r.second;

          if (!check_if_ele_mul(*graph, in_node_0.node_arg,
                                in_node_1.node_arg)) {
            return false;
          }

          auto out_scale_node = binder["constant_4"];
          auto out_zp_node = binder["constant_5"];
          auto out_node = binder["com_microsoft_QuantizeLinear_0"];
          auto out_scale = node_arg_get_const_data_as_float(
              *graph, *out_scale_node.node_arg);
          auto out_zp =
              node_arg_get_const_data_as_u16(*graph, *out_zp_node.node_arg);
          auto out_shape = node_arg_get_shape_i64(*out_node.node_arg);

          std::vector<std::string> ns = vaip::dd::get_node_names(graph, binder);
          //   MY_LOG(1) << "found match at " << ns.front();
          //   auto [success, modified_b_input_scale, modified_b_input_zp] =
          //       get_qlinear_param(*graph, in_node_0.node);
          //   if (!success) {
          //     MY_LOG(1) << "parameters for modified b input not found";
          //   }

          std::vector<float> input_q_params{in_0_scale, float(in_0_zp),
                                            in_1_scale, float(in_1_zp)};
          std::vector<float> output_q_params{out_scale, float(out_zp)};
          auto [a_scale, a_zp] =
              vaip::dd::qmatmulcalc::calc_lrn_coeff(in_0_scale, in_0_zp);
          auto [b_scale, b_zp] =
              vaip::dd::qmatmulcalc::calc_lrn_coeff(in_1_scale, in_1_zp);
          auto [final_out_scale, final_out_zp] =
              vaip::dd::qmatmulcalc::calc_lrn_coeff(1 / out_scale, out_zp);

          int32_t amat_uint16 = 0; // is_matA_uint16
          int32_t cmat_uint16 = 1; // is_matC_uint16

          std::vector<int32_t> elt_coeffs(16, 0);
          elt_coeffs[0] = b_scale;
          elt_coeffs[1] = b_zp;
          elt_coeffs[2] = a_scale;
          elt_coeffs[3] = a_zp;
          elt_coeffs[4] = final_out_scale;
          elt_coeffs[5] = final_out_zp;
          elt_coeffs[6] = amat_uint16;
          elt_coeffs[7] = cmat_uint16;
          auto node_name = node_arg_get_name(*out_node.node_arg);
          std::string elt_coeff_name = std::string(node_name + "_qdq_");
          auto& elt_arg = vaip::dd::insert_named_tensor_in_graph<int32_t>(
              graph, elt_coeff_name, elt_coeffs,
              std::vector({(int64_t)elt_coeffs.size()}));

          // hard code for mzdk5, may need to change
          std::vector<std::string> input_types{"bfloat16", "uint16", "int32"};
          std::vector<std::string> output_types{"uint16"};

          // input order is changed
          NodeBuilder(*graph, *self)
              .set_input_node_args(
                  {in_node_1.node_arg, in_node_0.node_arg, &elt_arg})
              .set_op_type("QELWEMUL_qdq", "com.xilinx")
              .add("nodes", ns)
              .add("input_shape", *out_shape) // same as python
              .add("orig_output_shape", *out_shape)
              .add("input_q_params", input_q_params)
              .add("output_q_params", output_q_params)
              //   .add("modified_b_input_scale",
              //        std::to_string(modified_b_input_scale))
              //   .add("modified_b_input_zp",
              //   std::to_string(modified_b_input_zp))
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

DEFINE_VAIP_PASS(Dd_merge_qelwemul, vaip_pass_dd_merge_qelwemul)
