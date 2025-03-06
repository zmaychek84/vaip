/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
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

static std::vector<const Node*> get_all_parent_nodes(const Node* cnode) {
  auto node_inputs = node_get_inputs(*cnode);
  std::vector<const Node*> ret;
  for (const auto& ni : node_inputs) {
    if (ni.node != nullptr) {
      ret.emplace_back(ni.node);
    }
  }
  return ret;
}

static bool check_no_op_parent(Graph& g, const Node* a,
                               NodeArg*& updated_node_arg,
                               std::string no_op_name) {
  auto inputs = get_all_parent_nodes(a);
  if (inputs.size() == 0)
    return false;
  auto x = inputs[0];
  auto parent_op_type = VAIP_ORT_API(node_op_type)(*x);
  if (parent_op_type != no_op_name)
    return false;
  else {
    inputs = get_all_parent_nodes(x);
    if (inputs.size() == 0)
      return false;
    x = inputs[0];
    parent_op_type = VAIP_ORT_API(node_op_type)(*x);
    if (parent_op_type != "DequantizeLinear")
      return false;
  }
  auto input_node_args = node_get_input_node_args(*x);
  for (auto ni : input_node_args) {
    if (!node_arg_is_constant(g, *ni)) {
      updated_node_arg = const_cast<NodeArg*>(ni);
      continue;
    }
  }
  return true;
}

static std::tuple<bool, bool> check_silu_in_parent(const Node* in_0) {
  if (in_0 && (node_is_op(*in_0, "QGelu", "com.xilinx") ||
               node_is_op(*in_0, "QConv2MatMulSilu", "com.xilinx"))) {
    return {true, true};
  }
  return {false, false};
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
          auto out_node = binder["com_microsoft_QuantizeLinear_0"];
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

          auto out_scale = node_arg_get_const_data_as_float(
              *graph, *out_scale_node.node_arg);
          auto out_zp =
              node_arg_get_const_data_as_u16(*graph, *out_zp_node.node_arg);
          auto out_shape = node_arg_get_shape_i64(*out_node.node_arg);

          std::vector<std::string> ns = vaip::dd::get_node_names(graph, binder);

          std::vector<float> input_q_params{in_0_scale, float(in_0_zp),
                                            in_1_scale, float(in_1_zp)};
          std::vector<float> output_q_params{out_scale, float(out_zp)};
          auto [a_scale, a_zp] =
              vaip::dd::qmatmulcalc::calc_lrn_coeff(in_0_scale, in_0_zp);
          auto [b_scale, b_zp] =
              vaip::dd::qmatmulcalc::calc_lrn_coeff(in_1_scale, in_1_zp);
          auto [final_out_scale, final_out_zp] =
              vaip::dd::qmatmulcalc::calc_lrn_coeff(1 / out_scale, out_zp);

          // Code for dmacompiler/qdq_mul
          std::string op_name = "QELWEMUL_qdq";
          int32_t amat_uint16 = 0; // is_matA_uint16
          int32_t cmat_uint16 = 1; // is_matC_uint16
          std::vector<std::string> input_types{"bfloat16", "uint16", "int32"};
          std::vector<std::string> output_types{"uint16"};
          std::vector<int32_t> elt_coeffs(16, 0);
          bool intmul = false;
          std::string design_param = "4x4";

          auto node_name = node_arg_get_name(*out_node.node_arg);

          auto args = self->get_pass_proto().args();
          if (!args.empty()) {
            std::vector<std::string> arg;
            for (const auto& str : args) {
              arg.push_back(str);
            }
            if (arg[0] == "intmul")
              intmul = true;
            else
              design_param = arg[0];
          }
          if (intmul == true) {
            op_name = "QIntEltwiseMul";
            input_types[0] = "uint16";
            elt_coeffs[1] = final_out_scale;
            elt_coeffs[0] = final_out_zp;
            NodeArg* no_op_node_arg = nullptr;
            std::string elt_coeff_name_mul = std::string(node_name + "_qdq_");
            auto& elt_arg_mul = vaip::dd::insert_named_tensor_in_graph<int32_t>(
                graph, elt_coeff_name_mul, elt_coeffs,
                std::vector({(int64_t)elt_coeffs.size()}));
            // input order is changed
            bool no_op_in_parent = check_no_op_parent(
                *graph, in_node_1.node, no_op_node_arg, "Unsqueeze");
            NodeBuilder(*graph, *self)
                .set_input_node_args(
                    {in_node_0.node_arg, no_op_node_arg, &elt_arg_mul})
                .set_op_type(op_name, "com.xilinx")
                .add("nodes", ns)
                .add("input_shape", *out_shape)
                .add("orig_output_shape", *out_shape)
                .add("input_q_params", input_q_params)
                .add("output_q_params", output_q_params)
                .add("in_dtypes", input_types)
                .add("out_dtypes", output_types)
                .set_anchor_point1(*out_node.node)
                .build();

          } else {
            amat_uint16 = 0; // is_matA_uint16
            cmat_uint16 = 1; // is_matC_uint16
            op_name = "QELWEMUL_qdq";
            auto [silu_in_parent, first_parent_is_silu] =
                check_silu_in_parent(in_node_0.node);
            if (silu_in_parent) {
              elt_coeffs[0] = a_scale;
              elt_coeffs[1] = a_zp;
              elt_coeffs[2] = b_scale;
              elt_coeffs[3] = b_zp;
              std::swap(input_q_params[0], input_q_params[2]);
              std::swap(input_q_params[1], input_q_params[3]);
            } else {
              elt_coeffs[0] = b_scale;
              elt_coeffs[1] = b_zp;
              elt_coeffs[2] = a_scale;
              elt_coeffs[3] = a_zp;
            }
            elt_coeffs[4] = final_out_scale;
            elt_coeffs[5] = final_out_zp;
            elt_coeffs[6] = amat_uint16;
            elt_coeffs[7] = cmat_uint16;

            std::string elt_coeff_name = std::string(node_name + "_qdq_");
            auto& elt_arg = vaip::dd::insert_named_tensor_in_graph<int32_t>(
                graph, elt_coeff_name, elt_coeffs,
                std::vector({(int64_t)elt_coeffs.size()}));

            std::vector<const NodeArg*> new_inputs;
            if (silu_in_parent) {
              new_inputs = {in_node_0.node_arg, in_node_1.node_arg, &elt_arg};
            } else {
              new_inputs = {in_node_1.node_arg, in_node_0.node_arg, &elt_arg};
            }

            // hard code for mzdk5, may need to change
            // input order is changed
            NodeBuilder(*graph, *self)
                .set_input_node_args(new_inputs)
                .set_op_type(op_name, "com.xilinx")
                .add("nodes", ns)
                .add("input_shape", *out_shape)
                .add("orig_output_shape", *out_shape)
                .add("input_q_params", input_q_params)
                .add("output_q_params", output_q_params)
                .add("design_param", design_param)
                //   .add("modified_b_input_scale",
                //        std::to_string(modified_b_input_scale))
                //   .add("modified_b_input_zp",
                //   std::to_string(modified_b_input_zp))
                .add("in_dtypes", input_types)
                .add("out_dtypes", output_types)
                .set_anchor_point1(*out_node.node)
                .build();
          }
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
