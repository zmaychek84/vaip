/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
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
    { "name": "vaip_pass_dd_merge_attentionprepro",
       "plugin": "vaip-pass_dd_merge_attentionprepro",
       "disabled": false
    }
*/
namespace {
using namespace vaip_core;

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
struct Dd_merge_attentionprepro_psw {
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
  Dd_merge_attentionprepro_psw(IPass& self) : self_{self} {}
  std::unique_ptr<Rule> create_rule(IPass* self) {
    auto com_microsoft_QuantizeLinear_3 =
        vaip::pattern_zoo::get_pattern("m_AttentionPrePro_3");
    CHECK(com_microsoft_QuantizeLinear_3 != nullptr)
        << "Pattern returned is null";
    return Rule::create_rule(
        com_microsoft_QuantizeLinear_3,
        [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
          // auto input = binder["input_"];
          auto in_node = binder["input_0"];
          auto out_node = binder["com_microsoft_QuantizeLinear_3"];
          auto mul_const_scale_node = binder["constant_6"];
          auto mul_const_zp_node = binder["constant_7"];
          auto dq_const_sub_node = binder["constant_14"];
          auto dq_sub_const_scale_node = binder["constant_15"];
          auto dq_sub_const_zp_node = binder["constant_16"];
          auto dq_const_mul_node = binder["constant_21"];
          auto dq_mul_const_scale_node = binder["constant_22"];
          auto dq_mul_const_zp_node = binder["constant_23"];
          auto last_mul_const_scale_node = binder["constant_24"];
          auto last_mul_const_zp_node = binder["constant_25"];

          auto node_name = node_arg_get_name(*out_node.node_arg);

          auto mul_const_scale = node_arg_get_const_data_as_float(
              *graph, *mul_const_scale_node.node_arg);
          uint16_t mul_const_zp = node_arg_get_const_data_as_u16(
              *graph, *mul_const_zp_node.node_arg);

          auto dq_mul_const_scale = node_arg_get_const_data_as_float(
              *graph, *dq_mul_const_scale_node.node_arg);
          uint16_t dq_mul_const_zp = node_arg_get_const_data_as_u16(
              *graph, *dq_mul_const_zp_node.node_arg);

          auto dq_const_sub = node_arg_get_const_data_as_u16s(
              *graph, *dq_const_sub_node.node_arg);
          auto dq_sub_const_scale = node_arg_get_const_data_as_float(
              *graph, *dq_sub_const_scale_node.node_arg);
          uint16_t dq_sub_const_zp = node_arg_get_const_data_as_u16(
              *graph, *dq_sub_const_zp_node.node_arg);

          // std::vector<uint16_t> sub_const_vec{dq_sub_const_zp};
          std::vector<uint16_t> sub_vec_in{dq_const_sub[0]};
          auto gamma = vaip::dd::qmatmulcalc::dq_vec_to_bf16(
              sub_vec_in, dq_sub_const_scale, dq_sub_const_zp);
          auto last_mul_const_scale = node_arg_get_const_data_as_float(
              *graph, *last_mul_const_scale_node.node_arg);
          uint16_t last_mul_const_zp = node_arg_get_const_data_as_u16(
              *graph, *last_mul_const_zp_node.node_arg);

          auto dq_const_mul = node_arg_get_const_data_as_u16s(
              *graph, *dq_const_mul_node.node_arg);
          // std::vector<uint16_t> mul_const_vec{dq_mul_const_zp};
          std::vector<uint16_t> mul_vec_in{dq_const_mul[0]};
          auto gamma1 = vaip::dd::qmatmulcalc::dq_vec_to_bf16(
              mul_vec_in, dq_mul_const_scale, dq_mul_const_zp);

          auto bf_mul_const_scale =
              vaip::dd::qmatmulcalc::float_to_bfloat16(mul_const_scale);
          auto bf_last_mul_const_scale =
              vaip::dd::qmatmulcalc::float_to_bfloat16((float)1.0 /
                                                       last_mul_const_scale);

          std::vector<int32_t> lrn_qdq_tensor(16, 0);
          lrn_qdq_tensor[0] = (int32_t)mul_const_zp;
          lrn_qdq_tensor[1] = (int32_t)bf_mul_const_scale;
          lrn_qdq_tensor[2] = (int32_t)last_mul_const_zp;
          lrn_qdq_tensor[3] = (int32_t)bf_last_mul_const_scale;
          lrn_qdq_tensor[4] = (int32_t)gamma[0];
          lrn_qdq_tensor[5] = (int32_t)gamma1[0];

          std::string qdq_name = std::string(node_name + "_qdq_");
          auto& lrn_qdq_arg = vaip::dd::insert_named_tensor_in_graph<int32_t>(
              graph, qdq_name, lrn_qdq_tensor,
              std::vector({(int64_t)lrn_qdq_tensor.size()}));

          std::vector<std::string> input_types{"uint16", "uint16"};
          std::vector<std::string> output_types{"uint16"};

          std::vector<std::string> ns = vaip::dd::get_node_names(graph, binder);
          MY_LOG(1) << "found match at " << ns.front();
          NodeArg* no_op_node_arg = nullptr;

          bool no_op_in_parent = check_no_op_parent(
              *graph, in_node.node, no_op_node_arg, "Unsqueeze");
          if (no_op_in_parent) {
            in_node.node_arg = no_op_node_arg;
          }

          NodeBuilder(*graph, *self)
              .set_input_node_args({in_node.node_arg, &lrn_qdq_arg})
              .set_op_type("AttentionMaskPrePro_win25", "com.xilinx")
              .clone_attrs(*out_node.node)
              .add("nodes", ns)
              .add("in_dtypes", input_types)
              .add("out_dtypes", output_types)
              .add("design_param", "4x4")
              //.add("input_q_params", input_q_params)
              //.add("output_q_params", output_q_params)
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

DEFINE_VAIP_PASS(Dd_merge_attentionprepro_psw,
                 vaip_pass_dd_merge_attentionprepro_psw)
