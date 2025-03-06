/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */
// #include "./get_merged_attributes.hpp"

// #include "qmha/qmha_processor.hpp"
#include "vaip/dd/coeffs.hpp"
#include "vaip/dd/dd_utils.hpp"
// #include "vaip/pattern_zoo.hpp"
#include "vaip/pattern_zoo.hpp"
#include "vaip/vaip.hpp"
#include "vitis/ai/env_config.hpp"
#include <glog/logging.h>

DEF_ENV_PARAM(DEBUG_DD_MERGE_QL2NORM, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_DD_MERGE_QL2NORM) >= n)

namespace {
using namespace vaip_core;

static std::vector<const Node*>
get_all_child_nodes(const onnxruntime::Graph& graph, const Node* node) {
  std::vector<const Node*> ret;
  for (const auto& output_arg : node_get_output_node_args(*node)) {
    std::string output_name = node_arg_get_name(*output_arg);
    std::vector<const onnxruntime::Node*> consumers =
        graph_get_consumer_nodes(graph, output_name);

    for (const auto consumer_node : consumers) {
      ret.push_back(consumer_node);
    }
  }
  return ret;
}

static bool check_no_op_child(Graph& g, const Node* a,
                              NodeArg*& updated_node_arg,
                              std::string no_op_name) {
  auto next_nodes = get_all_child_nodes(g, a);
  if (next_nodes.size() == 1) {
    auto x = next_nodes[0];
    auto child_op_type = VAIP_ORT_API(node_op_type)(*x);
    if (child_op_type ==
        "DequantizeLinear") { // check if dq --- sqeeze -- q  is found
      next_nodes = get_all_child_nodes(g, x);
      if (next_nodes.size() == 1) {
        auto x = next_nodes[0];

        auto child_op_type = VAIP_ORT_API(node_op_type)(*x);

        if (child_op_type == no_op_name) {
          next_nodes = get_all_child_nodes(g, x);
          // if (next_nodes.size() == 1) {
          auto x = next_nodes[0];
          auto child_op_type = VAIP_ORT_API(node_op_type)(*x);

          if (child_op_type == "QuantizeLinear") {
            auto output_node_args = node_get_output_node_args(*x);
            for (auto ni : output_node_args) {
              if (!node_arg_is_constant(g, *ni)) {
                updated_node_arg = const_cast<NodeArg*>(ni);
                continue;
              }
            }
            return true;
          }
          // }
        }
      }
    }
  }
  return false;
}

struct MergeQLPnorm_3 {
  MergeQLPnorm_3(IPass& self) : self_{self} {}
  std::unique_ptr<Rule> create_rule(IPass* self) {

    auto q2 = vaip::pattern_zoo::get_pattern("m_lpnorm_3");
    CHECK(q2 != nullptr) << "Pattern returned is null";

    return Rule::create_rule(
        q2, [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
          auto attr_nodes = vaip::dd::get_node_names(graph, binder);
          auto a_node = binder["a"];
          auto as_node = binder["a_s"];
          auto az_node = binder["a_z"];
          auto a_sc =
              node_arg_get_const_data_as_float(*graph, *as_node.node_arg);
          auto a_zp = vaip::dd::get_zp_from_node(*graph, *az_node.node_arg);

          auto q2_node = binder["q2"];
          auto q2_s_node = binder["q2_s"];
          auto q2_z_node = binder["q2_z"];
          auto q2_sc =
              node_arg_get_const_data_as_float(*graph, *q2_s_node.node_arg);
          auto q2_zp = vaip::dd::get_zp_from_node(*graph, *q2_z_node.node_arg);
          NodeArg* no_op_node_arg = nullptr;
          bool input_bf16 = false;

          auto args = self->get_pass_proto().args();
          std::string child_node_name = "Squeeze";
          if (!args.empty()) {
            std::vector<std::string> arg;
            for (const auto& str : args) {
              arg.push_back(str);
            }
            if (arg[0] == "bf16")
              input_bf16 = true;
            if (arg.size() == 2)
              child_node_name = arg[1];
          }

          bool no_op_in_parent = check_no_op_child(
              *graph, q2_node.node, no_op_node_arg, child_node_name);
          if (no_op_in_parent) {
            q2_node.node_arg = no_op_node_arg;
          }

          auto dq3_node = binder["dq3"];
          auto b_node = binder["b"];

          auto b_data =
              node_arg_get_const_data_as_u16s(*graph, *b_node.node_arg);

          std::vector<uint16_t> b_data_vec(b_data.begin(), b_data.end());

          auto dq3_s_node = binder["dq3_s"];
          auto dq3_sc =
              node_arg_get_const_data_as_float(*graph, *dq3_s_node.node_arg);

          auto scale_1 = b_data_vec[0] * dq3_sc;

          std::vector<std::string> in_dtypes = {"uint16", "int32"};
          std::vector<std::string> out_dtypes = {"uint16"};

          std::vector<int32_t> qdq_coeffs(16, 0);

          qdq_coeffs[0] =
              vaip::dd::qmatmulcalc::float_to_bfloat16(scale_1 / (q2_sc));
          qdq_coeffs[1] = q2_zp;
          qdq_coeffs[2] = 1; // in_dequant_enable
          qdq_coeffs[3] =
              vaip::dd::qmatmulcalc::float_to_bfloat16(a_sc); // 0;//dq2_zp;
          qdq_coeffs[4] = a_zp;
          std::string design_param = "4x2";
          if (input_bf16) {
            qdq_coeffs[5] = 0;
            in_dtypes[0] = "bfloat16";
            design_param = "4x4";
          } else {
            qdq_coeffs[5] = 1; // out_quant_enable
          }
          auto node_name = node_arg_get_name(*q2_node.node_arg);
          std::vector<std::string> ns = vaip::dd::get_node_names(graph, binder);

          std::string qdq_name = std::string(node_name + "_qdq_");
          auto& qdq_arg = vaip::dd::insert_named_tensor_in_graph<int32_t>(
              graph, qdq_name, qdq_coeffs,
              std::vector({(int64_t)qdq_coeffs.size()}));

          NodeBuilder(*graph, self_)
              .set_op_type("L2_Norm", "com.xilinx")
              .set_input_node_args({a_node.node_arg, &qdq_arg})
              .clone_attrs(*a_node.node)

              .clone_data_type(*q2_node.node_arg)
              .clone_shape(*q2_node.node_arg)
              .set_anchor_point1(*q2_node.node_arg)
              .add("in_dtypes", in_dtypes)
              .add("out_dtypes", out_dtypes)
              .add("design_param", design_param)
              .add("nodes", ns)
              .build();

          return true;
        });
  }
  void process(IPass& self, Graph& graph) { create_rule(&self)->apply(&graph); }

public:
  IPass& self_;
};
} // namespace

DEFINE_VAIP_PASS(MergeQLPnorm_3, vaip_pass_dd_merge_qlpnorm_3)
