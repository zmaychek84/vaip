/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */
#include "dtype_util.h"
#include "vaip/vaip.hpp"
#include "vitis/ai/env_config.hpp"
#include <algorithm>
#include <glog/logging.h>
#include <utility>

#include "vaip/dd/coeffs.hpp"
#include "vaip/dd/dd_utils.hpp"
DEF_ENV_PARAM(DEBUG_DD_PATTERN, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_DD_PATTERN) >= n)

namespace {
using namespace vaip_core;

struct DDMergeDtypeInfer {
  DDMergeDtypeInfer(IPass& self) : self_{self} {}

  bool process_qeltwiseadd_dtype_infer(vaip::dtype_util::NodeAttrContext& ctx,
                                       Graph& graph) {
    auto node = ctx.node;
    auto precision = ctx.precision;
    std::map<std::string, std::vector<std::string>> m;

    std::string attr_name = "in_dtypes";
    auto& attrs = node_get_attributes_ref(*node);
    auto attr_proto = node_attributes_get(attrs, attr_name);
    auto strs_value = VAIP_ORT_API(attr_proto_get_strings)(*attr_proto);
    std::vector<std::string> ret = strs_value;
    int parent_index = 0;
    bool swap_inputs = false;
    for (auto x : ctx.parent_ops) {
      if (VAIP_ORT_API(node_op_type)(*x) == "QEltWiseAdd" &&
          parent_index != 0) {
        swap_inputs = true;
      }

      if (VAIP_ORT_API(node_op_type)(*x) == "QEltWiseAdd") {
        auto sibling_nodes =
            vaip::dtype_util::get_all_child_nodes(*ctx.graph, x);
        for (auto y : sibling_nodes) {
          if (y == nullptr)
            continue;
          auto cop = VAIP_ORT_API(node_op_type)(*y);
          if (cop == "QLayerNorm") {
            int input_index = swap_inputs ? 1 - parent_index : parent_index;
            ret[input_index] = "bfloat16";
          }
        }
      }

      parent_index++;
    }
    m["in_dtypes"] = ret;

    attr_name = "out_dtypes";
    attr_proto = node_attributes_get(attrs, attr_name);
    strs_value = VAIP_ORT_API(attr_proto_get_strings)(*attr_proto);
    ret = strs_value;
    ret = {"uint16"};
    for (auto x : ctx.child_op) {
      auto cop = VAIP_ORT_API(node_op_type)(*x);
      if (cop == "QLayerNorm") {
        ret = {"bfloat16"};
      }
    }
    m["out_dtypes"] = ret;

    bool replaced = false;
    auto nab = NodeAttributesBuilder();
    for (const auto& kv : m) {
      if (kv.second.size()) {
        nab.add(kv.first, kv.second);
      }
    }
    auto x = const_cast<Node*>(node);
    nab.merge_into(*x);
    replaced = true;

    if (swap_inputs) {
      std::vector<const NodeArg*> new_inputs =
          node_get_input_node_args(*ctx.node);
      std::swap(new_inputs[0], new_inputs[1]);
      std::swap(new_inputs[2], new_inputs[4]);
      std::swap(new_inputs[3], new_inputs[5]);

      auto node_builder = vaip_core::NodeBuilder(graph, self_);
      node_builder.set_input_node_args(new_inputs);
      node_builder.set_op_type("QEltWiseAdd", "com.xilinx");
      node_builder.clone_attrs(*node);
      node_builder.set_anchor_point1(*node);
      node_builder.build();
    }

    return replaced;
  }

  bool process_qlayernorm_dtype_infer(vaip::dtype_util::NodeAttrContext& ctx) {
    auto node = ctx.node;
    auto precision = ctx.precision;
    std::map<std::string, std::vector<std::string>> m;

    std::string attr_name = "in_dtypes";
    auto& attrs = node_get_attributes_ref(*node);
    auto attr_proto = node_attributes_get(attrs, attr_name);
    auto strs_value = VAIP_ORT_API(attr_proto_get_strings)(*attr_proto);
    std::vector<std::string> ret = strs_value;
    int parent_index = 0;
    for (auto x : ctx.parent_ops) {
      ret[parent_index] = "bfloat16";
      auto p_op = VAIP_ORT_API(node_op_type)(*x);
      if (p_op == "IConv" || p_op == "QGlobalAvgPool" || p_op == "QMatMulAdd") {
        ret[parent_index] = "uint16";
      }
      parent_index++;
    }
    m["in_dtypes"] = ret;

    bool replaced = false;
    auto nab = NodeAttributesBuilder();
    for (const auto& kv : m) {
      if (kv.second.size()) {
        nab.add(kv.first, kv.second);
      }
    }
    auto x = const_cast<Node*>(node);
    nab.merge_into(*x);
    replaced = true;
    return replaced;
  }

  // apply the rule
  void process(IPass& self, Graph& graph) {
    MY_LOG(1) << self_.get_pass_proto().name() << "["
              << self_.get_pass_proto().plugin() << "] start processing graph";
    // create_rule(&self)->apply(&graph);
    MY_LOG(1) << self.get_context()
                     ->xclbin_path_to_cache_files(std::filesystem::path(
                         self_.get_pass_proto().pass_dd_param().xclbin()))
                     .string();
    std::string precision = "a16w8"; // TODO Remove this Hardcoding
    for (const auto node_idx : graph_get_node_in_topoligical_order(graph)) {

      auto node = VAIP_ORT_API(graph_get_node)(graph, node_idx);
      if ("QEltWiseAdd" == VAIP_ORT_API(node_op_type)(*node)) {
        auto node_ctx = vaip::dtype_util::build_context(graph, node, precision);
        if (process_qeltwiseadd_dtype_infer(node_ctx, graph)) {
          MY_LOG(1) << "Changed eltwiseadd attributes";
        }
      }

      if ("QLayerNorm" == VAIP_ORT_API(node_op_type)(*node)) {
        auto node_ctx = vaip::dtype_util::build_context(graph, node, precision);
        if (process_qlayernorm_dtype_infer(node_ctx)) {
          MY_LOG(1) << "Changed layernorm attributes";
        }
      }
    }
    MY_LOG(1) << self_.get_pass_proto().name() << "["
              << self_.get_pass_proto().plugin() << "] finish processing graph";
  }

  IPass& self_;
};
} // namespace

DEFINE_VAIP_PASS(DDMergeDtypeInfer, vaip_pass_dd_merge_dtype_infer)
