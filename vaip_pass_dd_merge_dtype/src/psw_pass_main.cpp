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

struct DDMergeShape_psw {
  DDMergeShape_psw(IPass& self) : self_{self} {}

  void update_node_attributes(vaip::dtype_util::NodeAttrContext& ctx) {
    auto node = ctx.node;
    auto node_op = VAIP_ORT_API(node_op_type)(*node);
    auto nab = NodeAttributesBuilder();
    if (node_op == "QLayerNorm") {
      std::vector<std::string> in_dtypes = {"uint16", "uint16", "uint16",
                                            "int32"};
      nab.add("in_dtypes", in_dtypes);
      nab.add("design_param", "4x4");
    } else if (node_op == "QIntEltwiseAdd" || node_op == "QGatherDivAdd") {
      nab.add("kernel_version", "v2");
      std::vector<std::string> out_dtypes = {"uint16"};
      nab.add("out_dtypes", out_dtypes);
      nab.add("design_param", "4x4PSW1.0");
    } else {
      nab.add("design_param", "4x4");
    }

    auto x = const_cast<Node*>(node);
    nab.merge_into(*x);
  }

  void update_qdq_tensor(Graph& graph, const Node* node) {
    auto node_op = VAIP_ORT_API(node_op_type)(*node);
    std::vector<const NodeArg*> node_args = node_get_input_node_args(*node);
    if (node_op == "QLayerNorm") {
      const NodeArg* qdq_node_arg = node_args[node_args.size() - 1];
      auto node_name = node_arg_get_name(*qdq_node_arg);
      auto c_qdq_tensor = node_arg_get_const_data_as_i32s(graph, *qdq_node_arg);

      std::vector<int32_t> qdq_tensor(c_qdq_tensor.begin(), c_qdq_tensor.end());
      // node_get_attr_int
      auto& attrs = node_get_attributes_ref(*node);
      auto attr_proto = node_attributes_get(attrs, "act_zp");
      auto act_zp = (int32_t)VAIP_ORT_API(attr_proto_get_int)(*attr_proto);
      qdq_tensor[5] = 1;
      qdq_tensor[4] = act_zp;

      auto& mqdq_arg = vaip::dd::insert_named_tensor_in_graph(
          &graph, node_name + "_", qdq_tensor,
          std::vector({(int64_t)qdq_tensor.size()}));

      NodeBuilder(graph, self_)
          .set_input_node_args(
              {node_args[0], node_args[1], node_args[2], &mqdq_arg})
          .set_op_type("QLayerNorm", "com.xilinx")
          .clone_attrs(*node)
          .set_anchor_point1(*node)
          .build();
    } else if (node_op == "QEltWiseAdd") {
      const NodeArg* qdq_node_arg = node_args[node_args.size() - 1];
      auto node_name = node_arg_get_name(*qdq_node_arg);
      auto c_qdq_tensor = node_arg_get_const_data_as_i32s(graph, *qdq_node_arg);

      std::vector<int32_t> qdq_tensor(c_qdq_tensor.begin(), c_qdq_tensor.end());
      // node_get_attr_int

      qdq_tensor[7] = 1;

      auto& mqdq_arg = vaip::dd::insert_named_tensor_in_graph(
          &graph, node_name + "_", qdq_tensor,
          std::vector({(int64_t)qdq_tensor.size()}));

      NodeBuilder(graph, self_)
          .set_input_node_args({node_args[0], node_args[1], &mqdq_arg})
          .set_op_type("QEltWiseAdd", "com.xilinx")
          .clone_attrs(*node)
          .set_anchor_point1(*node)
          .build();
    }
  }

  void process(IPass& self, Graph& graph) {
    std::string precision = "a168"; // not used
    for (const auto node_idx : graph_get_node_in_topoligical_order(graph)) {
      auto node = VAIP_ORT_API(graph_get_node)(graph, node_idx);
      auto node_ctx = vaip::dtype_util::build_context(graph, node, precision);
      auto node_op = VAIP_ORT_API(node_op_type)(*node);
      update_node_attributes(node_ctx);
      if (node_op == "QLayerNorm" || node_op == "QEltWiseAdd")
        update_qdq_tensor(graph, node);
    }
  }

  IPass& self_;
};

} // namespace

DEFINE_VAIP_PASS(DDMergeShape_psw, vaip_pass_dd_merge_DDMergeShape_psw)
