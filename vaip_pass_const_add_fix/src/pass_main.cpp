/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

#include <glog/logging.h>

#include "vaip/vaip.hpp"
#include "vitis/ai/env_config.hpp"
using namespace vaip_core;

/**
    com.xilinx:const() -> com.xilinx:fix(com.xilinx:const())
    where fix_point can find in PassContext
 */

struct ConstAddFixPass {
  ConstAddFixPass(IPass& self) : self_{self} {}
  void process(IPass& self, Graph& graph) {
    for (auto node_idx : graph_get_node_in_topoligical_order(graph)) {
      auto node = VAIP_ORT_API(graph_get_node)(graph, node_idx);
      CHECK(node != nullptr) << "node_idx " << node_idx << " ";

      if (!node_is_op(*node, "const", "com.xilinx")) {
        continue;
      }
      auto name = node_get_output_name(*node);
      auto has_fix_info = self.has_fix_info(name.c_str());
      if (!has_fix_info) {
        continue;
      }
      auto fix_point = self.get_fix_info(name.c_str());

      // TODO: add api is_scalar.
      // auto shape = node_get_output_shape(*node, 0);
      // // scalar no need fix point
      // if (shape.empty()) {
      //   continue;
      // }
      AnchorPointFixAttr attr;
      attr.set_fix_point(fix_point);
      auto& new_const =
          NodeBuilder(graph, self_)
              .clone_node(*node)
              .set_anchor_point2(node_get_output_node_arg(*node), {"fix", attr})
              .build();
      self.create_const_alias(new_const, *node);
      auto round_mode = "DPU_ROUND";
      if (self.get_context()->get_provider_option("xlnx_enable_py3_round") ==
          "1") {
        round_mode = "PY3_ROUND";
      }
      NodeBuilder(graph, self_)
          .set_op_type("fix")
          .set_input_nodes({&new_const})
          .add("fix_point", static_cast<int64_t>(fix_point))
          .add("bit_width", static_cast<int64_t>(8))
          .add("if_signed", static_cast<int64_t>(1))
          .add("round_mode", round_mode)
          .set_anchor_point1(*node)
          .build();
    }
  }
  ~ConstAddFixPass() {}
  IPass& self_;
};

DEFINE_VAIP_PASS(ConstAddFixPass, vaip_pass_const_add_fix)
