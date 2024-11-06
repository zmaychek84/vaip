/*
 *     The Xilinx Vitis AI Vaip in this distribution are provided under the
 * following free and permissive binary-only license, but are not provided in
 * source code form.  While the following free and permissive license is similar
 * to the BSD open source license, it is NOT the BSD open source license nor
 * other OSI-approved open source license.
 *
 *      Copyright (C) 2022 Xilinx, Inc. All rights reserved.
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

#include <glog/logging.h>

#include "vaip/vaip.hpp"
#include "vitis/ai/env_config.hpp"
using namespace vaip_core;
DEF_ENV_PARAM(XLNX_ENABLE_PY3_ROUND, "0")

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
      if (ENV_PARAM(XLNX_ENABLE_PY3_ROUND)) {
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
