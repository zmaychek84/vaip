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

#include "./const_fold_transpose.hpp"

#include <vitis/ai/dim_calc.hpp>

#include "glog/logging.h"

#include "vitis/ai/env_config.hpp"

DEF_ENV_PARAM(DEBUG_CONST_FOLD_TRANSPOSE, "0")
namespace vaip_pass_const_fold_transpose {

ConstFoldTransposeRule::ConstFoldTransposeRule() : Rule() {
  auto builder = PatternBuilder();
  input_weight_ = builder.xir_const_op();
  input_weight_fix_ = builder.node2("com.xilinx:fix", {input_weight_});
  transpose_ = builder.node2("com.xilinx:transpose", {input_weight_fix_});
  transpose_fix_ = builder.node2("com.xilinx:fix", {transpose_});
  input_a_ = builder.wildcard();
  matmul_ = builder.node2("com.xilinx:matmul", {input_a_, transpose_fix_});
}

const Pattern* ConstFoldTransposeRule::pattern() const { return matmul_.get(); }

bool ConstFoldTransposeRule::action(onnxruntime::Graph* graph,
                                    binder_t& binder) const {
  auto input_weight = binder[input_weight_->get_id()];
  auto input_weight_fix = binder[input_weight_fix_->get_id()];
  auto transpose = binder[transpose_->get_id()];
  auto transpose_fix = binder[transpose_fix_->get_id()];
  auto input_a = binder[input_a_->get_id()];
  auto matmul = binder[matmul_->get_id()];

  CHECK(input_weight.node != nullptr);
  CHECK(input_weight_fix.node != nullptr);
  CHECK(transpose.node != nullptr);
  CHECK(transpose_fix.node != nullptr);
  CHECK(input_a.node != nullptr);
  CHECK(matmul.node != nullptr);

  graph_add_node(
      *graph, VAIP_ORT_API(node_get_name)(*matmul.node), "matmul",
      "convert from const_fold_transpose pass.",
      {input_a.node_arg, input_weight_fix.node_arg}, {matmul.node_arg},
      NodeAttributesBuilder()
          .add("transpose_a", (int64_t)0)
          .add("transpose_b", (int64_t)1)
          .add("data_type", node_get_attr_string(*matmul.node, "data_type"))
          .add("shape", node_get_output_shape(*matmul.node, 0))
          .build(),
      "com.xilinx");

  VAIP_ORT_API(graph_remove_node)(*graph, transpose);
  VAIP_ORT_API(graph_remove_node)(*graph, transpose_fix);
  VAIP_ORT_API(graph_remove_node)(*graph, matmul);
  return true;
}
} // namespace vaip_pass_const_fold_transpose
