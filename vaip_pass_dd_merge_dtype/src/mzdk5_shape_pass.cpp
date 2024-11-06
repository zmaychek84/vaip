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
#include "dtype_util.h"
#include "vaip/vaip.hpp"
#include "vitis/ai/env_config.hpp"
#include <algorithm>
#include <glog/logging.h>
#include <utility>

#include "shape_util.h"
#include "vaip/dd/coeffs.hpp"
#include "vaip/dd/dd_utils.hpp"
DEF_ENV_PARAM(DEBUG_DD_PATTERN, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_DD_PATTERN) >= n)

namespace {
using namespace vaip_core;

struct DDMergeShapemzdk5 {
  DDMergeShapemzdk5(IPass& self) : self_{self} {}
  void add_string_attribute_to_node(vaip::dtype_util::NodeAttrContext& ctx,
                                    std::string attribute_name,
                                    std::vector<std::string> attribute_value) {
    auto node = ctx.node;

    auto nab = NodeAttributesBuilder();
    nab.add(attribute_name, attribute_value);
    auto x = const_cast<Node*>(node);
    nab.merge_into(*x);
  }

  void add_dd_shape(vaip::dtype_util::NodeAttrContext& ctx) {
    auto node = ctx.node;
    auto node_op = VAIP_ORT_API(node_op_type)(*node);
    if (node_op == "IConv") {
      auto first_input_shape = ctx.get_input_shape_at(0);
      auto first_output_shape = ctx.get_output_shape_at(0);
      auto orig_output_shape = vaip::shape_util::shape_as_dd_string(
          vaip::shape_util::same_shape(first_output_shape));
      auto input_shape_str = vaip::shape_util::shape_as_dd_string(
          vaip::shape_util::NCHW_to_NHWC_auto(first_input_shape));
      auto output_shape_str = vaip::shape_util::shape_as_dd_string(
          vaip::shape_util::NCHW_to_NHWC_auto(first_output_shape));
      add_string_attribute_to_node(ctx, "dd_op_in_shape", {input_shape_str});
      add_string_attribute_to_node(ctx, "dd_op_out_shape", {output_shape_str});
      add_string_attribute_to_node(ctx, "orig_output_shape",
                                   {orig_output_shape});
    } else if (node_op == "QGroupNorm") {
      MY_LOG(1) << VAIP_ORT_API(node_get_name)(*node);
      auto first_input_shape = ctx.get_input_shape_at(0);
      auto first_output_shape = ctx.get_output_shape_at(0);
      auto input_shape_str = vaip::shape_util::shape_as_dd_string(
          vaip::shape_util::NCHW_to_NHWC_auto(first_input_shape));
      // MY_LOG(1)<< input_shape_str;
      std::string output_shape_str = "";
      if (first_output_shape.size() == 4) {
        output_shape_str = vaip::shape_util::shape_as_dd_string(
            vaip::shape_util::NCHW_to_NHWC_auto(first_output_shape));
      } else if (first_output_shape.size() == 3)
        output_shape_str = vaip::shape_util::shape_as_dd_string(
            vaip::shape_util::NHC_to_NHC(first_output_shape));
      // MY_LOG(1)<< output_shape_str;
      add_string_attribute_to_node(ctx, "dd_op_in_shape", {input_shape_str});
      add_string_attribute_to_node(ctx, "dd_op_out_shape", {output_shape_str});
    } else if (node_op == "QMatMulAdd" || node_op == "QMatMul" ||
               node_op == "QMatMulAddGelu") {
      auto input_shape0 = ctx.get_input_shape_at(0);
      auto input_shape1 = ctx.get_input_shape_at(1);
      auto output_shape = ctx.get_output_shape_at(0);
      auto input0_shape_str = vaip::shape_util::shape_as_dd_string(
          vaip::shape_util::same_shape(input_shape0));
      auto input1_shape_str = vaip::shape_util::shape_as_dd_string(
          vaip::shape_util::same_shape(input_shape1));
      auto output_shape_str = vaip::shape_util::shape_as_dd_string(
          vaip::shape_util::same_shape(output_shape));
      if (input_shape0[0] == 1 && input_shape0.size() == 3) { // 3D
        int64_t M = input_shape0[1];
        int64_t N = input_shape1[input_shape1.size() - 1];
        output_shape_str = vaip::shape_util::shape_as_dd_string(
            vaip::shape_util::same_shape({1, M, N}));
      } else if (input_shape0[0] == 1 && input_shape0.size() == 2) { // 2D
        int64_t M = input_shape0[input_shape0.size() - 1];
        int64_t N = input_shape1[input_shape1.size() - 1];
        input0_shape_str = vaip::shape_util::shape_as_dd_string(
            vaip::shape_util::same_shape({1, input_shape0[0], M}));
        input1_shape_str = vaip::shape_util::shape_as_dd_string(
            vaip::shape_util::same_shape(input_shape1));
        output_shape_str = vaip::shape_util::shape_as_dd_string(
            vaip::shape_util::same_shape({1, output_shape[0], N}));
      } else if (input_shape0.size() == 4) { // 4D
        auto M = input_shape0[1] * input_shape0[2];
        auto N = input_shape1[input_shape1.size() - 1];
        auto K = input_shape0[input_shape0.size() - 1];
        input0_shape_str = vaip::shape_util::shape_as_dd_string(
            vaip::shape_util::same_shape({1, M, K}));
        input1_shape_str = vaip::shape_util::shape_as_dd_string(
            vaip::shape_util::same_shape(input_shape1));
        output_shape_str = vaip::shape_util::shape_as_dd_string(
            vaip::shape_util::same_shape({1, M, N}));
      } else { // FIXME What case is this?
        auto N = input_shape1[input_shape1.size() - 1];
        output_shape_str =
            vaip::shape_util::shape_as_dd_string(vaip::shape_util::same_shape(
                {input_shape0[0], input_shape0[1], N}));
      }
      add_string_attribute_to_node(ctx, "dd_op_in_shape",
                                   {input0_shape_str, input1_shape_str});
      add_string_attribute_to_node(ctx, "dd_op_out_shape", {output_shape_str});
    } else if (node_op == "mzdk5MHA") {
      auto input_shape0 = ctx.get_input_shape_at(0);
      auto input_shape1 = ctx.get_input_shape_at(1);
      auto input_shape2 = ctx.get_input_shape_at(2);
      auto N = input_shape0[0], C = input_shape0[1], H = input_shape0[2],
           W = input_shape0[3];
      auto input0_shape_str = vaip::shape_util::shape_as_dd_string(
          vaip::shape_util::same_shape({N * C * H, W}));
      N = input_shape1[0], C = input_shape1[1], H = input_shape1[2],
      W = input_shape1[3];
      auto input1_shape_str = vaip::shape_util::shape_as_dd_string(
          vaip::shape_util::same_shape({W, H * N * C}));
      N = input_shape2[0], C = input_shape2[1], H = input_shape2[2],
      W = input_shape2[3];
      auto input2_shape_str = vaip::shape_util::shape_as_dd_string(
          vaip::shape_util::same_shape({N * C * H, W}));
      auto output_shape = ctx.get_output_shape_at(0);
      auto output_shape_str = vaip::shape_util::shape_as_dd_string(
          vaip::shape_util::same_shape(output_shape));
      add_string_attribute_to_node(
          ctx, "dd_op_in_shape",
          {input0_shape_str, input1_shape_str, input2_shape_str});
      add_string_attribute_to_node(ctx, "dd_op_out_shape", {output_shape_str});

    } else if (node_op == "QConv2MatMul") {
      if (node_has_attr(*ctx.node, "from_iconv")) { // From Iconv
        auto input_shape0 = ctx.get_input_shape_at(0);
        MY_LOG(1) << "QConv2Matmul " << ctx.node_name() << ": "
                  << vaip::shape_util::shape_as_dd_string(
                         vaip::shape_util::same_shape(input_shape0));

        if (input_shape0.size() == 4) {
          auto output_shape = ctx.get_output_shape_at(0);
          auto NO = output_shape[0], CO = output_shape[1], HO = output_shape[2],
               WO = output_shape[3];
          auto output_shape_str = vaip::shape_util::shape_as_dd_string(
              vaip::shape_util::same_shape({NO, HO, WO, CO}));
          auto N = input_shape0[0], C = input_shape0[1], H = input_shape0[2],
               W = input_shape0[3];
          auto input0_shape_str = vaip::shape_util::shape_as_dd_string(
              vaip::shape_util::same_shape({N, H, W, C}));
          add_string_attribute_to_node(ctx, "dd_op_in_shape",
                                       {input0_shape_str});
          add_string_attribute_to_node(ctx, "dd_op_out_shape",
                                       {output_shape_str});
        }
      } else if (node_has_attr(*ctx.node, "from_gemmv")) {
        // Don't do anything
      } else { // We have to depend on attribute here.
        std::vector<int64_t> input_shape0 = ctx.get_input_shape_at(0);
        auto nhcw_shape = vaip::shape_util::NCHW_to_NHWC_auto(input_shape0);
        // auto N = input_shape0[0], C = input_shape0[1], H = input_shape0[2], W
        // = input_shape0[3];
        auto input0_shape_str = vaip::shape_util::shape_as_dd_string(
            vaip::shape_util::same_shape(nhcw_shape));
        add_string_attribute_to_node(ctx, "dd_op_in_shape", {input0_shape_str});
        auto output_shape = ctx.get_output_shape_at(0);
        auto output_shape_str = vaip::shape_util::shape_as_dd_string(
            vaip::shape_util::same_shape(output_shape));
        add_string_attribute_to_node(ctx, "dd_op_in_shape", {input0_shape_str});
        add_string_attribute_to_node(ctx, "dd_op_out_shape",
                                     {output_shape_str});
      }
    }
  }
  // apply the rule
  void process(IPass& self, Graph& graph) {
    MY_LOG(1) << self_.get_pass_proto().name() << "["
              << self_.get_pass_proto().plugin() << "] start processing graph";
    // create_rule(&self)->apply(&graph);

    std::string precision = "a16w8"; // TODO Remove this Hardcoding
    for (const auto node_idx : graph_get_node_in_topoligical_order(graph)) {

      auto node = VAIP_ORT_API(graph_get_node)(graph, node_idx);
      auto node_ctx = vaip::dtype_util::build_context(graph, node, precision);
      add_dd_shape(node_ctx);
    }
    MY_LOG(1) << self_.get_pass_proto().name() << "["
              << self_.get_pass_proto().plugin() << "] finish processing graph";
  }

  IPass& self_;
};
} // namespace

DEFINE_VAIP_PASS(DDMergeShapemzdk5, vaip_pass_dd_merge_shape_mzdk5)
