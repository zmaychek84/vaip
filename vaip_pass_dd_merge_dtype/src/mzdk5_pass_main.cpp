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

#include "vaip/dd/coeffs.hpp"
#include "vaip/dd/dd_utils.hpp"
DEF_ENV_PARAM(DEBUG_DD_PATTERN, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_DD_PATTERN) >= n)

namespace {
using namespace vaip_core;

struct DDMergeShapemzdk5 {
  DDMergeShapemzdk5(IPass& self) : self_{self} {}

  std::vector<std::string>
  get_new_in_dtype_attributes(vaip::dtype_util::NodeAttrContext& ctx) {
    auto node = ctx.node;
    auto precision = ctx.precision;
    auto node_op = VAIP_ORT_API(node_op_type)(*node);
    std::string attr_name = "in_dtypes";
    auto& attrs = node_get_attributes_ref(*node);
    auto attr_proto = node_attributes_get(attrs, attr_name);
    auto strs_value = VAIP_ORT_API(attr_proto_get_strings)(*attr_proto);
    std::vector<std::string> ret = strs_value;

    if (node_op == "QEltWiseAdd") {
      int parent_index = 0;
      for (auto x : ctx.parent_ops) {
        auto parent_op_type = VAIP_ORT_API(node_op_type)(*x);
        if (parent_op_type == "QEltWiseAdd") {
          auto sibling_nodes =
              vaip::dtype_util::get_all_child_nodes(*ctx.graph, x);
          for (auto y : sibling_nodes) {
            if (y == nullptr)
              continue;
            auto cop = VAIP_ORT_API(node_op_type)(*y);
            if (cop == "QLayerNorm") {
              MY_LOG(1) << "Child is " << cop << std::endl;
              ret[parent_index] = "bfloat16";
            }
          }
        }
        parent_index++;
      }
    }
    return ret;
  }

  std::vector<std::string>
  get_new_out_dtype_attributes(vaip::dtype_util::NodeAttrContext& ctx) {
    auto node = ctx.node;
    auto precision = ctx.precision;
    auto node_op = VAIP_ORT_API(node_op_type)(*node);
    auto child_op = ctx.child_op;
    std::string attr_name = "out_dtypes";
    auto& attrs = node_get_attributes_ref(*node);
    auto attr_proto = node_attributes_get(attrs, attr_name);
    auto strs_value = VAIP_ORT_API(attr_proto_get_strings)(*attr_proto);
    std::vector<std::string> ret = strs_value;

    if (node_op == "QGroupNorm") {
      // auto& attrs = node_get_attributes_ref(*node);
      // auto attr_proto = node_attributes_get(attrs, attr_name);
      // auto strs_value = VAIP_ORT_API(attr_proto_get_strings)(*attr_proto);
      ret = {"bfloat16"};
      for (auto x : child_op) {
        // std::cout<<VAIP_ORT_API(node_op_type)(*x)<<" ";
        auto cop = VAIP_ORT_API(node_op_type)(*x);
        if (cop == "QMatMulAdd") {
          MY_LOG(1) << "Child is " << cop << std::endl;
          ret = {"uint16"};
        }
      }

      // std::cout<<"\n";
    } else if (node_op == "QEltWiseAdd") {
      // auto& attrs = node_get_attributes_ref(*node);
      // auto attr_proto = node_attributes_get(attrs, attr_name);
      // auto strs_value = VAIP_ORT_API(attr_proto_get_strings)(*attr_proto);
      ret = {"uint16"};
      for (auto x : child_op) {
        // std::cout<<VAIP_ORT_API(node_op_type)(*x)<<" ";
        auto cop = VAIP_ORT_API(node_op_type)(*x);
        if (cop == "QLayerNorm") {
          MY_LOG(1) << "Child is " << cop << std::endl;
          ret = {"bfloat16"};
        }
      }

      // std::cout<<"\n";
    }
    return ret;
  }

  bool update_node_attributes(vaip::dtype_util::NodeAttrContext& ctx) {
    auto node = ctx.node;
    std::map<std::string, std::vector<std::string>> m;

    m["in_dtypes"] = get_new_in_dtype_attributes(ctx);
    m["out_dtypes"] = get_new_out_dtype_attributes(ctx);
    bool replaced = false;
    auto nab = NodeAttributesBuilder();
    for (const auto& kv : m) {
      if (kv.second.size()) {
        nab.add(kv.first, kv.second);
      }
    }
    nab.add("design_param", "4x4");
    auto x = const_cast<Node*>(node);
    nab.merge_into(*x);
    replaced = true;
    return replaced;
  }
  void update_qdq_tensor(Graph& graph, const Node* node) {
    auto node_op = VAIP_ORT_API(node_op_type)(*node);
    if (node_op == "QGroupNorm") {
      //

      std::vector<const NodeArg*> node_args = node_get_input_node_args(*node);
      const NodeArg* qdq_node_arg = node_args[node_args.size() - 1];
      auto node_name = node_arg_get_name(*qdq_node_arg);
      auto c_qdq_tensor = node_arg_get_const_data_as_i32s(graph, *qdq_node_arg);

      // auto c_ptr = c_qdq_tensor.data();
      // auto ptr = const_cast<int32_t*>(c_ptr);
      std::vector<int32_t> qdq_tensor(c_qdq_tensor.begin(), c_qdq_tensor.end());
      // ptr,
      // ptr +
      //     c_qdq_tensor.size() *
      //         sizeof(
      //             int32_t)); // =
      //                        // const_cast<gsl::span<int32_t>>(c_qdq_tensor);

      auto& attrs = node_get_attributes_ref(*node);
      auto attr_proto = node_attributes_get(attrs, "out_dtypes");
      auto strs_value = VAIP_ORT_API(attr_proto_get_strings)(*attr_proto);
      auto out_dtype = strs_value[0];
      auto attr_proto1 = node_attributes_get(attrs, "in_dtypes");
      auto strs_value1 = VAIP_ORT_API(attr_proto_get_strings)(*attr_proto1);
      auto in_dtype = strs_value1[0];

      if (in_dtype == "bfloat16") {
        qdq_tensor[5] = 0;
        qdq_tensor[4] = 0;
      } else {
        qdq_tensor[5] = 1;
      }
      if (out_dtype == "bfloat16") {
        qdq_tensor[2] = 0;
      } else
        qdq_tensor[2] = 1;

      auto& mqdq_arg = vaip::dd::insert_named_tensor_in_graph(
          &graph, node_name + "_modified_qdq_", qdq_tensor,
          std::vector({(int64_t)qdq_tensor.size()}));

      NodeBuilder(graph, self_)
          .set_input_node_args(
              {node_args[0], node_args[1], node_args[2], &mqdq_arg})
          .set_op_type("QGroupNorm", "com.xilinx")
          .clone_attrs(*node)
          // .add("nodes", ns)
          // .add("epsilon", epsilon)
          // .add("in_dtypes", input_types)
          // .add("out_dtypes", output_types)
          // .add("input_q_params", input_q_params)
          // .add("output_q_params", output_q_params)
          // .add("input_shape", nhwc_shape)
          // .add("output_shape", new_out_shape)
          .set_anchor_point1(*node)
          .build();
    } else if (node_op == "QEltWiseAdd") {
      //

      std::vector<const NodeArg*> node_args = node_get_input_node_args(*node);
      const NodeArg* qdq_node_arg = node_args[node_args.size() - 1];
      auto node_name = node_arg_get_name(*qdq_node_arg);
      auto c_qdq_tensor = node_arg_get_const_data_as_i32s(graph, *qdq_node_arg);

      // auto c_ptr = c_qdq_tensor.data();
      // auto ptr = const_cast<int32_t*>(c_ptr);
      std::vector<int32_t> qdq_tensor(c_qdq_tensor.begin(), c_qdq_tensor.end());
      // ptr,
      // ptr +
      //     c_qdq_tensor.size() *
      //         sizeof(
      //             int32_t)); // =
      //                        // const_cast<gsl::span<int32_t>>(c_qdq_tensor);

      auto& attrs = node_get_attributes_ref(*node);
      auto attr_proto = node_attributes_get(attrs, "out_dtypes");
      auto strs_value = VAIP_ORT_API(attr_proto_get_strings)(*attr_proto);
      auto out_dtype = strs_value[0];
      auto attr_proto1 = node_attributes_get(attrs, "in_dtypes");
      auto strs_value1 = VAIP_ORT_API(attr_proto_get_strings)(*attr_proto1);
      auto in1_dtype = strs_value1[0];
      auto in2_dtype = strs_value1[1];

      if (in1_dtype == "bfloat16") {
        qdq_tensor[6] = 0;
      } else {
        qdq_tensor[6] = 1;
      }
      if (in2_dtype == "bfloat16") {
        qdq_tensor[8] = 0;
      } else {
        qdq_tensor[8] = 1;
      }
      if (out_dtype == "bfloat16") {
        qdq_tensor[7] = 0;
      } else
        qdq_tensor[7] = 1;

      auto& mqdq_arg = vaip::dd::insert_named_tensor_in_graph(
          &graph, node_name + "_modified_qdq_", qdq_tensor,
          std::vector({(int64_t)qdq_tensor.size()}));

      NodeBuilder(graph, self_)
          .set_input_node_args({node_args[0], node_args[1], &mqdq_arg})
          .set_op_type("QEltWiseAdd", "com.xilinx")
          .clone_attrs(*node)
          // .add("nodes", ns)
          // .add("epsilon", epsilon)
          // .add("in_dtypes", input_types)
          // .add("out_dtypes", output_types)
          // .add("input_q_params", input_q_params)
          // .add("output_q_params", output_q_params)
          // .add("input_shape", nhwc_shape)
          // .add("output_shape", new_out_shape)
          .set_anchor_point1(*node)
          .build();
    }
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
      auto node_ctx = vaip::dtype_util::build_context(graph, node, precision);

      auto node_outputs = node_ctx.node_outputs;
      auto output_name = node_arg_get_name(*(node_outputs[0]));
      // excluded ops mzdk5
      if (output_name == "input_1_QuantizeLinear_Output" ||
          output_name == "input_1_channel_first_0_QuantizeLinear_Output" ||
          output_name == "input_2_QuantizeLinear_Output" ||
          output_name == "output_1_channel_first_0_DequantizeLinear_Output" ||
          output_name == "output_1" || output_name == "input_1_q_to_dq") {
        MY_LOG(1) << "excluded op " << output_name;
        continue;
      }
      if (update_node_attributes(node_ctx)) {
        MY_LOG(1) << "Changed out_dtype attribute";
      }
      update_qdq_tensor(graph, node);
    }
    MY_LOG(1) << self_.get_pass_proto().name() << "["
              << self_.get_pass_proto().plugin() << "] finish processing graph";
  }

  IPass& self_;
};
} // namespace

DEFINE_VAIP_PASS(DDMergeShapemzdk5, vaip_pass_dd_merge_dtype_mzdk5)
