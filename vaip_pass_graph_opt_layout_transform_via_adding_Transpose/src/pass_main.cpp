/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

#include "vaip/vaip.hpp"
#include "vitis/ai/env_config.hpp"
#include <glog/logging.h>
DEF_ENV_PARAM(DEBUG_GRAPH_OPT_ADD_TRANSPOSE, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_GRAPH_OPT_ADD_TRANSPOSE) >= n)

using namespace vaip_core;
namespace {
struct InputTransOrder {
  size_t index;
  std::vector<int64_t> order;
};

// static AnchorPoint::Type
// get_anchor_type_by_order(const InputTransOrder& trans_order) {
//   if (trans_order.order == std::vector<int64_t>{0, 2, 3, 1} &&
//       trans_order.index == 0) {
//     return AnchorPoint::NCHW2NHWC;
//   } else if (trans_order.order == std::vector<int64_t>{0, 2, 3, 1}) { //
//   weights
//     return AnchorPoint::OIHW2OHWI;
//   } else if (trans_order.order == std::vector<int64_t>{0, 3, 1, 2}) {
//     return AnchorPoint::NHWC2NCHW;
//   } else if (trans_order.order == std::vector<int64_t>{1, 2, 3, 0}) {
//     return AnchorPoint::IOHW2OHWI;
//   } else {
//     LOG(FATAL) << "no supported transpose order";
//   }
//   return AnchorPoint::NCHW2NHWC;
// }

// static bool is_IOHW2OHWI(const std::vector<int64_t>& order) {
//  return order == std::vector<int64_t>{1, 2, 3, 0};
//}

static std::vector<int64_t> trans_shape(const std::vector<int64_t>& shape,
                                        const std::vector<int64_t>& order) {
  CHECK_EQ(shape.size(), order.size());
  auto ret = std::vector<int64_t>();
  ret.reserve(shape.size());
  for (auto i : order) {
    ret.push_back(shape[i]);
  }
  return ret;
}
const Node& node_add_transpose(IPass& pass, Graph& graph,
                               const NodeArg& node_arg,
                               const std::vector<int64_t>& order) {
  auto pshape = node_arg_get_shape_i64(node_arg);
  CHECK(pshape != nullptr) << node_arg_as_string(node_arg) << " shape absent";
  auto new_shape = trans_shape(*pshape, order);
  auto builder = NodeBuilder(graph, pass);
  AnchorPointTransposeOpAttr attr;
  attr.mutable_order()->Assign(order.begin(), order.end());
  builder.set_op_type("Transpose", "")
      .set_input_node_args({&node_arg})
      .clone_data_type(node_arg)
      .add("perm", order)
      .set_anchor_point3(node_arg, {"Transpose", attr}, new_shape);
  //  if (is_IOHW2OHWI(order)) {
  //    // for transposed_conv
  //    builder.add("flip_hw", (int64_t)1);
  //  }
  return builder.build();
}

struct GraphOptAddTranspose {
  GraphOptAddTranspose(IPass& self) : self_{self} {}
  void process(IPass& self, Graph& graph) {
    for (auto node_idx : graph_get_node_in_topoligical_order(graph)) {
      auto node = VAIP_ORT_API(graph_get_node)(graph, node_idx);
      CHECK(node != nullptr) << "node_idx " << node_idx << " ";
      process_node(self, graph, *node);
    }
  }
  void process_node(IPass& self, Graph& graph, const Node& node) {
    auto op_type = VAIP_ORT_API(node_op_type)(node);
    // depthwise_conv2d shape has been converted when to_xir_ops
    // onnx:((M, Cin/group, H, W) -> xir:(channel_mutiplier, Cin, kH, kW))
    // so here depthwise_conv2d weights layout same with conv2d is (OIHW)
    if (op_type == "Conv") {
      MY_LOG(1) << "this is a Conv. ";
      node_around_add_transpose(graph, node,
                                {{0, {0, 2, 3, 1}}, {1, {0, 2, 3, 1}}});
      //    if (op_type == "conv2d" || op_type == "depthwise_conv2d") {
      //      node_around_add_transpose(graph, node,
      //                                {{0, {0, 2, 3, 1}}, {1, {0, 2, 3, 1}}});
    } else if (op_type == "ConvTranspose") {
      // model 1
      node_around_add_transpose(graph, node,
                                {{0, {0, 2, 3, 1}}, {1, {1, 2, 3, 0}}});

    } else if (op_type == "AveragePool" || op_type == "MaxPool" ||
               op_type == "GlobalAveragePool") {
      node_around_add_transpose(graph, node, {{0, {0, 2, 3, 1}}});
    }
  }

  void node_around_add_transpose(
      Graph& graph, const Node& node,
      const std::vector<InputTransOrder>& input_trans_orders) {
    auto origin_inputs = node_get_inputs(node);
    auto origin_output_args = node_get_output_node_args(node);
    auto input_size = origin_inputs.size();
    CHECK_GE(input_size, input_trans_orders.size());
    CHECK_EQ(origin_output_args.size(), 1) << "Only support sigle output now";

    auto new_inputs = std::vector<const Node*>();
    new_inputs.reserve(input_size);
    for (auto input : origin_inputs) {
      new_inputs.push_back(input.node);
    }
    for (auto order : input_trans_orders) {
      CHECK(origin_inputs[order.index].node_arg != nullptr);
      auto& input = *origin_inputs[order.index].node_arg;
      auto& new_input = node_add_transpose(self_, graph, input, order.order);
      auto arg_name = node_arg_get_name(input);
      if (self_.has_fix_info(arg_name.c_str())) {
        self_.copy_fix_info(node_arg_get_name(input),
                            node_get_output_name(new_input));
      }
      new_inputs[order.index] = &new_input;
    }
    // transpose node NCHW ->  NHWC
    auto new_shape = trans_shape(node_get_output_shape(node, 0),
                                 std::vector<int64_t>{0, 2, 3, 1});
    // AnchorPoint::NCHW2NHWC; {0, 2, 3, 1}
    AnchorPointTransposeOpAttr attr;
    attr.mutable_order()->Add(0);
    attr.mutable_order()->Add(2);
    attr.mutable_order()->Add(3);
    attr.mutable_order()->Add(1);
    auto& node_arg = node_get_output_node_arg(node);
    auto op_type = VAIP_ORT_API(node_op_type)(node);
    std::vector<std::string> custom_op_name = {"ConvTranspose",
                                               "GlobalAveragePool"};
    auto is_custom_op = std::find(custom_op_name.begin(), custom_op_name.end(),
                                  op_type) != custom_op_name.end();
    std::string new_op_type = is_custom_op ? op_type + "_nhwc" : op_type;
    std::string new_domain =
        is_custom_op ? "com.xilinx" : "com.ms.internal.nhwc";
    auto& transpose_node =
        NodeBuilder(graph, self_)
            .set_op_type(new_op_type, new_domain)
            .set_input_nodes(new_inputs)
            .clone_data_type(node)
            .clone_attrs(node) // attr maybe not match with NhwcConv def
            .set_anchor_point3(node_arg, {"Transpose", attr}, new_shape)
            .build();
    NodeBuilder(graph, self_)
        .set_op_type("Transpose", "")
        .set_input_nodes({&transpose_node})
        .add("perm", std::vector<int64_t>{0, 3, 1, 2})
        .set_anchor_point1(node)
        .build();
  }

  IPass& self_;
};
} // namespace
DEFINE_VAIP_PASS(GraphOptAddTranspose,
                 vaip_pass_graph_opt_layout_transform_via_adding_Transpose)
