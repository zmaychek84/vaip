/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

#include "vaip/vaip.hpp"
#include "vitis/ai/env_config.hpp"
#include <glog/logging.h>
DEF_ENV_PARAM(DEBUG_ADD_TRANSPOSE, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_ADD_TRANSPOSE) >= n)

using namespace vaip_core;
namespace {
struct InputTransOrder {
  size_t index;
  std::vector<int64_t> order;
  bool need_flip_hw = false;
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
                               const std::vector<int64_t>& order,
                               bool need_flip_hw = false) {
  auto pshape = node_arg_get_shape_i64(node_arg);
  CHECK(pshape != nullptr) << node_arg_as_string(node_arg) << " shape absent";
  auto new_shape = trans_shape(*pshape, order);
  auto builder = NodeBuilder(graph, pass);
  AnchorPointTransposeOpAttr attr;
  attr.mutable_order()->Assign(order.begin(), order.end());
  builder.set_op_type("transpose")
      .set_input_node_args({&node_arg})
      .clone_data_type(node_arg)
      .add("order", order)
      .set_anchor_point3(node_arg, {"transpose", attr}, new_shape);
  if (need_flip_hw) {
    // for transposed_conv
    builder.add("flip_hw", (int64_t)1);
  }
  return builder.build();
}

// assume the first node input's transpose order is node's transpose order.
static std::vector<int64_t> guess_node_transpose_order(
    const std::vector<InputTransOrder>& input_trans_orders) {
  auto ret = std::vector<int64_t>();
  for (auto order : input_trans_orders) {
    if (order.index == 0)
      return order.order;
  }
  LOG(FATAL) << "can't guess the node order";
  return ret;
}

static std::vector<int64_t> undo_order(const std::vector<int64_t>& order) {
  std::vector<int64_t> ret;
  ret.resize(order.size());
  for (auto index = 0u; index < order.size(); ++index) {
    ret[order[index]] = index;
  }
  return ret;
}

struct AddTranspose {
  AddTranspose(IPass& self) : self_{self} {}
  void process(IPass& self, Graph& graph) {
    for (auto node_idx : graph_get_node_in_topoligical_order(graph)) {
      auto node = VAIP_ORT_API(graph_get_node)(graph, node_idx);
      CHECK(node != nullptr) << "node_idx " << node_idx << " ";
      process_node(self, graph, *node);
    }
  }
  void process_node(IPass& self, Graph& graph, const Node& node) {
    auto domain = VAIP_ORT_API(node_op_domain)(node);
    if (domain != "com.xilinx") {
      return;
    }
    auto op_type = VAIP_ORT_API(node_op_type)(node);
    // depthwise_conv2d shape has been converted when to_xir_ops
    // onnx:((M, Cin/group, H, W) -> xir:(channel_mutiplier, Cin, kH, kW)) this
    // is {1, 0, 2, 3},
    // so here depthwise_conv2d weights layout same with conv2d is (OIHW),
    // {1, 0, 2, 3} + {0, 2, 3, 1} = {1, 2, 3, 0}
    if (op_type == "conv2d_nchw") {
      node_around_add_transpose(graph, node,
                                {{0, {0, 2, 3, 1}}, {1, {0, 2, 3, 1}}});
    } else if (op_type == "depthwise_conv2d_nchw") {
      // CONV convert to com.ms.internal.nhwc:Conv had convert weights from (M,
      // Cin/group, H, W) to (M, H, W, Cin/group), com.ms.internal.nhwc:Conv
      // convert to depthwise_conv2d weights need layout transform
      // {3, 1, 2, 0}, we can compared to the above CONV convert to
      // depthwise_conv2d_nchw directly. {0, 2, 3, 1} + {3, 1, 2, 0} = {1, 2, 3,
      // 0}.
      node_around_add_transpose(graph, node,
                                {{0, {0, 2, 3, 1}}, {1, {1, 2, 3, 0}}});
    } else if (op_type == "conv1d_ncd") {
      // test case: issue #1143
      node_around_add_transpose(graph, node, {{0, {0, 2, 1}}, {1, {0, 2, 1}}});
    } else if (op_type == "depthwise_conv1d_ncd") {
      node_around_add_transpose(graph, node, {{0, {0, 2, 1}}, {1, {1, 2, 0}}});
    } else if (op_type == "depthwise_conv2d_ihwo") {
      node_weight_add_transpose(graph, node, {3, 1, 2, 0});
    } else if (op_type == "transposed_conv2d_nchw") {
      // model 1
      node_around_add_transpose(graph, node,
                                {{0, {0, 2, 3, 1}}, {1, {1, 2, 3, 0}, true}});

    } else if (op_type == "avgpool2d_nchw" || op_type == "maxpool2d_nchw") {
      node_around_add_transpose(graph, node, {{0, {0, 2, 3, 1}}});
    } else if (op_type == "resize") {
      // xir resize op is layout sensitive, need nhwc format
      auto input_shape =
          *node_arg_get_shape_i64(*node_get_inputs(node)[0].node_arg);
      auto output_shape = node_get_output_shape(node, 0);
      // simple assume the first two dims equal is nchw, maybe lead to new
      // issue.
      bool is_nchw = (input_shape[0] == output_shape[0]) &&
                     (input_shape[1] == output_shape[1]);
      if (is_nchw) {
        node_around_add_transpose(graph, node, {{0, {0, 2, 3, 1}}});
      }
    } else if (op_type == "pixel_shuffle_nchw") {
      node_around_add_transpose(graph, node, {{0, {0, 2, 3, 1}}});
    } else if (op_type == "gstiling_nchw") {
      node_around_add_transpose(graph, node, {{0, {0, 2, 3, 1}}});
    } else if (op_type == "space_to_depth_nchw") {
      node_around_add_transpose(graph, node, {{0, {0, 2, 3, 1}}});
    } else if (op_type == "instancenorm_nchw") {
      node_around_add_transpose(graph, node, {{0, {0, 2, 3, 1}}});
    } else if (op_type == "instancenorm_ncd") {
      node_around_add_transpose(graph, node, {{0, {0, 2, 1}}});
    } else if (op_type == "groupnorm_nchw") {
      node_around_add_transpose(graph, node, {{0, {0, 2, 3, 1}}});
    } else if (op_type == "groupnorm_ncd") {
      node_around_add_transpose(graph, node, {{0, {0, 2, 1}}});
    }
  }

  void node_around_add_transpose(
      Graph& graph, const Node& node,
      const std::vector<InputTransOrder>& input_trans_orders) {
    auto op_type = VAIP_ORT_API(node_op_type)(node);
    std::string new_op_type = "";
    if (op_type == "resize") {
      new_op_type = op_type;
    } else if (op_type.size() >= 5u &&
               op_type.substr(op_type.size() - 5u) == "_nchw") {
      new_op_type = op_type.substr(0u, op_type.size() - 5u);
    } else if (op_type.size() >= 4u &&
               op_type.substr(op_type.size() - 4u) == "_ncd") {
      new_op_type = op_type.substr(0u, op_type.size() - 4u);
    } else {
      LOG(FATAL) << "TODO: ";
    }
    auto origin_inputs = node_get_inputs(node);
    auto origin_output_args = node_get_output_node_args(node);
    auto input_size = origin_inputs.size();
    CHECK_GE(input_size, input_trans_orders.size());
    CHECK_EQ(origin_output_args.size(), 1) << "Only support sigle output now";

    auto new_input_tensors = std::vector<const NodeArg*>();
    new_input_tensors.reserve(input_size);
    for (auto input : origin_inputs) {
      new_input_tensors.push_back(input.node_arg);
    }
    for (auto order : input_trans_orders) {
      CHECK(origin_inputs[order.index].node_arg != nullptr);
      auto& input = *origin_inputs[order.index].node_arg;
      auto& new_input = node_add_transpose(self_, graph, input, order.order,
                                           order.need_flip_hw);
      auto arg_name = node_arg_get_name(input);
      if (self_.has_fix_info(arg_name.c_str())) {
        self_.copy_fix_info(node_arg_get_name(input),
                            node_get_output_name(new_input));
      }
      auto new_input_tensor = node_get_output_node_args(new_input);
      CHECK_EQ(new_input_tensor.size(), 1u);
      new_input_tensors[order.index] = new_input_tensor[0];
    }
    // transpose node NCHW ->  NHWC or NCD  ->  NDC
    auto node_order = guess_node_transpose_order(input_trans_orders);
    auto new_shape = trans_shape(node_get_output_shape(node, 0), node_order);
    // AnchorPoint::NCHW2NHWC; {0, 2, 3, 1}
    // AnchorPoint::NCD2NDC; {0, 2, 1}
    AnchorPointTransposeOpAttr attr;
    attr.mutable_order()->Assign(node_order.begin(), node_order.end());
    auto& node_arg = node_get_output_node_arg(node);
    auto& transpose_node =
        NodeBuilder(graph, self_)
            .set_op_type(new_op_type)
            .set_input_node_args(new_input_tensors)
            .clone_data_type(node)
            .clone_attrs(node)
            .set_anchor_point3(node_arg, {"transpose", attr}, new_shape)
            .build();
    auto order_after = undo_order(node_order);
    NodeBuilder(graph, self_)
        .set_op_type("transpose")
        .set_input_nodes({&transpose_node})
        .add("order", order_after)
        .set_anchor_point1(node)
        .build();
  }

  void node_weight_add_transpose(Graph& graph, const Node& node,
                                 const std::vector<int64_t>& order) {
    auto op_type = VAIP_ORT_API(node_op_type)(node);
    CHECK(op_type.size() >= 5u &&
          op_type.substr(op_type.size() - 5u) == "_ihwo");
    std::string new_op_type = op_type.substr(0u, op_type.size() - 5u);
    auto origin_inputs = node_get_inputs(node);
    auto origin_output_args = node_get_output_node_args(node);
    auto input_size = origin_inputs.size();
    CHECK_GE(input_size, 2);
    CHECK_EQ(origin_output_args.size(), 1) << "Only support sigle output now";

    auto new_inputs = std::vector<const Node*>();
    new_inputs.reserve(input_size);
    for (auto input : origin_inputs) {
      new_inputs.push_back(input.node);
    }
    CHECK(origin_inputs[1].node_arg != nullptr);
    auto& input = *origin_inputs[1].node_arg;
    auto& new_input = node_add_transpose(self_, graph, input, order);
    auto arg_name = node_arg_get_name(input);
    if (self_.has_fix_info(arg_name.c_str())) {
      self_.copy_fix_info(node_arg_get_name(input),
                          node_get_output_name(new_input));
    }
    new_inputs[1] = &new_input;
    NodeBuilder(graph, self_)
        .set_op_type(new_op_type)
        .set_input_nodes(new_inputs)
        .clone_attrs(node)
        .set_anchor_point1(node)
        .build();
  }

  IPass& self_;
};
} // namespace
DEFINE_VAIP_PASS(AddTranspose, vaip_pass_layout_transform_via_adding_transpose)
