/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

#include <functional>
#include <glog/logging.h>
#include <numeric>

#include "node_arg.hpp"
#include "vaip/vaip.hpp"
#include "vitis/ai/env_config.hpp"

template <typename T>
void transpose_data(const gsl::span<const int64_t>& shape,
                    const gsl::span<const int64_t>& perm,
                    const gsl::span<const T>& data, bool flip_hw, T* output);
extern template void transpose_data<char>(const gsl::span<const int64_t>&,
                                          const gsl::span<const int64_t>&,
                                          const gsl::span<const char>&, bool,
                                          char*);
extern template void transpose_data<float>(const gsl::span<const int64_t>&,
                                           const gsl::span<const int64_t>&,
                                           const gsl::span<const float>&, bool,
                                           float*);
extern template void transpose_data<uint16_t>(const gsl::span<const int64_t>&,
                                              const gsl::span<const int64_t>&,
                                              const gsl::span<const uint16_t>&,
                                              bool, uint16_t*);
extern template void transpose_data<int16_t>(const gsl::span<const int64_t>&,
                                             const gsl::span<const int64_t>&,
                                             const gsl::span<const int16_t>&,
                                             bool, int16_t*);
DEF_ENV_PARAM(DEBUG_FUSE_TRANSPOSE, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_FUSE_TRANSPOSE) >= n)

using namespace vaip_core;
namespace {

static std::vector<int64_t> undo_order(const gsl::span<const int64_t>& order) {
  std::vector<int64_t> ret;
  ret.resize(order.size());
  for (auto index = 0u; index < order.size(); ++index) {
    ret[order[index]] = index;
  }
  return ret;
}

static std::vector<int64_t>
undo_transpose_shape(const gsl::span<const int64_t>& shape,
                     const gsl::span<const int64_t>& perm) {
  auto ret = std::vector<int64_t>(shape.size());
  CHECK_EQ(perm.size(), shape.size());
  for (auto i = 0u; i < perm.size(); ++i) {
    ret[perm[i]] = shape[i];
  }
  return ret;
}

static int64_t
layout_transform_axis_alike_attr(int64_t value,
                                 const std::vector<int64_t>& order) {
  int64_t ret = 0;
  if (value >= 0) {
    ret = order[value];
  } else {
    ret = order[order.size() + value];
  }
  return ret;
}

static std::vector<int64_t>
layout_transform_vector_axis_alike_attr(const gsl::span<const int64_t>& value,
                                        const gsl::span<const int64_t>& order) {
  auto ret = std::vector<int64_t>(value.size());
  CHECK_LE(value.size(), order.size());
  for (auto i = 0u; i < value.size(); ++i) {
    if (value[i] >= 0) {
      ret[i] = order[value[i]];
    } else {
      ret[i] = order[order.size() + value[i]];
    }
  }
  return ret;
}

static std::vector<int64_t> expand_shape(const std::vector<int64_t>& shape,
                                         size_t size) {
  auto expand_shape = std::vector<int64_t>();
  CHECK_LE(shape.size(), size);
  if (shape.size() < size) {
    for (auto i = 0u; i < size - shape.size(); i++) {
      expand_shape.push_back(1);
    }
    for (auto dim : shape) {
      expand_shape.push_back(dim);
    }
  } else {
    expand_shape = std::vector<int64_t>(shape.begin(), shape.end());
  }
  return expand_shape;
}

static bool change_paddings(NodeBuilder& builder, const Node& transpose_node,
                            const Node& siso_node) {
  auto order = node_get_attr_ints(transpose_node, "order");
  auto padding = node_get_attr_ints(siso_node, "paddings");
  CHECK_EQ(order.size() * 2, padding.size());
  auto new_padding = std::vector<int64_t>(padding.size());
  for (auto i = 0u; i < order.size(); i++) {
    new_padding[2 * order[i]] = padding[2 * i];
    new_padding[2 * order[i] + 1] = padding[2 * i + 1];
  }
  builder.add("paddings", new_padding);
  return true;
}
static bool change_begin_end_step(NodeBuilder& builder,
                                  const Node& transpose_node,
                                  const Node& siso_node) {
  auto order = node_get_attr_ints(transpose_node, "order");
  auto attrs = std::vector<std::string>{"begin", "end", "strides"};
  for (auto attr : attrs) {
    auto attr_value = node_get_attr_ints(siso_node, attr);
    auto new_attr_value = std::vector<int64_t>();
    new_attr_value.resize(attr_value.size());
    // CHECK
    for (auto i = 0u; i < order.size(); i++) {
      new_attr_value[order[i]] = attr_value[i];
    }
    builder.add(attr, new_attr_value);
  }
  return true;
}

static bool change_axis(NodeBuilder& builder, const Node& transpose_node,
                        const Node& siso_node) {
  auto order_i = node_get_attr_ints(transpose_node, "order");
  if (node_has_attr(siso_node, "axis")) {
    auto axis = node_get_attr_int(siso_node, "axis");
    auto order = std::vector<int64_t>();
    order.assign(order_i.begin(), order_i.end());
    auto new_axis = layout_transform_axis_alike_attr(axis, order);
    builder.add("axis", new_axis);
  }
  return true;
}

static bool change_vector_axis(NodeBuilder& builder, const Node& transpose_node,
                               const Node& siso_node) {
  auto order = node_get_attr_ints(transpose_node, "order");
  auto axis = node_get_attr_ints(siso_node, "axis");
  auto new_axis = layout_transform_vector_axis_alike_attr(axis, order);
  builder.add("axis", new_axis);
  return true;
}

static std::unique_ptr<Rule> create_siso_rule(
    IPass* self, const std::string& op_type,
    const std::vector<
        std::function<bool(NodeBuilder& builder, const Node& transpose_node,
                           const Node& siso_node)>>&& actions = {}) {
  auto builder = PatternBuilder();
  std::shared_ptr<Pattern> pat_x = builder.wildcard();
  std::shared_ptr<Pattern> pat_transpose =
      builder.node2("com.xilinx:transpose", {pat_x});
  std::shared_ptr<Pattern> pat_siso_op =
      builder.node2(op_type, {pat_transpose});
  return Rule::create_rule(
      pat_siso_op, [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
        // auto ni_x = binder[pat_x->get_id()];
        auto ni_transpose = binder[pat_transpose->get_id()];
        auto ni_siso_op = binder[pat_siso_op->get_id()];
        /// siso_op(transpose(x)) => transpose(siso_op(x))
        auto siso_builder = NodeBuilder(*graph, *self);

        siso_builder.clone_inputs(*ni_transpose.node)
            .clone_op_type(*ni_siso_op.node)
            .clone_data_type(*ni_siso_op.node)
            .clone_attrs(*ni_siso_op.node);
        for (auto& action : actions) {
          auto ret = action(siso_builder, *ni_transpose.node, *ni_siso_op.node);
          if (!ret)
            return false;
        }

        // undo_transpose_shape() maybe LOG(FATAL), so move this after check the
        // above actions(such as check_keep_dims).
        auto origin_siso_shape = node_get_output_shape(*ni_siso_op.node, 0);
        auto order = node_get_attr_ints(*ni_transpose.node, "order");
        auto siso_shape = undo_transpose_shape(origin_siso_shape, order);
        siso_builder.set_shape(siso_shape);

        auto new_order = undo_order(order);
        AnchorPointTransposeOpAttr attr;
        attr.mutable_order()->Assign(new_order.begin(), new_order.end());

        auto& new_siso_op =
            siso_builder
                .set_anchor_point2(*ni_siso_op.node_arg, {"transpose", attr})
                .build();

        NodeBuilder(*graph, *self)
            .set_input_nodes({&new_siso_op})
            .set_op_type("transpose")
            .clone_attrs(*ni_transpose.node)
            .set_anchor_point1(*ni_siso_op.node)
            .build();
        return true;
      });
}
static std::unique_ptr<Rule> create_qdq_rule(IPass* self,
                                             const std::string& op_type) {
  auto builder = PatternBuilder();
  std::shared_ptr<Pattern> pat_x = builder.wildcard();
  std::shared_ptr<Pattern> pat_transpose =
      builder.node2("com.xilinx:transpose", {pat_x});
  std::shared_ptr<Pattern> pat_scale = builder.wildcard();
  std::shared_ptr<Pattern> pat_zero_point = builder.wildcard();
  std::shared_ptr<Pattern> pat_qdq_op =
      builder.node3(op_type, {pat_transpose, pat_scale, pat_zero_point},
                    {false, false, true});
  return Rule::create_rule(
      pat_qdq_op, [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
        auto ni_x = binder[pat_x->get_id()];
        auto ni_transpose = binder[pat_transpose->get_id()];
        auto ni_qdq_op = binder[pat_qdq_op->get_id()];
        auto ni_scale = binder[pat_scale->get_id()];
        auto ni_zero_point = binder[pat_zero_point->get_id()];

        auto qdq_builder = NodeBuilder(*graph, *self);

        if (ni_zero_point.node_arg == nullptr) {
          qdq_builder.set_input_node_args({ni_x.node_arg, ni_scale.node_arg});
        } else {
          qdq_builder.set_input_node_args(
              {ni_x.node_arg, ni_scale.node_arg, ni_zero_point.node_arg});
        }
        qdq_builder.clone_op_type(*ni_qdq_op.node)
            .clone_data_type(*ni_qdq_op.node)
            .clone_attrs(*ni_qdq_op.node);

        auto origin_qdq_shape = node_get_output_shape(*ni_qdq_op.node, 0);
        auto order = node_get_attr_ints(*ni_transpose.node, "order");
        auto qdq_shape = undo_transpose_shape(origin_qdq_shape, order);
        qdq_builder.set_shape(qdq_shape);

        auto new_order = undo_order(order);
        AnchorPointTransposeOpAttr attr;
        attr.mutable_order()->Assign(new_order.begin(), new_order.end());

        auto& new_qdq_op =
            qdq_builder
                .set_anchor_point2(*ni_qdq_op.node_arg, {"transpose", attr})
                .build();

        NodeBuilder(*graph, *self)
            .set_input_nodes({&new_qdq_op})
            .set_op_type("transpose")
            .clone_attrs(*ni_transpose.node)
            .set_anchor_point1(*ni_qdq_op.node)
            .build();
        return true;
      });
}

static std::unique_ptr<Rule>
create_broadcast_op_rule(IPass* self, const std::string& op_type) {
  auto builder = PatternBuilder();
  std::shared_ptr<Pattern> pat_x = builder.wildcard();
  std::shared_ptr<Pattern> pat_y = builder.wildcard();
  std::shared_ptr<Pattern> pat_tr_x =
      builder.node2("com.xilinx:transpose", {pat_x});
  std::shared_ptr<Pattern> pat_tr_y =
      builder.node2("com.xilinx:transpose", {pat_y});
  std::shared_ptr<Pattern> pat_add =
      builder.node2(std::string("com.xilinx:") + op_type, {pat_tr_x, pat_tr_y});
  return Rule::create_rule(
      pat_add, [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
        auto ni_x = binder[pat_x->get_id()];
        auto ni_y = binder[pat_y->get_id()];
        auto ni_tr_x = binder[pat_tr_x->get_id()];
        auto ni_tr_y = binder[pat_tr_y->get_id()];
        auto ni_add = binder[pat_add->get_id()];
        auto order_x = node_get_attr_ints(*ni_tr_x.node, "order");
        auto order_y = node_get_attr_ints(*ni_tr_y.node, "order");
        if (order_x != order_y) {
          return false;
        }
        auto order = std::vector<int64_t>(order_x.begin(), order_x.end());
        auto pshape = node_arg_get_shape_i64(*ni_add.node_arg);
        CHECK(pshape != nullptr)
            << node_arg_as_string(*ni_add.node_arg) << " shape absent";
        auto new_shape = undo_transpose_shape(*pshape, order);
        // NCHW2NHWC {0, 2, 3, 1}
        AnchorPointTransposeOpAttr attr;
        attr.mutable_order()->Add(0);
        attr.mutable_order()->Add(2);
        attr.mutable_order()->Add(3);
        attr.mutable_order()->Add(1);

        NodeBuilder(*graph, *self)
            .set_input_nodes(
                {//
                 &NodeBuilder(*graph, *self)
                      .set_input_node_args({ni_x.node_arg, ni_y.node_arg})
                      .clone_op_type(*ni_add.node)
                      .clone_data_type(*ni_add.node)
                      .set_shape(new_shape)
                      .clone_attrs(*ni_add.node)
                      .set_anchor_point2(*ni_add.node_arg, {"transpose", attr})
                      .build()})
            .set_op_type("transpose")
            .add("order", order)
            .set_anchor_point1(*ni_add.node)
            .build();
        return true;
      });
}
template <typename From> struct AsSpan {
  AsSpan(From& other) : other_{other} {}
  From other_;
  template <typename To> gsl::span<To> as() {
    CHECK_EQ(other_.size_bytes() % sizeof(From::value_type), 0);
    return gsl::span<To>(reinterpret_cast<To*>(other_.data()),
                         other_.size_bytes() / sizeof(From::value_type));
  }
};
template <typename From> AsSpan<From> as_span(const From& from) {
  return AsSpan(from);
}

static std::unique_ptr<Rule>
create_broadcast_op_const_rule(IPass* self, const std::string& op_type) {
  auto builder = PatternBuilder();
  std::shared_ptr<Pattern> pat_x = builder.wildcard();
  std::shared_ptr<Pattern> pat_y = builder.xir_const_op();
  std::shared_ptr<Pattern> pat_tr_x =
      builder.node2("com.xilinx:transpose", {pat_x});
  std::shared_ptr<Pattern> pat_add1 =
      builder.node2(std::string("com.xilinx:") + op_type, {pat_tr_x, pat_y});
  std::shared_ptr<Pattern> pat_add2 =
      builder.node2(std::string("com.xilinx:") + op_type, {pat_y, pat_tr_x});
  std::shared_ptr<Pattern> pat_add = builder.Or({pat_add1, pat_add2});
  return Rule::create_rule(
      pat_add, [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
        auto ni_x = binder[pat_x->get_id()];
        auto ni_y = binder[pat_y->get_id()];
        auto ni_tr_x = binder[pat_tr_x->get_id()];
        auto ni_add = binder[pat_add->get_id()];
        auto ni_add1 = binder[pat_add1->get_id()];
        auto ni_add2 = binder[pat_add2->get_id()];
        auto order_x = node_get_attr_ints(*ni_tr_x.node, "order");
        auto order = std::vector<int64_t>(order_x.begin(), order_x.end());
        auto pshape = node_arg_get_shape_i64(*ni_add.node_arg);
        CHECK(pshape != nullptr)
            << node_arg_as_string(*ni_add.node_arg) << " shape absent";
        auto new_shape = undo_transpose_shape(*pshape, order);

        auto new_order = undo_order(order);
        AnchorPointTransposeOpAttr attr;
        attr.mutable_order()->Assign(new_order.begin(), new_order.end());

        auto pyshape = node_arg_get_shape_i64(*ni_y.node_arg);
        CHECK(pyshape != nullptr)
            << node_arg_as_string(*ni_y.node_arg) << " shape absent";
        auto y_shape = *pyshape;
        // usually order.size()==4 now, if y_shape.size()>order.size(), maybe
        // y_shape.size() is 5 or above, so we need to expand ni_x shape, not
        // expand y_shape. our dpu don't support 5-dims or above ops,
        // so we deal with this as don't support now and maybe support this in
        // future.
        if (y_shape.size() > order.size()) {
          return false; // don't support now
        }
        auto y_expand_shape = expand_shape(y_shape, order.size());
        auto data_type = node_get_attr_string(*ni_y.node, "data_type");
        CHECK_EQ(data_type, "float32");
        auto y_new_shape = undo_transpose_shape(y_expand_shape, order);
        auto& new_ni_y = NodeBuilder(*graph, *self)
                             .clone_inputs(*ni_y.node)
                             .clone_op_type(*ni_y.node)
                             .clone_data_type(*ni_y.node_arg)
                             .set_shape(y_new_shape)
                             .set_anchor_point2(*ni_y.node_arg, {"const"})
                             .build();
        auto size = (size_t)std::accumulate(
            y_new_shape.begin(), y_new_shape.end(), 1ull, std::multiplies());
        auto old_name = node_get_output_name(*ni_y.node);
        // test case: model xilinxSR
        if (self->has_fix_info(old_name.c_str())) {
          self->copy_fix_info(*ni_y.node, new_ni_y);
        }
        self->create_lazy_const(
            new_ni_y, size * sizeof(float),
            [self, y_expand_shape, new_order, old_name](gsl::span<char> data) {
              auto input_data = self->get_const_data<float>(old_name.c_str());
              transpose_data<float>(y_expand_shape, new_order, input_data,
                                    false,
                                    reinterpret_cast<float*>(data.data()));
            });
        auto input_args = std::vector<const NodeArg*>();
        input_args.reserve(2u);
        // need keep the input order, otherwise maybe lead to error result, such
        // as sub op, a-b not equal to b-a;
        // fix JIRA-6521
        if (ni_add1.node_arg != nullptr && ni_add2.node_arg == nullptr) {
          input_args.push_back(ni_x.node_arg);
          input_args.push_back(&(node_get_output_node_arg(new_ni_y)));
        } else if (ni_add1.node_arg == nullptr && ni_add2.node_arg != nullptr) {
          input_args.push_back(&(node_get_output_node_arg(new_ni_y)));
          input_args.push_back(ni_x.node_arg);
        } else {
          return false;
        }
        NodeBuilder(*graph, *self)
            .set_input_nodes(
                {//
                 &NodeBuilder(*graph, *self)
                      .set_input_node_args(input_args)
                      .clone_op_type(*ni_add.node)
                      .clone_data_type(*ni_add.node)
                      .clone_attrs(*ni_add.node)
                      .set_shape(new_shape)
                      .set_anchor_point2(*ni_add.node_arg, {"transpose", attr})
                      .build()})
            .set_op_type("transpose")
            .add("order", order)
            .set_anchor_point1(*ni_add.node)
            .build();
        return true;
      });
} // namespace

static bool is_tranpose_immune(const std::vector<int64_t>& shape) {
  auto ret = false;
  if (shape.empty()) {
    ret = true;
  } else if (shape.size() == 1) {
    ret = true;
  } else {
    auto n_of_non_ones = 0;
    for (auto s : shape) {
      if (s != 1u) {
        ++n_of_non_ones;
      }
    }
    ret = (n_of_non_ones == 1) || (n_of_non_ones == 0);
  }
  return ret;
}

static std::unique_ptr<Rule>
create_broadcast_op_transpose_immune_rule(IPass* self,
                                          const std::string& op_type) {
  auto builder = PatternBuilder();
  std::shared_ptr<Pattern> pat_x = builder.wildcard();
  std::shared_ptr<Pattern> pat_y = builder.wildcard();
  std::shared_ptr<Pattern> pat_tr_x =
      builder.node2("com.xilinx:transpose", {pat_x});
  std::shared_ptr<Pattern> pat_add1 =
      builder.node2(std::string("com.xilinx:") + op_type, {pat_tr_x, pat_y});
  std::shared_ptr<Pattern> pat_add2 =
      builder.node2(std::string("com.xilinx:") + op_type, {pat_y, pat_tr_x});
  std::shared_ptr<Pattern> pat_add = builder.Or({pat_add1, pat_add2});
  return Rule::create_rule(
      pat_add, [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
        auto ni_x = binder[pat_x->get_id()];
        auto ni_y = binder[pat_y->get_id()];
        auto ni_tr_x = binder[pat_tr_x->get_id()];
        auto ni_add = binder[pat_add->get_id()];
        auto ni_add1 = binder[pat_add1->get_id()];
        auto ni_add2 = binder[pat_add2->get_id()];
        auto order_x = node_get_attr_ints(*ni_tr_x.node, "order");
        auto pyshape = node_arg_get_shape_i64(*ni_y.node_arg);
        CHECK(pyshape != nullptr)
            << node_arg_as_string(*ni_y.node_arg) << " shape absent";
        auto y_shape = *pyshape;
        if (!is_tranpose_immune(y_shape)) {
          return false;
        }
        auto order = std::vector<int64_t>(order_x.begin(), order_x.end());
        auto pshape = node_arg_get_shape_i64(*ni_add.node_arg);
        CHECK(pshape != nullptr)
            << node_arg_as_string(*ni_add.node_arg) << " shape absent";
        auto new_shape = undo_transpose_shape(*pshape, order);

        auto new_order = undo_order(order);
        AnchorPointTransposeOpAttr attr;
        attr.mutable_order()->Assign(new_order.begin(), new_order.end());
        // usually order.size()==4 now, if y_shape.size()>order.size(), maybe
        // y_shape.size() is 5 or above, so we need to expand ni_x shape, not
        // expand y_shape. our dpu don't support 5-dims or above ops,
        // so we deal with this as don't support now and maybe support this in
        // future.
        if (y_shape.size() > order.size()) {
          return false; // don't support now
        }
        auto y_expand_shape = expand_shape(y_shape, order.size());
        auto y_new_shape = undo_transpose_shape(y_expand_shape, order);
        auto& new_ni_y =
            NodeBuilder(*graph, *self)
                .set_input_node_args({ni_y.node_arg})
                .set_op_type("reshape")
                .clone_data_type(*ni_y.node_arg)
                .set_shape(y_new_shape)
                .set_anchor_point2(*ni_y.node_arg, {"transpose_immune"})
                .build();
        auto input_args = std::vector<const NodeArg*>();
        input_args.reserve(2u);
        // need keep the input order, otherwise maybe lead to error result, such
        // as sub op, a-b not equal to b-a;
        if (ni_add1.node_arg != nullptr && ni_add2.node_arg == nullptr) {
          input_args.push_back(ni_x.node_arg);
          input_args.push_back(&(node_get_output_node_arg(new_ni_y)));
        } else if (ni_add1.node_arg == nullptr && ni_add2.node_arg != nullptr) {
          input_args.push_back(&(node_get_output_node_arg(new_ni_y)));
          input_args.push_back(ni_x.node_arg);
        } else {
          return false;
        }
        NodeBuilder(*graph, *self)
            .set_input_nodes(
                {//
                 &NodeBuilder(*graph, *self)
                      .set_input_node_args(input_args)
                      .clone_op_type(*ni_add.node)
                      .clone_data_type(*ni_add.node)
                      .set_shape(new_shape)
                      .clone_attrs(*ni_add.node)
                      .set_anchor_point2(*ni_add.node_arg, {"transpose", attr})
                      .build()})
            .set_op_type("transpose")
            .add("order", order)
            .set_anchor_point1(*ni_add.node)
            .build();
        return true;
      });
} // namespace

static std::unique_ptr<Rule> create_expand_op_rule(IPass* self) {
  auto builder = PatternBuilder();
  std::shared_ptr<Pattern> pat_input = builder.wildcard();
  std::shared_ptr<Pattern> pat_shape = builder.xir_const_op();
  std::shared_ptr<Pattern> pat_tr_input =
      builder.node2("com.xilinx:transpose", {pat_input});
  std::shared_ptr<Pattern> pat_expand = builder.node2(
      std::string("com.xilinx:expand"), {pat_tr_input, pat_shape});
  return Rule::create_rule(
      pat_expand, [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
        auto ni_input = binder[pat_input->get_id()];
        auto ni_shape = binder[pat_shape->get_id()];
        auto ni_tr_input = binder[pat_tr_input->get_id()];
        auto ni_expand = binder[pat_expand->get_id()];

        auto input_shape = node_arg_get_shape_i64(*ni_input.node_arg);
        auto expand_shape = node_arg_get_shape_i64(*ni_expand.node_arg);
        auto input_order = node_get_attr_ints(*ni_tr_input.node, "order");

        auto shape_name = node_get_output_name(*ni_shape.node);
        auto shape_data = self->get_const_data<int64_t>(shape_name.c_str());
        std::vector<int64_t> aligned_shape_data;
        if (input_shape->size() != input_order.size()) {
          LOG(INFO) << "input_shape->size() != input_order.size(), "
                    << input_shape->size() << " vs " << input_order.size();
          return false;
        }

        // onnx allow input > shape , so we insert 1 to the start of shape until
        // input.size() == shape.size()
        if (input_shape->size() > shape_data.size()) {
          auto redundant = input_shape->size() - shape_data.size();
          for (auto i = 0u; i < redundant; i++) {
            aligned_shape_data.push_back(1);
          }
        }
        aligned_shape_data.insert(aligned_shape_data.end(), shape_data.begin(),
                                  shape_data.end());
        CHECK(aligned_shape_data.size() >= input_shape->size())
            << aligned_shape_data.size() << " vs " << input_shape->size();
        auto redundant = aligned_shape_data.size() - input_shape->size();
        /**
         * for input.shape = [1,3], transpose = 1,0 ,shape.shape = [4]
         * so redundant = 2
         * new_order = [0,1,3,2]
         */
        std::vector<int64_t> new_order;

        for (auto i = 0u; i < redundant; i++) {
          new_order.push_back(i);
        }
        for (const auto& perm : input_order) {
          new_order.push_back(perm + redundant);
        }

        AnchorPointTransposeOpAttr attr;
        attr.mutable_order()->Assign(new_order.begin(), new_order.end());

        // undo transpose for shape's data
        auto new_shape_data = undo_transpose_shape(
            gsl::span<int64_t>(aligned_shape_data), new_order);
        auto new_shape_expand =
            undo_transpose_shape(gsl::span<int64_t>(*expand_shape), new_order);
        std::vector<int64_t> new_shape_shape = {
            static_cast<int64_t>(new_shape_data.size())};
        auto& new_ni_shape =
            NodeBuilder(*graph, *self)
                .clone_inputs(*ni_shape.node)
                .clone_op_type(*ni_shape.node)
                .clone_data_type(*ni_shape.node_arg)
                .set_shape(new_shape_shape)
                .set_anchor_point2(*ni_shape.node_arg, {"const"})
                .build();
        if (self->has_fix_info(shape_name.c_str())) {
          self->copy_fix_info(*ni_shape.node, new_ni_shape);
        }
        self->create_lazy_const(
            new_ni_shape, new_shape_data.size() * sizeof(int64_t),
            [new_shape_data](gsl::span<char> data) {
              for (auto i = 0u; i < new_shape_data.size(); i++) {
                reinterpret_cast<int64_t*>(data.data())[i] = new_shape_data[i];
              }
            });

        NodeBuilder(*graph, *self)
            .set_input_nodes(
                {//
                 &NodeBuilder(*graph, *self)
                      .set_input_node_args(
                          {ni_input.node_arg,
                           &(node_get_output_node_arg(new_ni_shape))})
                      .clone_op_type(*ni_expand.node)
                      .clone_data_type(*ni_expand.node)
                      .set_shape(new_shape_expand)
                      .clone_attrs(*ni_expand.node)
                      .set_anchor_point2(*ni_expand.node_arg,
                                         {"transpose", attr})
                      .build()})
            .set_op_type("transpose")
            .add("order", new_order)
            .set_anchor_point1(*ni_expand.node)
            .build();
        return true;
      });
}

static std::unique_ptr<Rule> create_tile_op_rule(IPass* self) {
  auto builder = PatternBuilder();
  std::shared_ptr<Pattern> pat_input = builder.wildcard();
  std::shared_ptr<Pattern> pat_repeats = builder.xir_const_op();
  std::shared_ptr<Pattern> pat_tr_input =
      builder.node2("com.xilinx:transpose", {pat_input});
  std::shared_ptr<Pattern> pat_tile = builder.node2(
      std::string("com.xilinx:broadcast_tile"), {pat_tr_input, pat_repeats});
  return Rule::create_rule(
      pat_tile, [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
        auto ni_input = binder[pat_input->get_id()];
        auto ni_repeats = binder[pat_repeats->get_id()];
        auto ni_tr_input = binder[pat_tr_input->get_id()];
        auto ni_tile = binder[pat_tile->get_id()];

        auto input_shape = node_arg_get_shape_i64(*ni_input.node_arg);
        auto tile_shape = node_arg_get_shape_i64(*ni_tile.node_arg);
        auto input_order = node_get_attr_ints(*ni_tr_input.node, "order");

        auto repeats_name = node_get_output_name(*ni_repeats.node);
        auto repeats_data = self->get_const_data<int64_t>(repeats_name.c_str());
        if (input_shape->size() != input_order.size()) {
          LOG(INFO) << "input_shape->size() != input_order.size(), "
                    << input_shape->size() << " vs " << input_order.size();
          return false;
        }
        // onnx tile requirement
        CHECK(repeats_data.size() == input_shape->size())
            << repeats_data.size() << " vs " << input_shape->size();

        std::vector<int64_t> new_order;
        new_order.assign(input_order.begin(), input_order.end());

        AnchorPointTransposeOpAttr attr;
        attr.mutable_order()->Assign(new_order.begin(), new_order.end());

        // undo transpose for repeats' data
        auto new_repeats_data =
            undo_transpose_shape(gsl::span<int64_t>(repeats_data), new_order);
        auto new_shape_tile =
            undo_transpose_shape(gsl::span<int64_t>(*tile_shape), new_order);
        std::vector<int64_t> new_repeats_shape = {
            static_cast<int64_t>(new_repeats_data.size())};
        auto& new_ni_repeats =
            NodeBuilder(*graph, *self)
                .clone_inputs(*ni_repeats.node)
                .clone_op_type(*ni_repeats.node)
                .clone_data_type(*ni_repeats.node_arg)
                .set_shape(new_repeats_shape)
                .set_anchor_point2(*ni_repeats.node_arg, {"const"})
                .build();
        if (self->has_fix_info(repeats_name.c_str())) {
          self->copy_fix_info(*ni_repeats.node, new_ni_repeats);
        }
        self->create_lazy_const(
            new_ni_repeats, new_repeats_data.size() * sizeof(int64_t),
            [new_repeats_data](gsl::span<char> data) {
              for (auto i = 0u; i < new_repeats_data.size(); i++) {
                reinterpret_cast<int64_t*>(data.data())[i] =
                    new_repeats_data[i];
              }
            });

        NodeBuilder(*graph, *self)
            .set_input_nodes(
                {//
                 &NodeBuilder(*graph, *self)
                      .set_input_node_args(
                          {ni_input.node_arg,
                           &(node_get_output_node_arg(new_ni_repeats))})
                      .clone_op_type(*ni_tile.node)
                      .clone_data_type(*ni_tile.node)
                      .set_shape(new_shape_tile)
                      .clone_attrs(*ni_tile.node)
                      .set_anchor_point2(*ni_tile.node_arg, {"transpose", attr})
                      .build()})
            .set_op_type("transpose")
            .add("order", new_order)
            .set_anchor_point1(*ni_tile.node)
            .build();
        return true;
      });
}

static std::unique_ptr<Rule> create_transpose_transpose_rule(IPass* self) {
  auto builder = PatternBuilder();
  std::shared_ptr<Pattern> pat_t1 =
      builder.node2("com.xilinx:transpose", {builder.wildcard()});
  std::shared_ptr<Pattern> pat_t2 =
      builder.node2("com.xilinx:transpose", {pat_t1});
  return Rule::create_rule(
      pat_t2, [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
        auto ni_t1 = binder[pat_t1->get_id()];
        auto ni_t2 = binder[pat_t2->get_id()];
        auto order1 = node_get_attr_ints(*ni_t1.node, "order");
        auto order2 = node_get_attr_ints(*ni_t2.node, "order");
        /// y = tr1(x); z = tr2(y)
        /// y[i] = x[order1[i]], for i = 0...N
        /// z[i] = y[order2[i]], for i = 0...N
        /// so
        /// z[i] = x[order1[order2[i]]
        /// so order3[i] = order1[order2[i]] , for i = 0..N
        /// z = tr3(x)
        auto order3 = std::vector<int64_t>(order1.size());
        CHECK_EQ(order1.size(), order2.size());
        for (auto i = 0u; i < order1.size(); ++i) {
          order3[i] = order1[order2[i]];
        }
        NodeBuilder(*graph, *self)
            .clone_inputs(*ni_t1.node)
            .clone_op_type(*ni_t1.node)
            .add("order", order3)
            .set_anchor_point1(*ni_t2.node)
            .build();
        return true;
      });
}

static std::unique_ptr<Rule> create_transpose_const_rule(IPass* pass) {
  auto builder = PatternBuilder();
  std::shared_ptr<Pattern> pat_const = builder.xir_const_op();
  std::shared_ptr<Pattern> pat_transpose =
      builder.node2("com.xilinx:transpose", {pat_const});
  return Rule::create_rule(
      pat_transpose, [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
        auto ni_transpose = binder[pat_transpose->get_id()];
        auto ni_const = binder[pat_const->get_id()];
        auto shape_span = node_get_attr_ints(*ni_const.node, "shape");
        auto shape = std::vector<int64_t>(shape_span.begin(), shape_span.end());
        auto data_type = node_get_attr_string(*ni_const.node, "data_type");
        CHECK_EQ(data_type, "float32");
        auto perm_span = node_get_attr_ints(*ni_transpose.node, "order");
        auto perm = std::vector<int64_t>(perm_span.begin(), perm_span.end());
        auto flip_hw =
            node_get_attr_int_with_default(*ni_transpose.node, "flip_hw", 0);
        auto& new_node = NodeBuilder(*graph, *pass)
                             .clone_inputs(*ni_const.node)
                             .clone_op_type(*ni_const.node)
                             .set_anchor_point1(*ni_transpose.node)
                             .build();
        auto size = (size_t)std::accumulate(shape.begin(), shape.end(), 1ull,
                                            std::multiplies());
        auto old_name = node_get_output_name(*ni_const.node);
        if (pass->has_fix_info(old_name.c_str())) {
          pass->copy_fix_info(*ni_const.node, new_node);
        }
        pass->create_lazy_const(
            new_node, size * sizeof(float),
            [pass, shape, perm, flip_hw, old_name](gsl::span<char> data) {
              auto input_data = pass->get_const_data<float>(old_name.c_str());
              transpose_data<float>(shape, perm, input_data, flip_hw == 1,
                                    reinterpret_cast<float*>(data.data()));
            });
        return true;
      });
}
static std::unique_ptr<Rule> create_const_dq_transpose_rule(IPass* pass) {
  auto builder = PatternBuilder();
  std::shared_ptr<Pattern> pat_x = builder.xir_const_op();
  std::shared_ptr<Pattern> pat_scale = builder.xir_const_op();
  std::shared_ptr<Pattern> pat_zero_point = builder.xir_const_op();
  std::shared_ptr<Pattern> pat_dq =
      builder.node3("com.xilinx:dequantize_linear",
                    {pat_x, pat_scale, pat_zero_point}, {false, false, true});
  std::shared_ptr<Pattern> pat_transpose =
      builder.node2("com.xilinx:transpose", {pat_dq});
  return Rule::create_rule(
      pat_transpose, [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
        auto ni_transpose = binder[pat_transpose->get_id()];
        auto ni_dq = binder[pat_dq->get_id()];
        auto ni_x = binder[pat_x->get_id()];
        auto ni_scale = binder[pat_scale->get_id()];
        auto ni_zero_point = binder[pat_zero_point->get_id()];

        auto x_shape_span = node_get_attr_ints(*ni_x.node, "shape");
        auto x_shape =
            std::vector<int64_t>(x_shape_span.begin(), x_shape_span.end());

        auto perm_span = node_get_attr_ints(*ni_transpose.node, "order");
        auto perm = std::vector<int64_t>(perm_span.begin(), perm_span.end());

        auto flip_hw =
            node_get_attr_int_with_default(*ni_transpose.node, "flip_hw", 0);

        AnchorPointTransposeOpAttr attr;
        attr.mutable_order()->Assign(perm.begin(), perm.end());

        auto new_shape = node_arg_get_shape_i64(*ni_transpose.node_arg);

        auto& new_x =
            NodeBuilder(*graph, *pass)
                .clone_inputs(*ni_x.node)
                .clone_op_type(*ni_x.node)
                .clone_data_type(*ni_x.node)
                .clone_attrs(*ni_x.node)
                .set_anchor_point3(*ni_x.node_arg, {"transpose", attr},
                                   *(new_shape.get()))
                .build();
        auto dq_builder = NodeBuilder(*graph, *pass);
        if (ni_zero_point.node == nullptr) {
          dq_builder.set_input_nodes({&new_x, ni_scale.node});
        } else {
          dq_builder.set_input_nodes(
              {&new_x, ni_scale.node, ni_zero_point.node});
        }
        dq_builder.clone_op_type(*ni_dq.node)
            .clone_attrs(*ni_dq.node)
            .set_anchor_point1(*ni_transpose.node)
            .build();

        auto size = (size_t)std::accumulate(x_shape.begin(), x_shape.end(),
                                            1ull, std::multiplies());

        auto old_name = node_get_output_name(*ni_x.node);
        auto data_type = node_arg_get_element_type(*ni_x.node_arg);
        if (data_type == onnx::TensorProto_DataType_UINT16) {
          pass->create_lazy_const(
              new_x, size * sizeof(uint16_t),
              [pass, x_shape, perm, flip_hw, old_name](gsl::span<char> data) {
                auto input_data =
                    pass->get_const_data<uint16_t>(old_name.c_str());
                transpose_data<uint16_t>(
                    x_shape, perm, input_data, flip_hw == 1,
                    reinterpret_cast<uint16_t*>(data.data()));
              });
        } else if (data_type == onnx::TensorProto_DataType_INT16) {
          pass->create_lazy_const(
              new_x, size * sizeof(int16_t),
              [pass, x_shape, perm, flip_hw, old_name](gsl::span<char> data) {
                auto input_data =
                    pass->get_const_data<int16_t>(old_name.c_str());
                transpose_data<int16_t>(
                    x_shape, perm, input_data, flip_hw == 1,
                    reinterpret_cast<int16_t*>(data.data()));
              });
        } else if (data_type == onnx::TensorProto_DataType_UINT8 ||
                   data_type == onnx::TensorProto_DataType_INT8) {
          // create_lazy_const int8/uint8
          pass->create_lazy_const(
              new_x, size * sizeof(char),
              [pass, x_shape, perm, flip_hw, old_name](gsl::span<char> data) {
                auto input_data = pass->get_const_data<char>(old_name.c_str());
                transpose_data<char>(x_shape, perm, input_data, flip_hw == 1,
                                     reinterpret_cast<char*>(data.data()));
              });
        } else {
          LOG(WARNING) << "cancel fuse transpose with dequantize_linear, not "
                          "supported data_type "
                       << data_type;
          return false;
        }
        return true;
      });
}

// test case modelzoo #1 #26
static std::unique_ptr<Rule> create_concat_rule(IPass* pass) {
  auto builder = PatternBuilder();
  std::shared_ptr<Pattern> pat_concat = builder.wildcard();
  return Rule::create_rule(
      pat_concat, [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
        auto ni_concat = binder[pat_concat->get_id()];
        if (ni_concat.node == nullptr) {
          return false;
        }
        if (!node_is_op(*ni_concat.node, "concat", "com.xilinx")) {
          return false;
        }
        auto concat_inputs = node_get_inputs(*ni_concat.node);
        auto at_least_one_transpose = 0u;
        auto all_transpose_order_is_same = true;
        auto order = std::vector<int64_t>();
        // concat inputs must be const or transpose, and at least one
        // transpose. model #26 SOLO.
        for (auto concat_input : concat_inputs) {
          auto node_is_transpose =
              concat_input.node &&
              node_is_op(*concat_input.node, "transpose", "com.xilinx");
          auto node_is_const =
              concat_input.node &&
              node_is_op(*concat_input.node, "const", "com.xilinx");
          if (!(node_is_transpose || node_is_const)) {
            return false;
          }
          if (node_is_transpose) {
            at_least_one_transpose++;
            if (order.empty()) {
              auto first_order =
                  node_get_attr_ints(*concat_input.node, "order");
              order.assign(first_order.begin(), first_order.end());
            } else {
              // all inputs must have same transpose(order:=...);
              all_transpose_order_is_same =
                  all_transpose_order_is_same &&
                  node_get_attr_ints(*concat_input.node, "order") ==
                      gsl::span<const int64_t>(order);
            }
          }
        }
        if (!((at_least_one_transpose > 0u) && all_transpose_order_is_same)) {
          return false;
        }
        auto new_order = undo_order(order);
        auto input_args = std::vector<const NodeArg*>();
        input_args.reserve(concat_inputs.size());
        for (auto concat_input : concat_inputs) {
          auto node_is_const =
              concat_input.node &&
              node_is_op(*concat_input.node, "const", "com.xilinx");
          if (node_is_const) {
            auto shape_span = node_get_attr_ints(*concat_input.node, "shape");
            auto shape =
                std::vector<int64_t>(shape_span.begin(), shape_span.end());
            auto data_type =
                node_get_attr_string(*concat_input.node, "data_type");
            CHECK_EQ(data_type, "float32");
            auto new_shape = undo_transpose_shape(shape, order);
            auto& new_node =
                NodeBuilder(*graph, *pass)
                    .clone_inputs(*concat_input.node)
                    .clone_op_type(*concat_input.node)
                    .clone_data_type(*concat_input.node_arg)
                    .set_shape(new_shape)
                    .set_anchor_point2(*concat_input.node_arg, {"const"})
                    .build();
            auto size = (size_t)std::accumulate(
                new_shape.begin(), new_shape.end(), 1ull, std::multiplies());
            auto old_name = node_get_output_name(*concat_input.node);
            if (pass->has_fix_info(old_name.c_str())) {
              pass->copy_fix_info(*concat_input.node, new_node);
            }
            pass->create_lazy_const(
                new_node, size * sizeof(float),
                [pass, shape, new_order, old_name](gsl::span<char> data) {
                  auto input_data =
                      pass->get_const_data<float>(old_name.c_str());
                  transpose_data<float>(shape, new_order, input_data, false,
                                        reinterpret_cast<float*>(data.data()));
                });
            input_args.push_back(&((node_get_output_node_arg)(new_node)));
          } else {
            auto transpose_inputs =
                node_get_input_node_args(*concat_input.node);
            CHECK_EQ(transpose_inputs.size(), 1u);
            input_args.push_back(transpose_inputs[0]);
          }
        }
        auto new_axis = layout_transform_axis_alike_attr(
            node_get_attr_int(*ni_concat.node, "axis"), order);
        auto shape = node_get_output_shape(*ni_concat.node, 0);
        auto new_shape = undo_transpose_shape(shape, order);
        // NCHW2NHWC {0, 2, 3, 1}
        AnchorPointTransposeOpAttr attr;
        attr.mutable_order()->Add(0);
        attr.mutable_order()->Add(2);
        attr.mutable_order()->Add(3);
        attr.mutable_order()->Add(1);

        NodeBuilder(*graph, *pass)
            .set_input_nodes({&NodeBuilder(*graph, *pass)
                                   .set_input_node_args(input_args)
                                   .set_op_type("concat")
                                   .add("axis", new_axis)
                                   .set_shape(new_shape)
                                   .clone_data_type(*ni_concat.node_arg)
                                   .set_anchor_point2(*ni_concat.node_arg,
                                                      {"transpose", attr})
                                   .build()})
            .set_op_type("transpose")
            .add("order", order)
            .set_anchor_point1(*ni_concat.node)
            .build();
        return true;
      });
}

static std::unique_ptr<Rule> create_batchnorm_rule(IPass* self) {
  auto builder = PatternBuilder();
  std::shared_ptr<Pattern> pat_input = builder.wildcard();
  std::shared_ptr<Pattern> pat_transpose =
      builder.node2("com.xilinx:transpose", {pat_input});
  std::shared_ptr<Pattern> pat_gamma = builder.xir_const_op();
  std::shared_ptr<Pattern> pat_beta = builder.xir_const_op();
  std::shared_ptr<Pattern> pat_moving_mean = builder.xir_const_op();
  std::shared_ptr<Pattern> pat_moving_var = builder.xir_const_op();
  std::shared_ptr<Pattern> pat_batchnorm =
      builder.node2("com.xilinx:batchnorm", {pat_transpose, pat_gamma, pat_beta,
                                             pat_moving_mean, pat_moving_var});
  return Rule::create_rule(
      pat_batchnorm, [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
        auto ni_input = binder[pat_input->get_id()];
        auto ni_transpose = binder[pat_transpose->get_id()];
        auto ni_gamma = binder[pat_gamma->get_id()];
        auto ni_beta = binder[pat_beta->get_id()];
        auto ni_moving_mean = binder[pat_moving_mean->get_id()];
        auto ni_moving_var = binder[pat_moving_var->get_id()];
        auto ni_batchnorm = binder[pat_batchnorm->get_id()];

        auto origin_batchnorm_shape =
            node_get_output_shape(*ni_batchnorm.node, 0);
        auto order = node_get_attr_ints(*ni_transpose.node, "order");
        if (origin_batchnorm_shape.size() != order.size()) {
          return false;
        }
        auto new_axis = order[node_get_attr_int(*ni_batchnorm.node, "axis")];
        auto batchnorm_shape =
            undo_transpose_shape(origin_batchnorm_shape, order);

        CHECK(ni_gamma.node != nullptr);
        CHECK(ni_beta.node != nullptr);
        CHECK(ni_moving_mean.node != nullptr);
        CHECK(ni_moving_var.node != nullptr);

        auto new_order = undo_order(order);
        AnchorPointTransposeOpAttr attr;
        attr.mutable_order()->Assign(new_order.begin(), new_order.end());
        auto& new_batchnorm =
            NodeBuilder(*graph, *self)
                .clone_op_type(*ni_batchnorm.node)

                .set_input_node_args({ni_input.node_arg, ni_gamma.node_arg,
                                      ni_beta.node_arg, ni_moving_mean.node_arg,
                                      ni_moving_var.node_arg})
                .clone_data_type(*ni_batchnorm.node)
                .set_shape(batchnorm_shape)
                .clone_attrs(*ni_batchnorm.node)
                .add("axis", new_axis)
                .set_anchor_point2(*ni_batchnorm.node_arg, {"transpose", attr})
                .build();

        NodeBuilder(*graph, *self)
            .set_input_nodes({&new_batchnorm})
            .set_op_type("transpose")
            .clone_attrs(*ni_transpose.node)
            .set_anchor_point1(*ni_batchnorm.node)
            .build();

        return true;
      });
}

static std::unique_ptr<Rule> create_shape_squeeze_siso_rule(
    IPass* self,
    const std::vector<
        std::function<bool(NodeBuilder& builder, const Node& transpose_node,
                           const Node& siso_node)>>&& actions = {}) {
  auto builder = PatternBuilder();
  std::shared_ptr<Pattern> pat_transpose =
      builder.node2("com.xilinx:transpose", {builder.wildcard()});
  std::shared_ptr<Pattern> pat_siso_op =
      builder.node2(std::string("com.xilinx:squeeze"), {pat_transpose});
  return Rule::create_rule(
      pat_siso_op, [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
        auto ni_transpose = binder[pat_transpose->get_id()];
        auto ni_siso_op = binder[pat_siso_op->get_id()];

        auto origin_siso_shape = node_get_output_shape(*ni_siso_op.node, 0);
        auto order = node_get_attr_ints(*ni_transpose.node, "order");
        auto origin_axis = node_get_attr_ints(*ni_siso_op.node, "axis");
        auto new_axis =
            layout_transform_vector_axis_alike_attr(origin_axis, order);

        auto squeeze_order = std::vector<int64_t>();
        auto calc_new_order = [new_axis](int64_t idx) {
          auto new_idx = idx;
          for (auto&& ax : new_axis)
            if (idx > ax)
              new_idx--;
          return new_idx;
        };
        for (size_t i = 0; i < order.size(); i++) {
          if (std::find(new_axis.begin(), new_axis.end(), order[i]) ==
              new_axis.end()) {
            squeeze_order.push_back(calc_new_order(order[i]));
          }
        }

        auto siso_shape =
            undo_transpose_shape(origin_siso_shape, squeeze_order);
        auto new_order = undo_order(squeeze_order);

        AnchorPointTransposeOpAttr attr;
        attr.mutable_order()->Assign(new_order.begin(), new_order.end());

        auto siso_builder = NodeBuilder(*graph, *self);
        auto& new_siso_op =
            siso_builder.clone_inputs(*ni_transpose.node)
                .clone_op_type(*ni_siso_op.node)
                .clone_data_type(*ni_siso_op.node)
                .clone_attrs(*ni_siso_op.node)
                .set_shape(siso_shape)
                .add("axis", new_axis)
                .set_anchor_point2(*ni_siso_op.node_arg, {"transpose", attr})
                .build();

        NodeBuilder(*graph, *self)
            .set_input_nodes({&new_siso_op})
            .set_op_type("transpose")
            .clone_attrs(*ni_transpose.node)
            .add("order", squeeze_order)
            .set_anchor_point1(*ni_siso_op.node)
            .build();
        return true;
      });
}

static std::unique_ptr<Rule> create_shape_reduce_siso_rule(
    IPass* self, const std::string& op_type,
    const std::vector<
        std::function<bool(NodeBuilder& builder, const Node& transpose_node,
                           const Node& siso_node)>>&& actions = {}) {
  auto builder = PatternBuilder();
  std::shared_ptr<Pattern> pat_x = builder.wildcard();
  std::shared_ptr<Pattern> pat_transpose =
      builder.node2("com.xilinx:transpose", {pat_x});
  std::shared_ptr<Pattern> pat_siso_op =
      builder.node2(std::string("com.xilinx:") + op_type, {pat_transpose});
  return Rule::create_rule(
      pat_siso_op, [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
        // auto ni_x = binder[pat_x->get_id()];
        auto ni_transpose = binder[pat_transpose->get_id()];
        auto ni_siso_op = binder[pat_siso_op->get_id()];
        if (node_arg_is_scalar(*ni_siso_op.node_arg)) {
          return false;
        }
        auto origin_siso_shape = node_get_output_shape(*ni_siso_op.node, 0);
        auto order = node_get_attr_ints(*ni_transpose.node, "order");
        int64_t keep_dims = 0;
        auto origin_axis = node_get_attr_ints(*ni_siso_op.node, "axis");
        auto new_axis =
            layout_transform_vector_axis_alike_attr(origin_axis, order);
        if (node_has_attr(*ni_siso_op.node, "keep_dims")) {
          keep_dims = node_get_attr_int(*ni_siso_op.node, "keep_dims");
        }

        auto reduce_order = std::vector<int64_t>();

        if (!keep_dims) {
          reduce_order.reserve(order.size() - origin_axis.size());
          auto sorted_origin_axis = std::vector<std::int64_t>();
          auto reduce_order_tmp = std::vector<int64_t>();
          for (auto i = 0u; i < order.size(); ++i) {
            if (std::find(origin_axis.begin(), origin_axis.end(), i) ==
                origin_axis.end()) {
              reduce_order_tmp.push_back(order[i]);
            } else {
              sorted_origin_axis.push_back(order[i]);
            }
          }
          std::sort(sorted_origin_axis.begin(), sorted_origin_axis.end());
          for (size_t i = 0; i < reduce_order_tmp.size(); i++) {
            auto pos =
                std::lower_bound(sorted_origin_axis.begin(),
                                 sorted_origin_axis.end(), reduce_order_tmp[i]);
            auto dis = std::distance(sorted_origin_axis.begin(), pos);
            reduce_order.push_back(reduce_order_tmp[i] - dis);
          }
        } else {
          reduce_order.assign(order.begin(), order.end());
        }
        auto siso_shape = undo_transpose_shape(origin_siso_shape, reduce_order);
        auto new_order = undo_order(reduce_order);

        AnchorPointTransposeOpAttr attr;
        attr.mutable_order()->Assign(new_order.begin(), new_order.end());

        /// siso_op(transpose(x)) => transpose(siso_op(x))
        auto siso_builder = NodeBuilder(*graph, *self);
        auto& new_siso_op =
            siso_builder.clone_inputs(*ni_transpose.node)
                .clone_op_type(*ni_siso_op.node)
                .clone_data_type(*ni_siso_op.node)
                .clone_attrs(*ni_siso_op.node)
                .set_shape(siso_shape)
                .add("axis", new_axis)
                .set_anchor_point2(*ni_siso_op.node_arg, {"transpose", attr})
                .build();

        NodeBuilder(*graph, *self)
            .set_input_nodes({&new_siso_op})
            .set_op_type("transpose")
            .clone_attrs(*ni_transpose.node)
            .add("order", reduce_order)
            .set_anchor_point1(*ni_siso_op.node)
            .build();
        return true;
      });
}

static std::unique_ptr<Rule> create_reshape_rule(IPass* self) {
  auto builder = PatternBuilder();
  std::shared_ptr<Pattern> pat_x = builder.wildcard();
  std::shared_ptr<Pattern> pat_transpose =
      builder.node2("com.xilinx:transpose", {pat_x});
  std::shared_ptr<Pattern> pat_reshape_op =
      builder.node2(std::string("com.xilinx:reshape"), {pat_transpose});
  return Rule::create_rule(
      pat_reshape_op, [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
        // auto ni_x = binder[pat_x->get_id()];
        auto ni_transpose = binder[pat_transpose->get_id()];
        auto ni_reshape_op = binder[pat_reshape_op->get_id()];

        auto origin_input_shape = node_get_output_shape(*ni_transpose.node, 0);
        auto origin_output_shape =
            node_get_output_shape(*ni_reshape_op.node, 0);
        auto guess_result =
            guess_reshape(origin_input_shape, origin_output_shape);
        bool is_possible_sink =
            guess_result.size() == origin_input_shape.size();
        if (is_possible_sink) {
          for (auto i = 0u; i < origin_input_shape.size(); ++i) {
            is_possible_sink =
                is_possible_sink && guess_result[i].first.size() == 1;
            if (is_possible_sink) {
              CHECK(guess_result[i].first[0] == i)
                  << "guess_result[i].first[0] need equal to i";
            }
          }
        }

        if (!is_possible_sink) {
          return false;
        }
        auto order = node_get_attr_ints(*ni_transpose.node, "order");
        auto size = std::vector<int64_t>(guess_result.size());
        for (auto i = 0u; i < guess_result.size(); ++i) {
          size[i] = (int64_t)guess_result[i].second.size();
        }
        auto undo_size = undo_transpose_shape(size, order);
        auto tmp_order = std::vector<std::vector<int64_t>>(guess_result.size());
        int64_t order_index = 0;
        for (auto i = 0u; i < undo_size.size(); ++i) {
          for (auto j = 0u; j < undo_size[i]; j++) {
            tmp_order[i].push_back(order_index);
            order_index++;
          }
        }
        auto increase_order = std::vector<int64_t>();
        for (auto i : order) {
          for (auto j : tmp_order[i]) {
            increase_order.push_back(j);
          }
        }

        auto reshape_shape =
            undo_transpose_shape(origin_output_shape, increase_order);
        auto new_order = undo_order(increase_order);

        AnchorPointTransposeOpAttr attr;
        attr.mutable_order()->Assign(new_order.begin(), new_order.end());

        auto reshape_builder = NodeBuilder(*graph, *self);
        auto& new_reshape_op =
            reshape_builder.clone_inputs(*ni_transpose.node)
                .clone_op_type(*ni_reshape_op.node)
                .clone_data_type(*ni_reshape_op.node)
                .clone_attrs(*ni_reshape_op.node)
                .set_shape(reshape_shape)
                .set_anchor_point2(*ni_reshape_op.node_arg, {"transpose", attr})
                .build();

        NodeBuilder(*graph, *self)
            .set_input_nodes({&new_reshape_op})
            .set_op_type("transpose")
            .clone_attrs(*ni_transpose.node)
            .add("order", increase_order)
            .set_anchor_point1(*ni_reshape_op.node)
            .build();
        return true;
      });
}

struct FuseTranspose {
  FuseTranspose(IPass& self) {}
  void process(IPass& self, Graph& graph) {
    std::unique_ptr<BaseRule> rules[] = {
        create_siso_rule(&self, "com.xilinx:gelu"),
        create_siso_rule(&self, "com.xilinx:cast"),                   //
        create_siso_rule(&self, "com.xilinx:relu"),                   //
        create_siso_rule(&self, "com.xilinx:relu6"),                  //
        create_siso_rule(&self, "com.xilinx:float2fix"),              //
        create_siso_rule(&self, "com.xilinx:fix2float"),              //
        create_qdq_rule(&self, "com.xilinx:quantize_linear_int8"),    //
        create_qdq_rule(&self, "com.xilinx:quantize_linear_uint8"),   //
        create_qdq_rule(&self, "com.xilinx:dequantize_linear_int8"),  //
        create_qdq_rule(&self, "com.xilinx:dequantize_linear_uint8"), //
        create_qdq_rule(&self, "com.xilinx:quantize_linear"),         //
        create_qdq_rule(&self, "com.xilinx:dequantize_linear"),       //
        create_siso_rule(&self, "com.xilinx:fix",
                         {change_axis}), // model resnet18_float
        create_siso_rule(&self, "com.xilinx:leaky_relu"),   // model 15
        create_siso_rule(&self, "com.xilinx:sigmoid"),      // model 18
        create_siso_rule(&self, "com.xilinx:hard_sigmoid"), // model 32
        create_siso_rule(&self, "com.xilinx:pad",
                         {change_paddings}),                // model 11
        create_siso_rule(&self, "com.xilinx:strided_slice",
                         {change_begin_end_step}),          // model hp_FDM1
        create_siso_rule(&self, "Floor"),                   //
        create_siso_rule(&self, "com.xilinx:tanh"),         // Iris，issue#1003
        create_siso_rule(&self, "com.xilinx:sqrt"),         // Win24 C4
        create_siso_rule(&self,
                         "com.xilinx:abs"),   // p1 u8s8 : edgenext_small_rw
        create_siso_rule(&self,
                         "com.xilinx:clamp"), // win24 K2
        // model 10&14 keep_dims = 1
        // model efficientnet-b4 keep_dims = 0
        create_siso_rule(&self, "com.xilinx:l2_normalize",
                         {change_vector_axis}),                 // vaip #1376
        create_siso_rule(&self, "com.xilinx:neg"),              // jira 4851
        create_shape_reduce_siso_rule(&self, "reduction_mean"), //
        create_shape_reduce_siso_rule(&self, "reduction_max"),  // model 10&14
        create_shape_reduce_siso_rule(&self, "reduction_min"),  //
        create_shape_reduce_siso_rule(&self,
                                      "reduction_sum"), // model resnest14d
        create_shape_squeeze_siso_rule(&self),          // model jx_nest_base
        create_transpose_transpose_rule(&self),         //
        create_transpose_const_rule(&self),             //
        create_const_dq_transpose_rule(&self),          //
        create_broadcast_op_rule(&self, "add"),         //
        create_broadcast_op_rule(&self, "sub"),         //
        create_broadcast_op_rule(&self, "mul"),         //
        create_broadcast_op_rule(&self, "pow"),         //
        create_expand_op_rule(&self),                   // model mxgan
        create_tile_op_rule(&self),                     // model G3/GT
        create_broadcast_op_const_rule(&self, "prelu"), // issue #1246
        create_broadcast_op_const_rule(&self, "add"),   //
        create_broadcast_op_const_rule(&self, "sub"),   //
        create_broadcast_op_const_rule(&self, "div"),   //
        create_broadcast_op_const_rule(&self, "max"),   //
        create_broadcast_op_const_rule(&self, "min"),   //
        create_broadcast_op_const_rule(&self, "mul"),   // model efficientnet-b4
        create_broadcast_op_const_rule(&self, "pow"),
        create_broadcast_op_transpose_immune_rule(&self, "pow"),
        create_broadcast_op_transpose_immune_rule(&self, "add"), // no test
        create_broadcast_op_transpose_immune_rule(&self, "sub"), // no PSO0
        create_broadcast_op_transpose_immune_rule(&self, "div"), // no PSO0
        // case now
        create_broadcast_op_transpose_immune_rule(
            &self, "mul"),            // model efficientnet-b4
        create_concat_rule(&self),    //
        create_batchnorm_rule(&self), // model 5001
                                      // model 22 : reshape(transpose(*))
        create_reshape_rule(&self),   // model # RetinaNet
        create_siso_rule(&self, "com.xilinx:identity"),
    };
    auto chain =
        BaseRule::create_rule_chain(std::vector<std::unique_ptr<BaseRule>>{
            std::make_move_iterator(std::begin(rules)),
            std::make_move_iterator(std::end(rules))});
    chain->apply(&graph);
  } // namespace

public:
};
} // namespace

DEFINE_VAIP_PASS(FuseTranspose, vaip_pass_fuse_transpose)
