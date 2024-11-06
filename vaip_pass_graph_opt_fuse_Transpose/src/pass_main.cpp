/*
 *     The Xilinx Vitis AI Vaip in this distribution are provided under the
 * following free and permissive binary-only license, but are not provided in
 * source code form.  While the following free and permissive license is similar
 * to the BSD open source license, it is NOT the BSD open source license nor
 * other OSI-approved open source license.
 *
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

#include <functional>
#include <glog/logging.h>
#include <numeric>

#include "vaip/vaip.hpp"
#include "vitis/ai/env_config.hpp"
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

static const Node& create_Constant_node(Graph& graph, IPass& pass,
                                        const NodeArg& node_arg,
                                        const std::vector<int64_t>& data) {
  std::vector<int64_t> data_shape = {(int64_t)data.size()};
  auto tensor = tensor_proto_new_i64("constant_data", data_shape, data);
  auto& ret = NodeBuilder(graph, pass)
                  .set_input_node_args({})
                  .set_op_type("Constant", "")
                  .set_data_type("int64")
                  .add("value", *tensor)
                  .set_shape(data_shape)
                  .set_anchor_point2(node_arg, {"Constant"})
                  .build();
  return ret;
}

static std::unique_ptr<Rule> create_siso_rule(
    IPass* self, const std::string& op_type,
    const std::vector<
        std::function<bool(NodeBuilder& builder, const Node& transpose_node,
                           const Node& siso_node)>>&& actions = {}) {
  auto builder = PatternBuilder();
  std::shared_ptr<Pattern> pat_x = builder.wildcard();
  std::shared_ptr<Pattern> pat_transpose = builder.node2("Transpose", {pat_x});
  std::shared_ptr<Pattern> pat_a = builder.constant();
  std::shared_ptr<Pattern> pat_b = builder.constant();
  std::shared_ptr<Pattern> pat_siso_op = builder.node3(
      op_type, {pat_transpose, pat_a, pat_b}, {false, true, true});
  return Rule::create_rule(
      pat_siso_op, [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
        auto ni_x = binder[pat_x->get_id()];
        auto ni_transpose = binder[pat_transpose->get_id()];
        auto ni_a = binder[pat_a->get_id()];
        auto ni_b = binder[pat_b->get_id()];
        auto ni_siso_op = binder[pat_siso_op->get_id()];
        /// siso_op(transpose(x)) => transpose(siso_op(x))
        auto siso_builder = NodeBuilder(*graph, *self);

        // test case:Clip op, model efficientnet-b4
        if ((ni_a.node_arg != nullptr) && (ni_b.node_arg != nullptr)) {
          siso_builder.set_input_node_args(
              {ni_x.node_arg, ni_a.node_arg, ni_b.node_arg});
        } else if ((ni_a.node_arg != nullptr)) {
          siso_builder.set_input_node_args({ni_x.node_arg, ni_a.node_arg});
        } else if ((ni_b.node_arg != nullptr)) {
          siso_builder.set_input_node_args({ni_x.node_arg, ni_b.node_arg});
        } else {
          siso_builder.set_input_node_args({ni_x.node_arg});
        }

        siso_builder.clone_op_type(*ni_siso_op.node)
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
        auto order = node_get_attr_ints(*ni_transpose.node, "perm");
        auto siso_shape = undo_transpose_shape(origin_siso_shape, order);
        siso_builder.set_shape(siso_shape);

        auto new_order = undo_order(order);
        AnchorPointTransposeOpAttr attr;
        attr.mutable_order()->Assign(new_order.begin(), new_order.end());

        auto& new_siso_op =
            siso_builder
                .set_anchor_point2(*ni_siso_op.node_arg, {"Transpose", attr})
                .build();

        NodeBuilder(*graph, *self)
            .set_input_nodes({&new_siso_op})
            .set_op_type("Transpose", "")
            .clone_attrs(*ni_transpose.node)
            .set_anchor_point1(*ni_siso_op.node)
            .build();
        return true;
      });
}

static std::unique_ptr<Rule> create_siso_constant_rule(
    IPass* self, const std::string& op_type,
    const std::vector<
        std::function<bool(NodeBuilder& builder, const Node& transpose_node,
                           const Node& siso_node)>>&& actions = {}) {
  auto builder = PatternBuilder();
  std::shared_ptr<Pattern> pat_x = builder.wildcard();
  std::shared_ptr<Pattern> pat_transpose = builder.node2("Transpose", {pat_x});
  std::shared_ptr<Pattern> pat_scale = builder.constant();
  std::shared_ptr<Pattern> pat_zero_point = builder.constant();
  std::shared_ptr<Pattern> pat_siso_op =
      builder.node3(op_type, {pat_transpose, pat_scale, pat_zero_point},
                    {false, false, true});
  return Rule::create_rule(
      pat_siso_op, [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
        auto ni_x = binder[pat_x->get_id()];
        auto ni_transpose = binder[pat_transpose->get_id()];
        auto ni_scale = binder[pat_scale->get_id()];
        auto ni_zero_point = binder[pat_zero_point->get_id()];
        auto ni_siso_op = binder[pat_siso_op->get_id()];
        /// siso_op(transpose(x)) => transpose(siso_op(x))
        auto siso_builder = NodeBuilder(*graph, *self);

        CHECK(ni_scale.node_arg != nullptr);
        if (ni_zero_point.node_arg != nullptr) {
          siso_builder.set_input_node_args(
              {ni_x.node_arg, ni_scale.node_arg, ni_zero_point.node_arg});
        } else {
          siso_builder.set_input_node_args({ni_x.node_arg, ni_scale.node_arg});
        }

        siso_builder.clone_op_type(*ni_siso_op.node)
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
        auto order = node_get_attr_ints(*ni_transpose.node, "perm");
        auto siso_shape = undo_transpose_shape(origin_siso_shape, order);
        siso_builder.set_shape(siso_shape);

        auto new_order = undo_order(order);
        AnchorPointTransposeOpAttr attr;
        attr.mutable_order()->Assign(new_order.begin(), new_order.end());

        auto& new_siso_op =
            siso_builder
                .set_anchor_point2(*ni_siso_op.node_arg, {"Transpose", attr})
                .build();

        NodeBuilder(*graph, *self)
            .set_input_nodes({&new_siso_op})
            .set_op_type("Transpose", "")
            .clone_attrs(*ni_transpose.node)
            .set_anchor_point1(*ni_siso_op.node)
            .build();
        return true;
      });
}

static std::unique_ptr<Rule>
create_broadcast_op_rule(IPass* self, const std::string& op_type) {
  auto builder = PatternBuilder();
  std::shared_ptr<Pattern> pat_x = builder.wildcard();
  std::shared_ptr<Pattern> pat_y = builder.wildcard();
  std::shared_ptr<Pattern> pat_tr_x = builder.node2("Transpose", {pat_x});
  std::shared_ptr<Pattern> pat_tr_y = builder.node2("Transpose", {pat_y});
  std::shared_ptr<Pattern> pat_add =
      builder.node2(op_type, {pat_tr_x, pat_tr_y});
  return Rule::create_rule(
      pat_add, [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
        auto ni_x = binder[pat_x->get_id()];
        auto ni_y = binder[pat_y->get_id()];
        auto ni_tr_x = binder[pat_tr_x->get_id()];
        auto ni_tr_y = binder[pat_tr_y->get_id()];
        auto ni_add = binder[pat_add->get_id()];
        auto order_x = node_get_attr_ints(*ni_tr_x.node, "perm");
        auto order_y = node_get_attr_ints(*ni_tr_y.node, "perm");
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
                      .set_anchor_point2(*ni_add.node_arg, {"Transpose", attr})
                      .build()})
            .set_op_type("Transpose", "")
            .add("perm", order)
            .set_anchor_point1(*ni_add.node)
            .build();
        return true;
      });
}

static std::unique_ptr<Rule>
create_broadcast_op_const_rule(IPass* self, const std::string& op_type) {
  auto builder = PatternBuilder();
  std::shared_ptr<Pattern> pat_x = builder.wildcard();
  std::shared_ptr<Pattern> pat_y = builder.constant();
  std::shared_ptr<Pattern> pat_tr_x = builder.node2("Transpose", {pat_x});
  std::shared_ptr<Pattern> pat_add1 = builder.node2(op_type, {pat_tr_x, pat_y});
  std::shared_ptr<Pattern> pat_add2 = builder.node2(op_type, {pat_y, pat_tr_x});
  std::shared_ptr<Pattern> pat_add = builder.Or({pat_add1, pat_add2});
  return Rule::create_rule(
      pat_add, [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
        auto ni_x = binder[pat_x->get_id()];
        auto ni_y = binder[pat_y->get_id()];
        auto ni_tr_x = binder[pat_tr_x->get_id()];
        auto ni_add = binder[pat_add->get_id()];
        auto order_x = node_get_attr_ints(*ni_tr_x.node, "perm");
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
        auto y_new_shape = undo_transpose_shape(y_expand_shape, order);
        // test case: model#26(SOLO)/model efficientnet-b4
        auto& constant_shape =
            create_Constant_node(*graph, *self, *ni_y.node_arg, y_expand_shape);
        auto& ni_y_reshape =
            NodeBuilder(*graph, *self)
                .set_input_node_args({ni_y.node_arg, &(node_get_output_node_arg(
                                                         constant_shape))})
                .set_op_type("Reshape", "")
                .clone_data_type(*ni_y.node_arg)
                .set_shape(y_expand_shape)
                .set_anchor_point2(*ni_y.node_arg, {"Reshape"})
                .build();
        auto& new_ni_y =
            NodeBuilder(*graph, *self)
                .set_input_node_args(
                    {&(node_get_output_node_arg(ni_y_reshape))})
                .set_op_type("Transpose", "")
                .add("perm", new_order)
                .set_shape(y_new_shape)
                .clone_data_type(*ni_y.node_arg)
                .set_anchor_point2(*ni_y.node_arg, {"Transpose", attr})
                .build();
        NodeBuilder(*graph, *self)
            .set_input_nodes(
                {//
                 &NodeBuilder(*graph, *self)
                      .set_input_node_args(
                          {ni_x.node_arg,
                           &(node_get_output_node_arg(new_ni_y))})
                      .clone_op_type(*ni_add.node)
                      .clone_data_type(*ni_add.node)
                      .set_shape(new_shape)
                      .clone_attrs(*ni_add.node)
                      .set_anchor_point2(*ni_add.node_arg, {"Transpose", attr})
                      .build()})
            .set_op_type("Transpose", "")
            .add("perm", order)
            .set_anchor_point1(*ni_add.node)
            .build();
        return true;
      });
} // namespace

bool is_tranpose_immune(const gsl::span<const int64_t>& shape) {
  auto ret = false;
  if (shape.empty()) {
    ret = true;
  } else if (shape.size() == 1) {
    ret = true;
  } else {
    size_t n_of_non_ones = 0;
    for (auto s : shape) {
      if (s != 1u) {
        n_of_non_ones = n_of_non_ones + 1;
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
  std::shared_ptr<Pattern> pat_tr_x = builder.node2("Transpose", {pat_x});
  std::shared_ptr<Pattern> pat_add1 = builder.node2(op_type, {pat_tr_x, pat_y});
  std::shared_ptr<Pattern> pat_add2 = builder.node2(op_type, {pat_y, pat_tr_x});
  std::shared_ptr<Pattern> pat_add = builder.Or({pat_add1, pat_add2});
  return Rule::create_rule(
      pat_add, [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
        auto ni_x = binder[pat_x->get_id()];
        auto ni_y = binder[pat_y->get_id()];
        auto ni_tr_x = binder[pat_tr_x->get_id()];
        auto ni_add = binder[pat_add->get_id()];
        auto order_x = node_get_attr_ints(*ni_tr_x.node, "perm");
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
        // test case: model#26(SOLO)/model efficientnet-b4
        auto& constant_shape =
            create_Constant_node(*graph, *self, *ni_y.node_arg, y_new_shape);
        auto& new_ni_y = NodeBuilder(*graph, *self)
                             .set_input_node_args(
                                 {ni_y.node_arg,
                                  &(node_get_output_node_arg(constant_shape))})
                             .set_op_type("Reshape", "")
                             .clone_data_type(*ni_y.node_arg)
                             .set_shape(y_new_shape)
                             .set_anchor_point2(*ni_y.node_arg, {"Reshape"})
                             .build();
        NodeBuilder(*graph, *self)
            .set_input_nodes(
                {//
                 &NodeBuilder(*graph, *self)
                      .set_input_node_args(
                          {ni_x.node_arg,
                           &((node_get_output_node_arg)(new_ni_y))})
                      .clone_op_type(*ni_add.node)
                      .clone_data_type(*ni_add.node)
                      .set_shape(new_shape)
                      .clone_attrs(*ni_add.node)
                      .set_anchor_point2(*ni_add.node_arg, {"Transpose", attr})
                      .build()})
            .set_op_type("Transpose", "")
            .add("perm", order)
            .set_anchor_point1(*ni_add.node)
            .build();
        return true;
      });
} // namespace

static std::unique_ptr<Rule> create_Transpose_Transpose_rule(IPass* self) {
  auto builder = PatternBuilder();
  std::shared_ptr<Pattern> pat_t1 =
      builder.node2("Transpose", {builder.wildcard()});
  std::shared_ptr<Pattern> pat_t2 = builder.node2("Transpose", {pat_t1});
  return Rule::create_rule(
      pat_t2, [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
        auto ni_t1 = binder[pat_t1->get_id()];
        auto ni_t2 = binder[pat_t2->get_id()];
        auto order1 = node_get_attr_ints(*ni_t1.node, "perm");
        auto order2 = node_get_attr_ints(*ni_t2.node, "perm");
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
            .add("perm", order3)
            .set_anchor_point1(*ni_t2.node)
            .build();
        return true;
      });
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
        if (!node_is_op(*ni_concat.node, "Concat", "")) {
          return false;
        }
        auto concat_inputs = node_get_inputs(*ni_concat.node);
        auto at_least_one_transpose = 0u;
        auto all_transpose_order_is_same = true;
        auto order = std::vector<int64_t>();
        // concat inputs must be const or transpose, and at least one transpose.
        // model #26 SOLO.
        for (auto concat_input : concat_inputs) {
          auto node_is_transpose =
              concat_input.node &&
              node_is_op(*concat_input.node, "Transpose", "");
          auto node_is_const =
              concat_input.node_arg != nullptr &&
              VAIP_ORT_API(node_arg_is_exists)(*concat_input.node_arg) &&
              VAIP_ORT_API(node_arg_is_constant)(*graph,
                                                 *concat_input.node_arg);
          if (!(node_is_transpose || node_is_const)) {
            return false;
          }
          if (node_is_transpose) {
            at_least_one_transpose++;
            if (order.empty()) {
              auto first_order = node_get_attr_ints(*concat_input.node, "perm");
              order.assign(first_order.begin(), first_order.end());
            } else {
              // all inputs must have same transpose(order:=...);
              all_transpose_order_is_same =
                  all_transpose_order_is_same &&
                  node_get_attr_ints(*concat_input.node, "perm") ==
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
        // NCHW2NHWC {0, 2, 3, 1}
        AnchorPointTransposeOpAttr attr;
        attr.mutable_order()->Assign(new_order.begin(), new_order.end());

        for (auto concat_input : concat_inputs) {
          auto node_is_const =
              concat_input.node_arg != nullptr &&
              VAIP_ORT_API(node_arg_is_exists)(*concat_input.node_arg) &&
              VAIP_ORT_API(node_arg_is_constant)(*graph,
                                                 *concat_input.node_arg);
          if (node_is_const) {
            auto pshape = node_arg_get_shape_i64(*concat_input.node_arg);
            CHECK(pshape != nullptr)
                << node_arg_as_string(*concat_input.node_arg)
                << " shape absent";
            auto new_shape = undo_transpose_shape(*pshape, order);
            auto& new_node = NodeBuilder(*graph, *pass)
                                 .set_input_node_args({concat_input.node_arg})
                                 .set_op_type("Transpose", "")
                                 .add("perm", new_order)
                                 .set_shape(new_shape)
                                 .clone_data_type(*concat_input.node_arg)
                                 .set_anchor_point2(*concat_input.node_arg,
                                                    {"Transpose", attr})
                                 .build();
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

        NodeBuilder(*graph, *pass)
            .set_input_nodes({&NodeBuilder(*graph, *pass)
                                   .set_input_node_args(input_args)
                                   .set_op_type("Concat", "")
                                   .add("axis", new_axis)
                                   .set_shape(new_shape)
                                   .clone_data_type(*ni_concat.node_arg)
                                   .set_anchor_point2(*ni_concat.node_arg,
                                                      {"Transpose", attr})
                                   .build()})
            .set_op_type("Transpose", "")
            .add("perm", order)
            .set_anchor_point1(*ni_concat.node)
            .build();
        return true;
      });
}

// static std::unique_ptr<Rule> create_batchnorm_rule(IPass* self) {
//  auto builder = PatternBuilder();
//  std::shared_ptr<Pattern> pat_input = builder.wildcard();
//  std::shared_ptr<Pattern> pat_transpose =
//      builder.node2("com.xilinx:transpose", {pat_input});
//  std::shared_ptr<Pattern> pat_gamma = builder.xir_const_op();
//  std::shared_ptr<Pattern> pat_beta = builder.xir_const_op();
//  std::shared_ptr<Pattern> pat_moving_mean = builder.xir_const_op();
//  std::shared_ptr<Pattern> pat_moving_var = builder.xir_const_op();
//  std::shared_ptr<Pattern> pat_batchnorm =
//      builder.node2("com.xilinx:batchnorm", {pat_transpose, pat_gamma,
//      pat_beta,
//                                             pat_moving_mean,
//                                             pat_moving_var});
//  return Rule::create_rule(
//      pat_batchnorm, [=](onnxruntime::Graph* graph, binder_t& binder) -> bool
//      {
//        auto ni_input = binder[pat_input->get_id()];
//        auto ni_transpose = binder[pat_transpose->get_id()];
//        auto ni_gamma = binder[pat_gamma->get_id()];
//        auto ni_beta = binder[pat_beta->get_id()];
//        auto ni_moving_mean = binder[pat_moving_mean->get_id()];
//        auto ni_moving_var = binder[pat_moving_var->get_id()];
//        auto ni_batchnorm = binder[pat_batchnorm->get_id()];
//
//        auto origin_batchnorm_shape =
//            node_get_output_shape(*ni_batchnorm.node, 0);
//        auto order =
//            node_get_attr_ints(*ni_transpose.node, "order");
//        if (origin_batchnorm_shape.size() != order.size()) {
//          return false;
//        }
//        auto new_axis =
//            order[node_get_attr_int(*ni_batchnorm.node,
//            "axis")];
//        auto batchnorm_shape =
//            undo_transpose_shape(origin_batchnorm_shape, order);
//
//        CHECK(ni_gamma.node != nullptr);
//        CHECK(ni_beta.node != nullptr);
//        CHECK(ni_moving_mean.node != nullptr);
//        CHECK(ni_moving_var.node != nullptr);
//
//        auto new_order = undo_order(order);
//        AnchorPointTransposeOpAttr attr;
//        attr.mutable_order()->Assign(new_order.begin(), new_order.end());
//        auto& new_batchnorm =
//            NodeBuilder(*graph, *self)
//                .clone_op_type(*ni_batchnorm.node)
//
//                .set_input_node_args({ni_input.node_arg, ni_gamma.node_arg,
//                                      ni_beta.node_arg,
//                                      ni_moving_mean.node_arg,
//                                      ni_moving_var.node_arg})
//                .clone_data_type(*ni_batchnorm.node)
//                .set_shape(batchnorm_shape)
//                .clone_attrs(*ni_batchnorm.node)
//                .add("axis", new_axis)
//                .set_anchor_point2(*ni_batchnorm.node_arg, {"transpose",
//                attr}) .build();
//
//        NodeBuilder(*graph, *self)
//            .set_input_nodes({&new_batchnorm})
//            .set_op_type("transpose")
//            .clone_attrs(*ni_transpose.node)
//            .set_anchor_point1(*ni_batchnorm.node)
//            .build();
//
//        return true;
//      });
//}
//
static std::unique_ptr<Rule> create_shape_reduce_siso_rule(
    IPass* self, const std::string& op_type,
    const std::vector<
        std::function<bool(NodeBuilder& builder, const Node& transpose_node,
                           const Node& siso_node)>>&& actions = {}) {
  auto builder = PatternBuilder();
  std::shared_ptr<Pattern> pat_x = builder.wildcard();
  std::shared_ptr<Pattern> pat_transpose = builder.node2("Transpose", {pat_x});
  std::shared_ptr<Pattern> pat_siso_op =
      builder.node2(op_type, {pat_transpose});
  return Rule::create_rule(
      pat_siso_op, [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
        // auto ni_x = binder[pat_x->get_id()];
        auto ni_transpose = binder[pat_transpose->get_id()];
        auto ni_siso_op = binder[pat_siso_op->get_id()];

        auto origin_siso_shape = node_get_output_shape(*ni_siso_op.node, 0);
        auto order = node_get_attr_ints(*ni_transpose.node, "perm");
        int64_t keep_dims = 0;
        auto origin_axis = node_get_attr_ints(*ni_siso_op.node, "axes");
        auto new_axis =
            layout_transform_vector_axis_alike_attr(origin_axis, order);
        if (node_has_attr(*ni_siso_op.node, "keepdims")) {
          keep_dims = node_get_attr_int(*ni_siso_op.node, "keepdims");
        }

        auto reduce_order = std::vector<int64_t>();
        reduce_order.assign(order.begin(), order.end());
        if (!keep_dims) {
          auto size = order.size() - origin_axis.size();
          auto reduce_order_tmp = std::vector<int64_t>();
          for (auto i = 0u; i < order.size(); ++i) {
            if (std::find(origin_axis.begin(), origin_axis.end(), i) ==
                origin_axis.end()) {
              reduce_order_tmp.push_back(order[i]);
            }
          }
          auto index = std::vector<int64_t>(size);
          std::iota(index.begin(), index.end(), 0);
          std::sort(index.begin(), index.end(),
                    [&reduce_order_tmp](int64_t pos1, int64_t pos2) {
                      return (reduce_order_tmp[pos1] < reduce_order_tmp[pos2]);
                    });
          //          MY_LOG(1) << "reduce_order_tmp: " << reduce_order_tmp
          //                    << " index: " << index;
          reduce_order.resize(size);
          reduce_order.assign(index.begin(), index.end());
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
                .add("axes", new_axis)
                .set_anchor_point2(*ni_siso_op.node_arg, {"Transpose", attr})
                .build();

        NodeBuilder(*graph, *self)
            .set_input_nodes({&new_siso_op})
            .set_op_type("Transpose", "")
            .clone_attrs(*ni_transpose.node)
            .add("perm", reduce_order)
            .set_anchor_point1(*ni_siso_op.node)
            .build();
        return true;
      });
}

// test case: model #43  #15:salsanext_v2  #100:3_se_model1
static std::unique_ptr<Rule> create_Pad_rule(IPass* self) {
  auto builder = PatternBuilder();
  std::shared_ptr<Pattern> pat_x = builder.wildcard();
  std::shared_ptr<Pattern> pat_pads = builder.wildcard();
  std::shared_ptr<Pattern> pat_constant_value = builder.wildcard();
  std::shared_ptr<Pattern> pat_tr_x = builder.node2("Transpose", {pat_x});
  std::shared_ptr<Pattern> pat_pad = builder.node3(
      "Pad", {pat_tr_x, pat_pads, pat_constant_value}, {false, false, true});
  return Rule::create_rule(
      pat_pad, [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
        auto ni_x = binder[pat_x->get_id()];
        auto ni_pads = binder[pat_pads->get_id()];
        auto ni_constant_value = binder[pat_constant_value->get_id()];
        auto ni_tr_x = binder[pat_tr_x->get_id()];
        auto ni_pad = binder[pat_pad->get_id()];
        auto order_x = node_get_attr_ints(*ni_tr_x.node, "perm");
        auto order = std::vector<int64_t>(order_x.begin(), order_x.end());
        auto pshape = node_arg_get_shape_i64(*ni_pad.node_arg);
        CHECK(pshape != nullptr)
            << node_arg_as_string(*ni_pad.node_arg) << " shape absent";
        auto new_shape = undo_transpose_shape(*pshape, order);

        auto new_order = undo_order(order);
        AnchorPointTransposeOpAttr attr;
        attr.mutable_order()->Assign(new_order.begin(), new_order.end());

        auto pads_pshape = node_arg_get_shape_i64(*ni_pads.node_arg);
        CHECK(pads_pshape != nullptr)
            << node_arg_as_string(*ni_pads.node_arg) << " shape absent";
        auto pads_shape = *pads_pshape;
        CHECK(pads_shape.size() == 1) << "pads is 1D tensor.";
        // now only support pads.size() = 2 * pad_input.size(), also means Pad
        // op don't have axes input node_arg, axes input node_arg is optional
        // for Pad op. This maybe support in the future.
        if (pads_shape[0] != 2 * (int64_t)order.size()) {
          return false; // don't support now
        }
        auto new_pads_order = std::vector<int64_t>(pads_shape[0]);
        for (auto i = 0u; i < new_order.size(); i++) {
          new_pads_order[i] = new_order[i];
          new_pads_order[i + new_order.size()] =
              new_order[i] + new_order.size();
        }
        AnchorPointTransposeOpAttr pads_attr;
        pads_attr.mutable_order()->Assign(new_pads_order.begin(),
                                          new_pads_order.end());
        auto& new_pads_indices_node = create_Constant_node(
            *graph, *self, *ni_pads.node_arg, new_pads_order);
        std::vector<int64_t> new_pads_shape = {(int64_t)new_pads_order.size()};
        auto& new_pads_node =
            NodeBuilder(*graph, *self)
                .set_input_node_args(
                    {ni_pads.node_arg,
                     &(node_get_output_node_arg(new_pads_indices_node))})
                .set_op_type("GatherElements", "")
                .set_data_type("int64")
                .set_shape(new_pads_shape)
                .set_anchor_point2(*ni_pads.node_arg, {"Transpose", pads_attr})
                .build();
        std::vector<const NodeArg*> input_args = {
            ni_x.node_arg, &(node_get_output_node_arg(new_pads_node))};
        if (ni_constant_value.node_arg != nullptr) {
          input_args.push_back(ni_constant_value.node_arg);
        }
        NodeBuilder(*graph, *self)
            .set_input_nodes(
                {//
                 &NodeBuilder(*graph, *self)
                      .set_input_node_args(input_args)
                      .clone_op_type(*ni_pad.node)
                      .clone_data_type(*ni_pad.node)
                      .set_shape(new_shape)
                      .clone_attrs(*ni_pad.node)
                      .set_anchor_point2(*ni_pad.node_arg, {"Transpose", attr})
                      .build()})
            .set_op_type("Transpose", "")
            .add("perm", order)
            .set_anchor_point1(*ni_pad.node)
            .build();
        return true;
      });
} // namespace

// test case: model # text_quantized_detection
static std::unique_ptr<Rule> create_Resize_rule(IPass* self) {
  auto builder = PatternBuilder();
  std::shared_ptr<Pattern> pat_x = builder.wildcard();
  std::shared_ptr<Pattern> pat_tr_x = builder.node2("Transpose", {pat_x});
  std::shared_ptr<Pattern> pat_roi = builder.wildcard();
  std::shared_ptr<Pattern> pat_scales = builder.wildcard();
  std::shared_ptr<Pattern> pat_sizes = builder.wildcard();
  std::shared_ptr<Pattern> pat_resize =
      builder.node3("Resize", {pat_tr_x, pat_roi, pat_scales, pat_sizes},
                    {false, false, false, true});
  return Rule::create_rule(
      pat_resize, [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
        auto ni_x = binder[pat_x->get_id()];
        auto ni_tr_x = binder[pat_tr_x->get_id()];
        auto ni_roi = binder[pat_roi->get_id()];
        auto ni_scales = binder[pat_scales->get_id()];
        auto ni_sizes = binder[pat_sizes->get_id()];
        auto ni_resize = binder[pat_resize->get_id()];
        auto order_x = node_get_attr_ints(*ni_tr_x.node, "perm");
        auto order = std::vector<int64_t>(order_x.begin(), order_x.end());
        auto pshape = node_arg_get_shape_i64(*ni_resize.node_arg);
        CHECK(pshape != nullptr)
            << node_arg_as_string(*ni_resize.node_arg) << " shape absent";
        auto new_shape = undo_transpose_shape(*pshape, order);

        auto new_order = undo_order(order);
        AnchorPointTransposeOpAttr attr;
        attr.mutable_order()->Assign(new_order.begin(), new_order.end());
        std::vector<const NodeArg*> input_args = {ni_x.node_arg,
                                                  ni_roi.node_arg};
        // test case: model #1 & #20
        if (node_arg_exists(*ni_scales.node_arg) &&
            !node_arg_is_zero_shape(*ni_scales.node_arg)) {
          auto& new_scales_indices_node = create_Constant_node(
              *graph, *self, *ni_scales.node_arg, new_order);
          std::vector<int64_t> new_scales_shape = {(int64_t)new_order.size()};
          auto& new_scales_node =
              NodeBuilder(*graph, *self)
                  .set_input_node_args(
                      {ni_scales.node_arg,
                       &(node_get_output_node_arg(new_scales_indices_node))})
                  .set_op_type("GatherElements", "")
                  .set_data_type("float32")
                  .set_shape(new_scales_shape)
                  .set_anchor_point2(*ni_scales.node_arg, {"Transpose", attr})
                  .build();
          input_args.push_back(&(node_get_output_node_arg(new_scales_node)));
        } else {
          input_args.push_back(ni_scales.node_arg);
        }

        if (ni_sizes.node_arg != nullptr &&
            node_arg_exists(*ni_sizes.node_arg) &&
            !node_arg_is_zero_shape(*ni_sizes.node_arg)) {
          auto& new_indices_node = create_Constant_node(
              *graph, *self, *ni_sizes.node_arg, new_order);
          std::vector<int64_t> new_sizes_shape = {(int64_t)new_order.size()};
          auto& new_sizes_node =
              NodeBuilder(*graph, *self)
                  .set_input_node_args(
                      {ni_sizes.node_arg,
                       &(node_get_output_node_arg(new_indices_node))})
                  .set_op_type("GatherElements", "")
                  .set_data_type("int64")
                  .set_shape(new_sizes_shape)
                  .set_anchor_point2(*ni_sizes.node_arg, {"Transpose", attr})
                  .build();
          input_args.push_back(&(node_get_output_node_arg(new_sizes_node)));
        }
        // TO-DO: if ni_roi exist, it need deal with like the above ni_scales/
        // ni_sizes, otherwise it's error, now don't deal with it.
        NodeBuilder(*graph, *self)
            .set_input_nodes({//
                              &NodeBuilder(*graph, *self)
                                   .set_input_node_args(input_args)
                                   .clone_op_type(*ni_resize.node)
                                   .clone_data_type(*ni_resize.node)
                                   .set_shape(new_shape)
                                   .clone_attrs(*ni_resize.node)
                                   .set_anchor_point2(*ni_resize.node_arg,
                                                      {"Transpose", attr})
                                   .build()})
            .set_op_type("Transpose", "")
            .add("perm", order)
            .set_anchor_point1(*ni_resize.node)
            .build();
        return true;
      });
} // namespace

// test case: model # RetinaNet
static std::unique_ptr<Rule> create_reshape_rule(IPass* self) {
  auto builder = PatternBuilder();
  std::shared_ptr<Pattern> pat_x = builder.wildcard();
  std::shared_ptr<Pattern> pat_transpose = builder.node2("Transpose", {pat_x});
  std::shared_ptr<Pattern> pat_shape = builder.wildcard();
  std::shared_ptr<Pattern> pat_reshape_op =
      builder.node2("Reshape", {pat_transpose, pat_shape});
  return Rule::create_rule(
      pat_reshape_op, [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
        auto ni_x = binder[pat_x->get_id()];
        auto ni_transpose = binder[pat_transpose->get_id()];
        auto ni_shape = binder[pat_shape->get_id()];
        auto ni_reshape_op = binder[pat_reshape_op->get_id()];

        // if Reshape's input data node_arg or output node_arg is UNKNOWN shape,
        // skip fuse. From Reshape op v14 defination, if Reshape's input shape
        // node_arg is not a ConstantInitializer, it will not shape_infer and
        // return UNKNOWN output node_arg shape.
        // test case: model # RetinaNet
        if (node_arg_is_unknown_shape(*ni_transpose.node_arg) ||
            node_arg_is_unknown_shape(*ni_reshape_op.node_arg)) {
          return false;
        }
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
        auto order = node_get_attr_ints(*ni_transpose.node, "perm");
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
        auto& new_indices_node =
            create_Constant_node(*graph, *self, *ni_shape.node_arg, new_order);
        std::vector<int64_t> new_shape_shape = {(int64_t)new_order.size()};
        auto& new_shape_node =
            NodeBuilder(*graph, *self)
                .set_input_node_args(
                    {ni_shape.node_arg,
                     &(node_get_output_node_arg(new_indices_node))})
                .set_op_type("GatherElements", "")
                .set_data_type("int64")
                .set_shape(new_shape_shape)
                .set_anchor_point2(*ni_shape.node_arg, {"Transpose", attr})
                .build();

        auto reshape_builder = NodeBuilder(*graph, *self);
        auto& new_reshape_op =
            reshape_builder
                .set_input_node_args({ni_x.node_arg, &(node_get_output_node_arg(
                                                         new_shape_node))})
                .clone_op_type(*ni_reshape_op.node)
                .clone_data_type(*ni_reshape_op.node)
                .clone_attrs(*ni_reshape_op.node)
                .set_shape(reshape_shape)
                .set_anchor_point2(*ni_reshape_op.node_arg, {"Transpose", attr})
                .build();

        NodeBuilder(*graph, *self)
            .set_input_nodes({&new_reshape_op})
            .set_op_type("Transpose", "")
            .clone_attrs(*ni_transpose.node)
            .add("perm", increase_order)
            .set_anchor_point1(*ni_reshape_op.node)
            .build();
        return true;
      });
}

struct GraphOptFuseTranspose {
  GraphOptFuseTranspose(IPass& self) {}
  void process(IPass& self, Graph& graph) {
    std::unique_ptr<BaseRule> rules[] = {
        create_siso_rule(&self, "Relu"), //
        create_siso_rule(&self, "Clip"), // model efficientnet-b4
        create_siso_constant_rule(&self, "QuantizeLinear"),   //
        create_siso_constant_rule(&self, "DequantizeLinear"), //
        create_siso_rule(&self, "LeakyRelu"),                 // model 15
        create_siso_rule(&self, "Sigmoid"),                   // model 18
        create_siso_rule(&self, "Floor"), // model text_quantized_detection
        //        create_siso_rule(&self, "com.xilinx:hard_sigmoid"),  // model
        //        32 create_siso_rule(&self, "com.xilinx:pixel_shuffle"), //
        //        model 28
        create_Pad_rule(&self),    // model 43 & 11
        create_Resize_rule(&self), // model text_quantized_detection
        //        // model 10&14 keep_dims = 1
        //        // model efficientnet-b4 keep_dims = 0
        create_shape_reduce_siso_rule(&self, "ReduceMean"), //
        create_shape_reduce_siso_rule(&self, "ReduceMax"),  // model 10&14
        create_shape_reduce_siso_rule(&self, "ReduceMin"),  //
        create_shape_reduce_siso_rule(&self, "ReduceSum"),  // model resnest14d
        //        create_shape_reduce_siso_rule(&self, "squeeze"),        //
        create_Transpose_Transpose_rule(&self),       //
        create_broadcast_op_rule(&self, "Add"),       //
        create_broadcast_op_rule(&self, "Mul"),       //
        create_broadcast_op_const_rule(&self, "Add"), //
        create_broadcast_op_const_rule(&self, "Div"), //
        create_broadcast_op_const_rule(&self, "Max"), //
        create_broadcast_op_const_rule(&self, "Min"), //
        create_broadcast_op_const_rule(&self, "Mul"), // model efficientnet-b4
        create_broadcast_op_transpose_immune_rule(&self, "Add"), // no test case
                                                                 // now
        create_broadcast_op_transpose_immune_rule(
            &self, "Mul"),         // model efficientnet-b4
        create_concat_rule(&self), //
        //        create_batchnorm_rule(&self), // model 5001
        // model 22 : reshape(transpose(*))
        create_reshape_rule(&self), // model # RetinaNet
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

DEFINE_VAIP_PASS(GraphOptFuseTranspose, vaip_pass_graph_opt_fuse_Transpose)
