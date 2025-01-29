/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */
#include <glog/logging.h>

#include "vaip/vaip.hpp"
namespace {
using namespace vaip_core;
/*
  test case model 112

  merge mul pass
  From : mul(mul(input,const_a),const_b)
  To  : mul(input,const_a*const_b)
*/
struct MergeMul {
  MergeMul(IPass& self) : self_{self} {}
  std::unique_ptr<Rule> create_rule() {
    auto builder = PatternBuilder();
    std::shared_ptr<Pattern> input_ = builder.wildcard();
    std::shared_ptr<Pattern> const_a_ = builder.xir_const_op();
    std::shared_ptr<Pattern> mul_a_ =
        builder.node2("com.xilinx:mul", {input_, const_a_});
    std::shared_ptr<Pattern> const_b_ = builder.xir_const_op();
    std::shared_ptr<Pattern> mul_b_ =
        builder.node2("com.xilinx:mul", {mul_a_, const_b_});

    return Rule::create_rule(
        mul_b_, [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
          auto mul_a = binder[mul_a_->get_id()];
          auto& arg_name = node_arg_get_name(*mul_a.node_arg);
          auto consumers = graph_get_consumer_nodes(*graph, arg_name);
          if (consumers.size() > 1) {
            return false;
          }
          auto const_a = binder[const_a_->get_id()];
          auto const_b = binder[const_b_->get_id()];
          auto const_a_shape = node_arg_get_shape_i64(*const_a.node_arg);
          auto const_b_shape = node_arg_get_shape_i64(*const_b.node_arg);
          if (const_a_shape == nullptr || const_b_shape == nullptr) {
            return false;
          }
          if (*const_a_shape != *const_b_shape) {
            return false;
          }

          auto& new_const_b =
              NodeBuilder(*graph, self_)
                  .set_op_type("const")
                  .clone_data_type(*const_b.node)
                  .clone_shape(*const_b.node)
                  .set_anchor_point2(*const_b.node_arg, {"const"})
                  .build();
          auto const_a_const_data = self_.get_const_data<float>(*const_a.node);
          auto const_a_const_data_vec = std::vector<float>(
              const_a_const_data.begin(), const_a_const_data.end());
          auto const_b_const_data = self_.get_const_data<float>(*const_b.node);
          auto const_b_const_data_vec = std::vector<float>(
              const_b_const_data.begin(), const_b_const_data.end());
          auto data = std::vector<float>(const_a_const_data_vec.size());
          for (size_t i = 0; i < const_a_const_data_vec.size(); ++i) {
            data[i] = const_a_const_data_vec[i] * const_b_const_data_vec[i];
          }
          gsl::span<char> span_data =
              gsl::span<char>((char*)data.data(), data.size() * sizeof(float));
          self_.create_const(new_const_b, span_data);
          auto input = binder[input_->get_id()];
          auto mul_b = binder[mul_b_->get_id()];

          NodeBuilder(*graph, self_)
              .clone_op_type(*mul_b.node)
              .set_input_nodes({input.node, &new_const_b})
              .clone_attrs(*mul_b.node)
              .set_anchor_point1(*mul_b.node)
              .build();
          return true;
        });
  }
  void process(IPass& self, Graph& graph) { create_rule()->apply(&graph); }

  IPass& self_;
};
} // namespace

DEFINE_VAIP_PASS(MergeMul, vaip_pass_merge_mul)
