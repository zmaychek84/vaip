/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */
#include "vaip/vaip.hpp"
#include <cmath>
#include <glog/logging.h>
namespace {
using namespace vaip_core;
/*
  test case model nightly_84_opset17_u8s8_p1/res2net50_26w_4s

  convert split to xir op pass
  From : Split(input,[split])
  To  : strided_slice(input) + strided_slice(input) + ...
*/
struct ConvertSplitToXirOp {
  ConvertSplitToXirOp(IPass& self) : self_{self} {}
  std::unique_ptr<Rule> create_rule() {
    auto builder = PatternBuilder();
    std::shared_ptr<Pattern> pat_input = builder.wildcard();
    std::shared_ptr<Pattern> pat_s = builder.xir_const_op();
    std::shared_ptr<Pattern> pat_split =
        builder.node3("Split", {pat_input, pat_s}, {false, true});

    return Rule::create_rule(
        pat_split, [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
          auto ni_input = binder[pat_input->get_id()];
          auto ni_s = binder[pat_s->get_id()];
          auto ni_split = binder[pat_split->get_id()];

          // check_supported_op
          auto supported_ops = std::vector<std::string>();
          for (auto& pass_conf :
               self_.get_context()->get_config_proto().passes()) {
            if (pass_conf.name() == "fuse_DPU") {
              for (auto op : pass_conf.pass_dpu_param().supported_op()) {
                supported_ops.push_back(op);
              }
            }
          }
          if (!supported_ops.empty()) {
            auto it =
                std::find(supported_ops.begin(), supported_ops.end(), "Split");
            if (it == supported_ops.end()) {
              return false;
            }
          }

          // check_unknown_shape and check_dynamic_shape
          auto inputs = node_get_inputs(*ni_split.node);
          bool is_unknown_shape = false;
          bool is_dynamic_shape = false;
          for (auto ni : inputs) {
            if (!node_arg_exists(*ni.node_arg)) {
              continue;
            }
            is_unknown_shape =
                is_unknown_shape || node_arg_is_unknown_shape(*ni.node_arg);
            is_dynamic_shape =
                is_dynamic_shape || node_arg_is_dynamic_shape(*ni.node_arg);
          }
          auto outputs = node_get_output_node_args(*ni_split.node);
          for (auto output : outputs) {
            is_unknown_shape =
                is_unknown_shape || node_arg_is_unknown_shape(*output);
            is_dynamic_shape =
                is_dynamic_shape || node_arg_is_dynamic_shape(*output);
          }
          if (is_unknown_shape) {
            LOG(WARNING) << "cancel xir conversion, unknown shape found: "
                         << node_as_string(*ni_split.node);
            return false;
          }
          if (is_dynamic_shape) {
            LOG(WARNING) << "cancel xir conversion, dynamic shape found: "
                         << node_as_string(*ni_split.node);
            return false;
          }

          // action
          auto pshape = node_arg_get_shape_i64(*ni_input.node_arg);
          CHECK(pshape != nullptr)
              << node_arg_as_string(*ni_input.node_arg) << " shape absent";
          auto input_shape = *pshape;
          auto rank = (int64_t)input_shape.size();
          auto axis_i =
              node_get_attr_int_with_default(*ni_split.node, "axis", 0);
          // axis_i maybe negative
          axis_i = rank + axis_i;
          auto axis = axis_i % input_shape.size();
          auto num_outputs = (int64_t)outputs.size();

          if (node_has_attr(*ni_split.node, "num_outputs")) {
            num_outputs = node_get_attr_int(*ni_split.node, "num_outputs");
          }
          auto split_size =
              static_cast<int64_t>(std::ceil(input_shape[axis] / num_outputs));
          std::vector<int64_t> split;
          if ((ni_s.node_arg != nullptr)) {
            split = self_.const_data_into<int64_t>(*ni_s.node_arg);
          } else {
            split = std::vector<int64_t>(num_outputs, split_size);
          }

          auto begins = std::vector<int64_t>(num_outputs, 0);
          int64_t sum = 0;
          for (auto i = 1u; i < num_outputs; ++i) {
            sum += split[i - 1];
            begins[i] = sum;
          }
          auto begin = std::vector<int64_t>(input_shape.size(), 0);
          auto end = std::vector<int64_t>(input_shape.size(),
                                          std::numeric_limits<int32_t>::max());
          auto strides = std::vector<int64_t>(input_shape.size(), 1);
          // output_node_arg[0] need remove at last.
          for (int64_t i = num_outputs - 1; i >= 0; --i) {
            begin[axis] = begins[i];
            end[axis] = std::min(begins[i] + split[i], input_shape[axis]);
            NodeBuilder(*graph, self_)
                .set_input_node_args({ni_input.node_arg})
                .set_op_type("strided_slice")
                .add("begin", begin)
                .add("end", end)
                .add("strides", strides)
                .set_anchor_point1(*outputs[i])
                .build();
          }
          return true;
        });
  }
  void process(IPass& self, Graph& graph) { create_rule()->apply(&graph); }

  IPass& self_;
};
} // namespace

DEFINE_VAIP_PASS(ConvertSplitToXirOp, vaip_pass_convert_split_to_xir_op)
