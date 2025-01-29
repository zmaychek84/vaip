/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

template <typename T> auto Shape_tmpl() {
  return [](IPass& self, const Node& node, GTensorView<int64_t> output,
            GTensorView<T> input) -> bool {
    auto input_args = node_get_input_node_args(node);
    CHECK_EQ(input_args.size(), 1) << "Shape input_arg size must 1";
    auto& input_arg = *input_args[0];
    if (node_arg_is_dynamic_shape(input_arg)) {
      return false;
    }
    if (node_arg_is_scalar(input_arg)) {
      return false;
    }
    auto pshape = node_arg_get_shape_i64(input_arg);
    CHECK(pshape != nullptr)
        << node_arg_as_string(input_arg) << " shape absent";
    auto input_shape = *pshape;
    auto rank = (int64_t)input_shape.size();
    auto start = node_get_attr_int_with_default(node, "start", 0);
    auto end = node_get_attr_int_with_default(node, "end", rank);
    if (end < 0) {
      end = rank + end;
    }
    for (auto i = start; i < end; ++i) {
      CHECK_GT(input_shape[i], 0);
      output.data[i] = input_shape[i];
    }
    return true;
  };
}

template <typename... T> static std::unique_ptr<BaseRule> Shape(IPass& pass) {
  return std::make_unique<ConstantFoldRule>(pass, "Shape", Shape_tmpl<T>()...);
}
