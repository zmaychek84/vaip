/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

static void copy_fix_info_from_input(IPass& pass, const Node& node,
                                     size_t arg_index) {
  auto inputs = node_get_inputs(node);
  CHECK_LT(arg_index, inputs.size()) << node_as_string(node);
  auto& input_node_arg = *inputs[arg_index].node_arg;
  auto input_name = node_arg_get_name(input_node_arg);
  if (pass.has_fix_info(input_name.c_str())) {
    auto node_arg_name = node_get_output_name(node);
    pass.set_fix_info(node_arg_name.c_str(),
                      pass.get_fix_info(input_name.c_str()));
  }
}
template <typename T> static auto Reshape_tmpl() {
  return [](IPass& self, const Node& node, GTensorView<T> output,
            GTensorView<T> input) -> bool {
    copy_fix_info_from_input(self, node, 0u);
    std::copy(input.data.begin(), input.data.end(), output.data.begin());
    return true;
  };
}

template <typename... T> static std::unique_ptr<BaseRule> Reshape(IPass& pass) {
  return std::make_unique<ConstantFoldRule>(pass, "Reshape",
                                            Reshape_tmpl<T>()...);
}
