/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

template <typename T> auto Gather_tmpl() {
  return [](IPass& self, const Node& node, GTensorView<T> output,
            GTensorView<T> input, GTensorView<int64_t> indices) -> bool {
    auto args = node_get_input_node_args(node);
    auto data_shape = input.shape;
    auto indices_shape = indices.shape;
    auto output_shape = output.shape;
    auto r = (int64_t)data_shape.size();
    auto q = (int64_t)indices_shape.size();
    CHECK_GE(r, 1);
    auto axis = node_get_attr_int_with_default(node, "axis", 0);
    if (axis < 0) {
      axis = r + axis;
    }
    auto output_dim_calc = std::make_unique<vitis::ai::DimCalc>(
        trans_shape_i64_to_i32(data_shape));
    auto data_dim_calc = std::make_unique<vitis::ai::DimCalc>(
        output_shape.empty() ? std::vector<int32_t>{1}
                             : trans_shape_i64_to_i32(output_shape));
    auto indices_dim_calc = std::make_unique<vitis::ai::DimCalc>(
        indices_shape.empty() ? std::vector<int32_t>{1}
                              : trans_shape_i64_to_i32(indices_shape));
    for (auto offset = 0u; offset < output.data.size(); ++offset) {
      auto output_idx = output_dim_calc->index(offset);
      auto data_idx = decltype(output_idx)(data_shape.size());
      auto indices_idx = decltype(output_idx)((size_t)q);
      for (int64_t i = 0; i < (int64_t)output_idx.size(); ++i) {
        if (i < axis) {
          data_idx[i] = output_idx[i];
        } else if (i == axis && i < axis + q) {
          indices_idx[i - axis] = output_idx[i];
        } else {
          if (i == axis + q) {
            auto is_scalar = indices_idx.empty();
            size_t src_offset = 0u;
            if (is_scalar) {
              src_offset = 0u;
              CHECK_EQ(output_idx.size(), 1U);
            } else {
              src_offset = indices_dim_calc->offset(indices_idx);
            }
            data_idx[axis] = (int32_t)indices.data[src_offset];
          } else {
            data_idx[i - q + 1] = output_idx[i];
          }
        }
      }
      output.data[offset] = input.data[data_dim_calc->offset(data_idx)];
    }
    return true;
  };
}

template <typename... T> static std::unique_ptr<BaseRule> Gather(IPass& pass) {
  return std::make_unique<ConstantFoldRule>(pass, "Gather",
                                            Gather_tmpl<T>()...);
}
