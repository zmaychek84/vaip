/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

#include "vaip/vaip.hpp"
#include <glog/logging.h>
#include <vitis/ai/dim_calc.hpp>
static std::vector<int32_t>
get_dst_shape(const std::vector<int32_t>& src_shape,
              const gsl::span<const int32_t>& order) {
  auto dim_size = src_shape.size();
  auto dst_shape = std::vector<int32_t>(); // src OIHW dst OHWI
  dst_shape.reserve(dim_size);
  for (auto idx : order) {
    dst_shape.push_back(src_shape[idx]);
  }
  return dst_shape;
}
static int get_element_num(const std::vector<int32_t> shape) {
  auto ret = 1;
  for (auto dim : shape) {
    ret *= dim;
  }
  return ret;
}
static std::vector<int32_t>
trans_shape_i64_to_i32(const gsl::span<const int64_t>& shape) {
  auto ret = std::vector<int32_t>(shape.size());
  for (auto i = 0u; i < shape.size(); ++i) {
    ret[i] = (int32_t)shape[i];
  }
  return ret;
}

static std::vector<int32_t> trans_idx(const std::vector<int32_t>& input_idx,
                                      const std::vector<int32_t>& order) {
  auto sz = input_idx.size();
  auto output_idx = std::vector<int32_t>(sz, 0);
  for (auto i = 0u; i < sz; ++i) {
    output_idx[i] = input_idx[order[i]];
  }
  return output_idx;
}

template <typename T>
void transpose_data(const gsl::span<const int64_t>& shape,
                    const gsl::span<const int64_t>& perm,
                    const gsl::span<const T>& data, bool flip_hw, T* ret) {
  auto src_shape = trans_shape_i64_to_i32(shape);
  auto order = trans_shape_i64_to_i32(perm);
  CHECK_EQ(src_shape.size(), order.size());
  auto dst_shape = get_dst_shape(src_shape, order);

  auto size = get_element_num(src_shape);
  CHECK_EQ(data.size(), size);
  auto src_dim_calc = std::make_unique<vitis::ai::DimCalc>(src_shape);
  auto dst_dim_calc = std::make_unique<vitis::ai::DimCalc>(dst_shape);
  if (flip_hw) {
    for (auto i = 0u; i < data.size(); ++i) {
      auto src_idx = src_dim_calc->index(i);
      src_idx[2] = src_shape[2] - src_idx[2] - 1;
      src_idx[3] = src_shape[3] - src_idx[3] - 1;
      auto dst_idx = trans_idx(src_idx, order);
      auto dst_offset = dst_dim_calc->offset(dst_idx);
      ret[dst_offset] = data[i];
    }
  } else {
    for (auto i = 0u; i < data.size(); ++i) {
      auto src_idx = src_dim_calc->index(i);
      auto dst_idx = trans_idx(src_idx, order);
      auto dst_offset = dst_dim_calc->offset(dst_idx);
      ret[dst_offset] = data[i];
    }
  }
  return;
}
template void transpose_data<char>(const gsl::span<const int64_t>&,
                                   const gsl::span<const int64_t>&,
                                   const gsl::span<const char>&, bool, char*);
template void transpose_data<float>(const gsl::span<const int64_t>&,
                                    const gsl::span<const int64_t>&,
                                    const gsl::span<const float>&, bool,
                                    float*);
template void transpose_data<uint16_t>(const gsl::span<const int64_t>&,
                                       const gsl::span<const int64_t>&,
                                       const gsl::span<const uint16_t>&, bool,
                                       uint16_t*);
template void transpose_data<int16_t>(const gsl::span<const int64_t>&,
                                      const gsl::span<const int64_t>&,
                                      const gsl::span<const int16_t>&, bool,
                                      int16_t*);
