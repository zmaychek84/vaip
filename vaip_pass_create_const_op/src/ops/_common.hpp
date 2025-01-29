/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

static std::vector<int32_t>
trans_shape_i64_to_i32(const std::vector<int64_t>& dims) {
  auto ret = std::vector<int32_t>();
  ret.reserve(dims.size());
  for (auto dim : dims) {
    ret.push_back((int32_t)dim);
  }
  return ret;
}
