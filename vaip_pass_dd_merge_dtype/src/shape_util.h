/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */
#pragma once
#include "vaip/vaip.hpp"
#include "vitis/ai/env_config.hpp"

using namespace vaip_core;

namespace vaip::shape_util {

static std::vector<int64_t> same_shape(const std::vector<int64_t>& shape) {
  return std::vector<int64_t>(shape.begin(), shape.end());
}

static std::vector<int64_t> NHC_to_NHC(const std::vector<int64_t>& shape) {
  CHECK(shape.size() == 3) << "Not NHC Shape";
  auto N = shape[0];
  auto H = shape[1];
  auto C = shape[2];
  return std::vector<int64_t>({N, H, C});
}
[[maybe_unused]] static std::vector<int64_t>
NCHW_to_NHWC(const std::vector<int64_t>& shape) {
  CHECK(shape.size() == 4) << "Not NCHW Shape";

  auto N = shape[0];
  auto C = shape[1];
  auto H = shape[2];
  auto W = shape[3];
  return std::vector<int64_t>({N, H, W, C});
}

static std::string shape_as_dd_string(const std::vector<int64_t>& shape) {
  std::stringstream ss;
  for (const auto& item : shape) {
    ss << item << " ";
  }
  return ss.str();
}
// TODO Super Hacky
static std::vector<int64_t> NCHW_to_NHWC_auto(std::vector<int64_t> shape) {
  if (shape.size() == 4) {
    if (shape[1] == shape[2]) { // H and W are same
      return {shape[0], shape[1], shape[2], shape[3]};
    } else if (shape[2] == shape[3]) {
      return {shape[0], shape[2], shape[3], shape[1]};
    }
  }
  return shape;
}

} // namespace vaip::shape_util