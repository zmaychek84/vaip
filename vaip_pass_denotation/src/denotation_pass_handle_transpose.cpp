/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

#include <glog/logging.h>

#include <algorithm>
#include <unordered_map>

#include "./denotation_pass2.hpp"
#include "vaip/vaip.hpp"
#include "vitis/ai/env_config.hpp"
DEF_ENV_PARAM(DEBUG_DENOTATION_PASS, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_DENOTATION_PASS) >= n)
namespace vaip_pass_denotation {
void NodeActionState::handle_transpose() {
  CHECK_GE(input_layouts_.size(), 1u);
  CHECK_EQ(output_layouts_.size(), 1u);
  auto input_layout = input_layouts_[0].get();
  auto output_layout = output_layouts_[0].get();
  if (nullptr == input_layout || nullptr == output_layout) {
    return;
  }
  auto perm = node_get_attr_ints(*node_, "perm");
  CHECK_EQ(perm.size(), input_layout->size())
      << "rank of must be same." << node_as_string(*node_);
  CHECK_EQ(perm.size(), output_layout->size())
      << "rank of must be same." << node_as_string(*node_);
  for (auto i = 0u; i < perm.size(); ++i) {
    negotiate_layout(std::vector<denotation_view_t>{
        {(*input_layout)[(size_t)perm[i]]}, {(*output_layout)[i]}});
  }
}

} // namespace vaip_pass_denotation
