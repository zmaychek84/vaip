/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

#pragma once
#include <vaip/vaip.hpp>

namespace qconv2matmul {

// util function for skipping 3 parents of in_node
static std::pair<const NodeArg*, std::vector<std::string>>
find_new_input(const Node* in_node) {
  auto current_node = in_node;
  int node_cnt = 3;
  std::vector<std::string> n_names;
  const NodeArg* new_input_arg = nullptr;
  while (current_node && node_cnt > 0) {
    auto node_inputs = node_get_inputs(*current_node);
    if (node_inputs.size() > 0) {
      current_node = node_inputs[0].node;
      n_names.push_back(node_arg_get_name(*node_inputs[0].node_arg));
      if (node_cnt == 1) {
        new_input_arg = node_inputs[0].node_arg;
      }
    } else {
      break;
    }
    node_cnt--;
  }
  return {new_input_arg, n_names};
}

// direct python translation, may need to change later
static std::pair<std::vector<int64_t>, std::vector<int64_t>>
get_NCHW_NHWC(const std::vector<int64_t>& shapes) {
  if (shapes.size() == 4) {
    if (shapes[1] == shapes[2]) {
      return {{shapes[0], shapes[3], shapes[1], shapes[2]}, shapes};
    } else if (shapes[2] == shapes[3]) {
      return {shapes, {shapes[0], shapes[2], shapes[3], shapes[1]}};
    }
  }
  return {shapes, shapes};
}

} // namespace qconv2matmul
