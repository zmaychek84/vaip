/*
 *     The Xilinx Vitis AI Vaip in this distribution are provided under the
 * following free and permissive binary-only license, but are not provided in
 * source code form.  While the following free and permissive license is similar
 * to the BSD open source license, it is NOT the BSD open source license nor
 * other OSI-approved open source license.
 *
# Copyright (C) 2022 Xilinx, Inc.
 *      Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights
 * reserved.
 *
 *      Redistribution and use in binary form only, without modification, is
 * permitted provided that the following conditions are met:
 *
 *      1. Redistributions must reproduce the above copyright notice, this list
 * of conditions and the following disclaimer in the documentation and/or other
 * materials provided with the distribution.
 *
 *      2. The name of Xilinx, Inc. may not be used to endorse or promote
 * products redistributed with this software without specific prior written
 * permission.
 *
 *      THIS SOFTWARE IS PROVIDED BY XILINX, INC. "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL XILINX, INC. BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 *      PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE
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
