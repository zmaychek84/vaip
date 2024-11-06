/*
 *     The Xilinx Vitis AI Vaip in this distribution are provided under the
 * following free and permissive binary-only license, but are not provided in
 * source code form.  While the following free and permissive license is similar
 * to the BSD open source license, it is NOT the BSD open source license nor
 * other OSI-approved open source license.
 *
 *      Copyright (C) 2022 Xilinx, Inc. All rights reserved.
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

#include <glog/logging.h>

#include <algorithm>
#include <unordered_map>

#include "./denotation_pass2.hpp"
#include "vaip/vaip.hpp"
#include "vitis/ai/env_config.hpp"
DEF_ENV_PARAM(DEBUG_DENOTATION_PASS, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_DENOTATION_PASS) >= n)
namespace vaip_pass_denotation {
void NodeActionState::handle_gemm(const Node& node) {
  MY_LOG(1) << "GEMM layout " << node_as_string(node);
  auto trans_a = node_get_attr_int_with_default(node, "transA", 0);
  auto trans_b = node_get_attr_int_with_default(node, "transB", 0);
  CHECK_GE(this->input_layouts_.size(), 2u);
  auto pl1 = this->input_layouts_[0].get();
  auto pl2 = this->input_layouts_[1].get();
  if (nullptr == pl1 || nullptr == pl2) {
    return;
  }
  CHECK_EQ(pl1->size(), 2u);
  CHECK_EQ(pl2->size(), 2u);
  auto& l1 = *pl1;
  auto& l2 = *pl2;
  using denotation_view_t = NodeActionState::denotation_view_t;
  auto M = std::vector<denotation_view_t>{};
  auto N = std::vector<denotation_view_t>{};
  auto K = std::vector<denotation_view_t>{};
  if (trans_a == 0) {
    M.emplace_back(denotation_view_t{l1[0]});
    K.emplace_back(denotation_view_t{l1[1]});
  } else {
    K.emplace_back(denotation_view_t{l1[0]});
    M.emplace_back(denotation_view_t{l1[1]});
  }
  if (trans_b == 0) {
    K.emplace_back(denotation_view_t{l2[0]});
    N.emplace_back(denotation_view_t{l2[1]});
  } else {
    N.emplace_back(denotation_view_t{l2[0]});
    K.emplace_back(denotation_view_t{l2[1]});
  }
  auto pout = this->output_layouts_[0].get();
  auto& out = *pout;
  M.emplace_back(denotation_view_t{out[0]});
  N.emplace_back(denotation_view_t{out[1]});
  if (this->input_layouts_.size() >= 3) {
    auto pl3 = this->input_layouts_[2].get();
    auto& l3 = *pl3;
    auto dim_size = l3.size();
    if (dim_size == 2) {
      M.emplace_back(denotation_view_t{l3[0]});
      N.emplace_back(denotation_view_t{l3[1]});
    } else if (dim_size == 1) {
      N.emplace_back(denotation_view_t{l3[0]});
    } else {
      // ignore scalar.
    }
  }
  LOG(INFO) << "trans_a " << trans_a << " " //
            << "trans_b " << trans_b << " " //
      ;
  this->negotiate_layout(M);
  this->negotiate_layout(N);
  this->negotiate_layout(K);
}
} // namespace vaip_pass_denotation
