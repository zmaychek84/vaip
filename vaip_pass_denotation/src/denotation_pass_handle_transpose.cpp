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
