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

#include "vaip/vaip.hpp"
#include "vitis/ai/env_config.hpp"
#include <glog/logging.h>
DEF_ENV_PARAM(DEBUG_MERGE_CONSECUTIVE_FIX, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_MERGE_CONSECUTIVE_FIX) >= n)
namespace {
using namespace vaip_core;
struct MergeConsecutiveFix {
  std::unique_ptr<Rule> create_rule() {
    auto builder = PatternBuilder();
    std::shared_ptr<Pattern> pat_fix =
        builder.node2("com.xilinx:fix", {builder.wildcard()});
    std::shared_ptr<Pattern> pat_fix_fix =
        builder.node2("com.xilinx:fix", {pat_fix});
    return Rule::create_rule(
        pat_fix_fix, [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
          auto ni_fix = binder[pat_fix->get_id()];
          auto ni_fix_fix = binder[pat_fix_fix->get_id()];
          NodeBuilder(*graph, self_)
              .clone_node(*ni_fix.node)
              .set_anchor_point1(*ni_fix_fix.node)
              .build();
          PASS_LOG(self_, 100) << "hello log";
          return true;
        });
  }
  MergeConsecutiveFix(IPass& self) : self_{self} {}
  void process(IPass& self, Graph& graph) { create_rule()->apply(&graph); }
  IPass& self_;
};
} // namespace
DEFINE_VAIP_PASS(MergeConsecutiveFix, vaip_pass_merge_consecutive_fix)
