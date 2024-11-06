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
namespace {
using namespace vaip_core;
struct MergeFix {
  MergeFix(IPass& self) : self_{self} {}
  static std::unique_ptr<Rule> create_rule(IPass* self) {
    auto builder = PatternBuilder();
    std::shared_ptr<Pattern> pat_float2fix =
        builder.node2("com.xilinx:float2fix", {builder.wildcard()});
    std::shared_ptr<Pattern> pat_fix2float =
        builder.node2("com.xilinx:fix2float", {pat_float2fix});
    return Rule::create_rule(
        pat_fix2float,
        [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
          auto ni_float2fix = binder[pat_float2fix->get_id()];
          auto ni_fix2float = binder[pat_fix2float->get_id()];
          NodeBuilder(*graph, *self)
              .clone_inputs(*ni_float2fix.node)
              .set_op_type("fix")
              .clone_attrs(*ni_float2fix.node)
              .set_anchor_point1(*ni_fix2float.node)
              .build();
          return true;
        });
  }
  void process(IPass& self, Graph& graph) { create_rule(&self)->apply(&graph); }

public:
  IPass& self_;
};
} // namespace

DEFINE_VAIP_PASS(MergeFix, vaip_pass_merge_fix)
