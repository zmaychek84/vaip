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
// testcase 110
#include "vaip/vaip.hpp"
#include "vitis/ai/env_config.hpp"
#include <glog/logging.h>
DEF_ENV_PARAM(DEBUG_REMOVE_BOTTOM_TRANSPOSE, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_REMOVE_BOTTOM_TRANSPOSE) >= n)

/// remove the bottom tranpose op.  Why? for model #100, it triggers a
/// strange bug, `xcompiler(feat_ipu_compiler_demo)` complains that
/// size is too large, I don't why, to work around this bug, we remove
/// the last op if it is `transpose`
///
namespace {
using namespace vaip_core;
struct RemoveBottomTranspose {
  RemoveBottomTranspose(const IPass& self) : self_{self} {}
  void process(const IPass& self, Graph& graph) {
    auto graph_outputs = graph_get_outputs(graph);
    std::vector<const NodeArg*> new_outputs;
    new_outputs.resize(graph_outputs.size());
    auto index = 0u;
    for (auto output : graph_outputs) {
      auto node =
          VAIP_ORT_API(graph_producer_node)(graph, node_arg_get_name(*output));
      if (node_is_op(*node, "transpose", "xilinx.com")) {
        new_outputs[index] = node_get_input_node_args(*node)[0];
      } else {
        new_outputs[index] = graph_outputs[index];
      }
      index = index + 1;
    }
    VAIP_ORT_API(graph_set_outputs)(graph, new_outputs);
  }
  const IPass& self_;
};
} // namespace
DEFINE_VAIP_PASS(RemoveBottomTranspose, vaip_pass_remove_bottom_transpose)
