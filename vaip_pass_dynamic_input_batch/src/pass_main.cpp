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
DEF_ENV_PARAM(DEBUG_DYNAMIC_INPUT_BATCH, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_DYNAMIC_INPUT_BATCH) >= n)
namespace {
using namespace vaip_core;
struct DynamicInputBatch {
  DynamicInputBatch(IPass& self) {}

  // change the shape of input from -1 to 1  to support dynamic batching
  bool process(IPass& self, Graph& graph) {
    auto changed = false;
    auto graph_inputs = graph_get_inputs(graph);
    for (auto input : graph_inputs) {
      auto input_shape_ptr = node_arg_get_shape_i64(*input);
      CHECK(input_shape_ptr != nullptr)
          << node_arg_as_string(*input) << " shape absent";
      auto input_shape = *input_shape_ptr;
      if (input_shape.size() == 0) {
        // scalar
        continue;
      }
      for (auto i = 1u; i < input_shape.size(); i++) {
        CHECK_NE(input_shape[i], -1)
            << "don't support other shape dimension except batch"
            << "is -1 now";
      }
      if (input_shape[0] == -1) {
        MY_LOG(1) << "do graph_input_modify_batch.";
        input_shape[0] = 1;
        VAIP_ORT_API(node_arg_set_shape_i64)(*input, input_shape);
        changed = true;
        graph_resolve(graph, true);
      }
    }
    return changed;
  }

private:
};
} // namespace
DEFINE_VAIP_PASS(DynamicInputBatch, vaip_pass_dynamic_input_batch)
