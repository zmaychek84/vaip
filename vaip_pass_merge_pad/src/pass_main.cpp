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

#include "vaip/vaip.hpp"
#include "vitis/ai/env_config.hpp"

DEF_ENV_PARAM(DEBUG_MERGE_PAD, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_MERGE_PAD) >= n)
/**
 * test case : model 43 51
 * MergePad pattern pass
 * X : wildcard()
 * From : avgpool2d(pad(input))
 * To  : avgpool2d(input)
 *
 *  where avgpool2d.pads is changed.
 */
namespace {
using namespace vaip_core;
struct MergePad {
  MergePad(IPass& self) {}
  static std::unique_ptr<Rule> create_rule(IPass* self) {
    auto builder = PatternBuilder();
    std::shared_ptr<Pattern> pat_input = builder.wildcard();
    std::shared_ptr<Pattern> pat_pad =
        builder.node2("com.xilinx:pad", {pat_input});
    std::shared_ptr<Pattern> pat_avgpool2d =
        builder.node2("com.xilinx:avgpool2d", {pat_pad});
    return Rule::create_rule(
        pat_avgpool2d,
        [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
          auto input = binder[pat_input->get_id()];
          auto pad = binder[pat_pad->get_id()];
          auto avgpool2d = binder[pat_avgpool2d->get_id()];
          auto pdenotation =
              node_arg_get_denotation(node_get_output_node_arg(*pad.node));
          CHECK(pdenotation != nullptr)
              << node_as_string(*pad.node) << " denotation absent";
          auto pad_layout = *pdenotation;
          auto pad_values_v = node_get_attr_ints(*pad.node, "paddings");
          auto num_of_pad_values = pad_values_v.size();
          if (num_of_pad_values != pad_layout.size() * 2) {
            MY_LOG(1) << "padding value number should be 2 times the number of "
                         "dimensions of pad_layout, num_of_pad_values="
                      << num_of_pad_values;
            return false;
          }
          // here pad is com.xilinx:op layout must be NHWC
          CHECK_EQ(pad_layout.size(), 4);
          auto pos_h = 1;
          auto pos_w = 2;
          auto H1 = pad_values_v[2 * pos_h];
          auto H2 = pad_values_v[2 * pos_h + 1];
          auto W1 = pad_values_v[2 * pos_w];
          auto W2 = pad_values_v[2 * pos_w + 1];
          auto sum_pad = 0;
          for (auto i = 0u; i < num_of_pad_values; i++) {
            sum_pad = sum_pad + (int)pad_values_v[i];
          }
          if ((sum_pad - H1 - H2 - W1 - W2) != 0) {
            MY_LOG(1) << "only support pading along H and W dimentsion.";
            return false;
          }
          // TO-DO: pad op's attr "mode" don't check now, maybe some mode can't
          // merge.
          //
          // now model_zoo.list No.50(vehicle_type)/No.51(vehicle_make)/
          // No.52(vehicle_color) model need this MergePad pass
          auto xir_pad = node_get_attr_ints(*avgpool2d.node, "pad");
          CHECK_EQ(xir_pad.size(), 4u)
              << node_as_string(*avgpool2d.node)
              << " pad=" << container_as_string(xir_pad);
          MY_LOG(1) << "merge pad " << node_as_string(*pad.node) << " into "
                    << node_as_string(*avgpool2d.node);
          auto new_xir_pad =
              std::vector<int64_t>(xir_pad.begin(), xir_pad.end());
          new_xir_pad[0] = new_xir_pad[0] + W1; // left
          new_xir_pad[1] = new_xir_pad[1] + W2; // right
          new_xir_pad[2] = new_xir_pad[2] + H1; // top
          new_xir_pad[3] = new_xir_pad[3] + H2; // bottom
          NodeBuilder(*graph, *self)
              .clone_node(*avgpool2d.node)
              .set_input_nodes({input.node})
              .add("pad", new_xir_pad)
              .set_anchor_point1(*avgpool2d.node)
              .build();
          return true;
        });
  }
  void process(IPass& self, Graph& graph) { create_rule(&self)->apply(&graph); }

public:
};
} // namespace

DEFINE_VAIP_PASS(MergePad, vaip_pass_merge_pad)
