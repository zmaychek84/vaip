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
#include "./get_merged_attributes.hpp"
#include "vaip/pattern_zoo.hpp"

namespace {
using namespace vaip_core;

struct MergeQConcateOPs {
  MergeQConcateOPs(IPass& self) : self_{self} {}
  std::unique_ptr<Rule> create_rule(IPass* self) {

    auto com_microsoft_QuantizeLinear_42 =
        vaip::pattern_zoo::get_pattern("m_qconcateops_1_1");
    CHECK(com_microsoft_QuantizeLinear_42 != nullptr)
        << "Pattern returned is null";

    return Rule::create_rule(
        com_microsoft_QuantizeLinear_42,
        [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
          auto ni_input = binder["input_0"];
          auto ni_output = binder["com_microsoft_QuantizeLinear_42"];

          const auto& provider_option =
              (*self).get_config_proto().provider_options();
          std::string model_variant = "";
          if (provider_option.find("model_variant") != provider_option.end()) {
            model_variant = provider_option.find("model_variant")->second;
            LOG(INFO) << "model_variant = " << model_variant;
          }

          auto new_node = NodeBuilder(*graph, self_);
          new_node.set_op_type("QConcateOPs", "com.xilinx")
              .set_anchor_point1(*ni_output.node);

          get_merged_attributes(new_node, ni_input, ni_output, graph, &binder);
          if (model_variant != "Not Defined")
            new_node.add("model_variant", model_variant);

          new_node.build();
          return true;
        });
  }
  void process(IPass& self, Graph& graph) { create_rule(&self)->apply(&graph); }

public:
  IPass& self_;
};
} // namespace

DEFINE_VAIP_PASS(MergeQConcateOPs, vaip_pass_dd_merge_qconcateops_v_1_1)
