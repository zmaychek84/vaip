/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */
#include "./get_merged_attributes.hpp"
#include "vaip/pattern_zoo.hpp"

namespace {
using namespace vaip_core;

struct MergeQConcateOPs {
  MergeQConcateOPs(IPass& self) : self_{self} {}
  std::unique_ptr<Rule> create_rule(IPass* self) {

    auto com_microsoft_QuantizeLinear_29 =
        vaip::pattern_zoo::get_pattern("m_qconcateops_1_2");
    CHECK(com_microsoft_QuantizeLinear_29 != nullptr)
        << "Pattern returned is null";

    return Rule::create_rule(
        com_microsoft_QuantizeLinear_29,
        [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
          auto ni_input = binder["input_0"];
          auto ni_output = binder["com_microsoft_QuantizeLinear_29"];

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

          get_merged_attributes(new_node, ni_input, ni_output, graph, &binder,
                                model_variant);
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

DEFINE_VAIP_PASS(MergeQConcateOPs, vaip_pass_dd_merge_qconcateops_v_1_2)
