/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */
#ifdef __GNUC__
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wpedantic"
#  pragma GCC diagnostic ignored "-Wconversion"
#endif

#include "vaip/dd/coeffs.hpp"
#include "vaip/dd/dd_utils.hpp"

#ifdef __GNUC__
#  pragma GCC diagnostic pop
#endif
#include "../../src/pass_imp.hpp"
#include "vaip/vaip.hpp"
#include "vitis/ai/env_config.hpp"
#include <glog/logging.h>

#include "helper.hpp"
#include "modifiers.hpp"
#include "vaip/pattern_zoo.hpp"

DEF_ENV_PARAM(EXTRACT_SUBGRAPHS, "0")
using namespace vaip_core;

struct Merge_node_builder {
  Merge_node_builder(IPass& self) : self_{self} {}

  std::unique_ptr<Rule> create_rule(IPass* self) {
    auto& pass_proto = self_.get_pass_proto();

    MY_LOG(1) << "Opname" << pass_proto.pass_fusion_param().op_name();
    MY_LOG(1) << "Fuse Pattern"
              << pass_proto.pass_fusion_param().pattern().pattern_name();

    auto in_pattern = vaip::pattern_zoo::get_pattern(
        pass_proto.pass_fusion_param().pattern().pattern_name());
    CHECK(in_pattern != nullptr) << "Pattern returned is null";

    return Rule::create_rule(
        in_pattern, [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
          PassProto pass_proto = self_.get_pass_proto();
          FuseState* fuse_state = new FuseState(
              graph, binder, NodeBuilder(*graph, *self), pass_proto, self);

          std::vector<std::string> task_list = {
              "infer_input_node_args", "infer_input_dtypes",
              "infer_output_dtypes",   "infer_input_shape",
              "infer_output_shape",    "infer_io_qparams",
              "set_initializers",      "set_accessor_attributes",
              "infer_input_shapes",    "populate_node",
              "set_explicit_attribute"};

          // Extract subgraph if required
          if (ENV_PARAM(EXTRACT_SUBGRAPHS) == 1)
            task_list.push_back("extract_subgraph");

          // Append modifiers to task list
          task_list.insert(task_list.end() - 2, fuse_state->modifiers.begin(),
                           fuse_state->modifiers.end());

          task_runner(fuse_state, task_list);

          if (!fuse_state->generic_fusion)
            return false;

          fuse_state->node_builder.build();
          return true;
        });
  }

  void process(IPass& self, Graph& graph) { create_rule(&self)->apply(&graph); }

public:
  IPass& self_;
};

DEFINE_VAIP_PASS(Merge_node_builder, vaip_pass_dd_merge_fuse_general)
