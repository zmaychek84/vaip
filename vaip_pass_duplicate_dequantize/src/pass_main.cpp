/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */
#include <glog/logging.h>

#include "vitis/ai/env_config.hpp"
#include <vaip/vaip.hpp>

DEF_ENV_PARAM(DEBUG_DUPLICATE_DEQUANTIZE, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_DUPLICATE_DEQUANTIZE) >= n)

/**
 * test case: any cases
 *
 *
 * Replace pattern:
 *
 * duplicate DequantizeLinear if there are more than one consumers.
 *
 */

// add the following line in your vaip_config.json
/*
    { "name": "vaip_pass_duplicate_dequantize",
       "plugin": "vaip-pass_duplicate_dequantize",
       "disabled": false
    }
*/
namespace {
using namespace vaip_core;
struct DuplicateDequantize {
  DuplicateDequantize(IPass& self) : self_{self} {}
  std::unique_ptr<Rule> create_rule(IPass* self) {
    static const std::string DUPLICATE = "duplicate";
    auto builder = PatternBuilder();
    std::shared_ptr<Pattern> input_0 = builder.wildcard();
    std::shared_ptr<Pattern> input_1 = builder.wildcard();
    std::shared_ptr<Pattern> input_2 = builder.wildcard();
    std::shared_ptr<Pattern> pattern_ = builder.node2(
        "com.microsoft:DequantizeLinear", {input_0, input_1, input_2});
    return Rule::create_rule(
        pattern_, [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
          auto& dq_node = *binder[pattern_->get_id()].node;
          auto& dq_node_arg = *binder[pattern_->get_id()].node_arg;
          auto& dq_node_arg_name = node_arg_get_name(dq_node_arg);

          auto consumers = graph_get_consumer_nodes(*graph, dq_node_arg_name);
          auto num_of_consumers = consumers.size();
          auto ret = false;
          for (auto i = 1u; i < num_of_consumers; ++i) {
            CHECK(consumers[i] != nullptr) << "graph need to be resolved.";
            auto& consumer = *consumers[i];
            auto& duplicate_dq = NodeBuilder(*graph, self_)
                                     .clone_inputs(dq_node)
                                     .clone_op_type(dq_node)
                                     .clone_data_type(dq_node)
                                     .clone_attrs(dq_node)
                                     .set_anchor_point2(dq_node_arg, DUPLICATE)
                                     .build();
            auto& duplicate_arg_dq =
                node_get_first_output_node_arg(duplicate_dq);
            auto input_node_args = node_get_input_node_args(consumer);
            for (auto& input_node_arg : input_node_args) {
              if (input_node_arg == &dq_node_arg) {
                input_node_arg = &duplicate_arg_dq;
              }
            }
            MY_LOG(1) << "duplicate dq: " << node_as_string(duplicate_dq)
                      << "consumer: " << node_as_string(consumer);
            NodeBuilder(*graph, self_)
                .set_input_node_args(input_node_args)
                .clone_op_type(consumer)
                .clone_attrs(consumer)
                .set_anchor_point1(consumer)
                .build();
            graph_resolve(*graph);
            ret = true;
          }
          if (ret) {
            MY_LOG(1) << "done ret=" << ret;
          }
          return ret;
        });
  }
  // apply the rule
  void process(IPass& self, Graph& graph) {
    MY_LOG(1) << self_.get_pass_proto().name() << "["
              << self_.get_pass_proto().plugin() << "] start processing graph";
    create_rule(&self)->apply(&graph);
    MY_LOG(1) << self_.get_pass_proto().name() << "["
              << self_.get_pass_proto().plugin() << "] finish processing graph";
  }

  IPass& self_;
};
} // namespace

DEFINE_VAIP_PASS(DuplicateDequantize, vaip_pass_duplicate_dequantize)
