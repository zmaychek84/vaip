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
// #include "vaip/pattern_zoo.hpp"
#ifdef __GNUC__
#  pragma GCC diagnostic pop
#endif

#include "vaip/dd/dd_utils.hpp"
#include "vaip/vaip.hpp"
#include "vitis/ai/env_config.hpp"

#include <cmath>
#include <glog/logging.h>
#include <unordered_map>
#include <vector>

DEF_ENV_PARAM(DEBUG_DD_MERGE_QDQ_UNSQUEEZE, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_DD_MERGE_QDQ_UNSQUEEZE) >= n)
namespace {
using namespace vaip_core;

static void add_node_attr_qdq_unsqueeze_single_node(NodeBuilder& building_node,
                                                    const NodeInput& concat) {
  std::vector<std::string> nodes = {node_arg_get_name(*concat.node_arg)};
  building_node.add("nodes", nodes);
}

// gets all the nodes that are matched with binder and stores in nodes attribute
// of new node

static void add_node_attr_qdq_unsqueeze(onnxruntime::Graph* graph,
                                        NodeBuilder& building_node,
                                        binder_t& binder) {
  std::vector<std::string> nodes = vaip::dd::get_node_names(graph, binder);

  building_node.add("nodes", nodes);
}

static void update_shape(NodeBuilder& building_node, const NodeInput& output) {
  auto shape = *node_arg_get_shape_i64(*output.node_arg);
  building_node.add("orig_output_shape", shape);
}
std::vector<const NodeArg*>
get_matched_input(const std::vector<std::shared_ptr<Pattern>>& dequant,
                  const binder_t& binder) {
  std::vector<const NodeArg*> ret;
  for (auto p : dequant) {
    auto node_input = binder[p->get_id()];
    if (!node_input.node_arg) { // no more input
      break;
    }
    ret.push_back(node_input.node_arg);
  }
  return ret;
}
/* Pass for Q+DQ+Unsqueeze no op in ort optimized PSU */
struct MergeQDQUnsqueeze {
  MergeQDQUnsqueeze(IPass& self) : self_{self} {}
  std::unique_ptr<Rule> create_rule(IPass* self) {
    auto builder = vaip_core::PatternBuilder();
    auto constant_0 = builder.wildcard();
    auto constant_1 = builder.constant();
    auto constant_2 = builder.constant();
    auto com_microsoft_quantizeLinear_0 = builder.node2(
        "com.microsoft:QuantizeLinear", {constant_0, constant_1, constant_2});
    auto constant_4 = builder.constant();
    auto constant_5 = builder.constant();
    auto com_microsoft_DequantizeLinear_1 =
        builder.node2("com.microsoft:DequantizeLinear",
                      {com_microsoft_quantizeLinear_0, constant_4, constant_5});
    auto constant_un = builder.constant();
    auto pattern_ = builder.node2(
        "Unsqueeze", {com_microsoft_DequantizeLinear_1, constant_un});
    CHECK(pattern_ != nullptr) << "Pattern returned is null";
    return Rule::create_rule(
        pattern_, [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
          auto ni_output = binder[pattern_->get_id()];
          auto ni_input = binder[constant_0->get_id()];
          std::vector<std::string> in_dtypes = {"float32"};
          std::vector<std::string> out_dtypes = {"float32"};
          auto node_name = node_arg_get_name(*ni_output.node_arg);
          std::vector<std::string> inputs = {
              node_arg_get_name(*ni_input.node_arg)};
          std::vector<std::string> outputs = {node_name};
          auto constant_initializers = std::vector<std::string>{};
          auto [meta_def, err] =
              self->try_fuse(*graph, "QDQUnsqueeze" + node_name, inputs,
                             outputs, constant_initializers, "QDQUNSQUEEZE");
          if (meta_def == nullptr) {
            LOG(FATAL) << "Cannot fuse DQ:  " << err.comments;
          }
          [[maybe_unused]] auto& fused_node =
              self->fuse(*graph, std::move(*meta_def));
          return true;
        });
  }
  void process(IPass& self, Graph& graph) { create_rule(&self)->apply(&graph); }

public:
  IPass& self_;
};
} // namespace

DEFINE_VAIP_PASS(MergeQDQUnsqueeze, vaip_pass_dd_merge_qdq_unsqueeze)
