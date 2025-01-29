/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

// test case 43: python main.py run 43
#include "glog/logging.h"
#include "vaip/vaip.hpp"
#include "vitis/ai/env_config.hpp"
DEF_ENV_PARAM(DEBUG_GRAPH_INPUT_ADD_TRANSPOSE, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_GRAPH_INPUT_ADD_TRANSPOSE) >= n)
namespace {
using namespace vaip_core;

static std::vector<int64_t>
reverse_transpose_shape(const std::vector<int64_t>& shape,
                        const std::vector<int64_t>& perm) {
  auto ret = std::vector<int64_t>(shape.size());
  CHECK_EQ(perm.size(), shape.size());
  for (auto i = 0u; i < perm.size(); ++i) {
    ret[perm[i]] = shape[i];
  }
  return ret;
}

struct GraphInputAddTranspose {
  GraphInputAddTranspose(IPass& self) : self_{self} {}
  void preprocess(IPass& self, Graph& graph) {
    graph_inputs_ = graph_get_inputs(graph);
    CHECK_GT(graph_inputs_.size(), 0);
    MY_LOG(1) << "graph_inputs_ size is " << graph_inputs_.size();

    std::vector<std::string> dest{"N", "C", "H", "W"};
    nchw_graph_input_.resize(graph_inputs_.size(), false);

    for (auto idx = 0u; idx < graph_inputs_.size(); ++idx) {
      auto pdenotation = node_arg_get_denotation(*graph_inputs_[idx]);
      if (nullptr == pdenotation) {
        LOG(INFO) << "get absent denotation for nodearg. "
                  << node_arg_as_string(*graph_inputs_[idx]);
        return;
      }
      if (pdenotation->size() != 4) {
        LOG(INFO) << "denotation size must be 4. "
                  << node_arg_as_string(*graph_inputs_[idx]);
        return;
      }
      MY_LOG(1) << "graph_inputs_[" << idx << "] denotation: " //
                << "size is " << pdenotation->size() << ". "   //
                << (*pdenotation)[0] << ", "                   //
                << (*pdenotation)[1] << ", "                   //
                << (*pdenotation)[2] << ", "                   //
                << (*pdenotation)[3];                          //
      if (*pdenotation != dest) {
        LOG(INFO) << "graph_inputs_[" << idx << "] denotation is not NCHW. "
                  << node_arg_as_string(*graph_inputs_[idx]);
        return;
      }
      nchw_graph_input_[idx] = true;
    }
  }

  void process(const IPass& self, Graph& graph) {
    // one assumption: all graph inputs must have the same denotation NCHW
    if (std::any_of(nchw_graph_input_.begin(), nchw_graph_input_.end(),
                    [](const bool flag) { return !flag; })) {
      LOG(INFO) << "not all graph inputs have the denotation NCHW";
      return;
    }

    for (auto graph_input : graph_inputs_) {
      auto consumers =
          graph_get_consumer_nodes(graph, node_arg_get_name(*graph_input));
      MY_LOG(1) << "consumers size is " << consumers.size();

      // build new transpose node
      std::vector<int64_t> perm{0, 3, 1, 2};
      auto shape = node_arg_get_shape_i64(*graph_input);
      if (nullptr == shape) {
        LOG(INFO) << "absent shape. " << node_arg_as_string(*graph_input);
        return;
      }
      auto new_shape = reverse_transpose_shape(*shape, perm);
      MY_LOG(1) << "graph input new shape: size is " //
                << new_shape.size() << ", shape[ "   //
                << new_shape[0] << ", "              //
                << new_shape[1] << ", "              //
                << new_shape[2] << ", "              //
                << new_shape[3] << " ]";
      auto& transpose_node =
          NodeBuilder(graph, self_)
              .set_op_type("Transpose", "")
              .set_input_node_args({graph_input})
              .clone_data_type(*graph_input)
              .add("perm", perm)
              .set_anchor_point3(*graph_input, {"transpose"}, *shape)
              .build();
      // correct graph input shape and denotation
      VAIP_ORT_API(node_arg_set_shape_i64)(*graph_input, new_shape);
      VAIP_ORT_API(node_arg_set_denotation)(*graph_input, {"N", "H", "W", "C"});

      // get new transpose node arg and correct its denotation
      auto& transpose_output_node_arg =
          node_get_output_node_arg(transpose_node);
      VAIP_ORT_API(node_arg_set_denotation)
      (transpose_output_node_arg, {"N", "C", "H", "W"});

      // replace consumer nodes
      for (auto node : consumers) {
        auto node_inputs = node_get_inputs(*node);
        std::vector<const NodeArg*> node_input_args(node_inputs.size());
        for (auto i = 0u; i < node_inputs.size(); i++) {
          if (node_inputs[i].node_arg == graph_input) {
            node_input_args[i] = &transpose_output_node_arg;
          } else {
            node_input_args[i] = node_inputs[i].node_arg;
          }
        }
        NodeBuilder(graph, self_)
            .set_input_node_args(node_input_args)
            .clone_op_type(*node)
            .clone_attrs(*node)
            .set_anchor_point1(*node)
            .build();
      }
    }
  }

  IPass& self_;
  std::vector<bool> nchw_graph_input_;
  std::vector<const NodeArg*> graph_inputs_;
};
} // namespace

DEFINE_VAIP_PASS(GraphInputAddTranspose, vaip_pass_graph_input_add_transpose)
