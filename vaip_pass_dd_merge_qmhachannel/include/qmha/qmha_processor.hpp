/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */
#include "vaip/vaip.hpp"
#include <cstdint>
#include <glog/logging.h>
#include <string>
#include <unordered_map>
#include <vector>
namespace vaip_dd_merge_qma {
using namespace vaip_core;
struct DdMergeQmhaProcessor {
public:
  DdMergeQmhaProcessor(
      IPass& self, onnxruntime::Graph* graph, binder_t* binder,
      const std::unordered_map<std::string, std::vector<std::string>>&
          binder_params);
  const NodeArg& process(int output_pat_id);
  const NodeArg& process_m7h4xjg(int output_pat_id);

private:
  float node_arg_get_const_data_as_float(const std::string& name, size_t index);
  uint16_t node_arg_get_const_data_as_u16(const std::string& name,
                                          size_t index);
  int64_t get_k_dim(NodeInput ni1, NodeInput ni2);
  const NodeArg& create_node_arg(onnxruntime::Graph& graph,
                                 const std::string& name,
                                 const std::vector<int64_t>& shape,
                                 const std::vector<int64_t>& value);

  NodeInput get_node_input(const std::string& name, size_t index) const;

private:
  IPass& self_;
  onnxruntime::Graph* graph_;
  binder_t* binder_;
  const std::unordered_map<std::string, std::vector<std::string>>&
      binder_params_;
};
} // namespace vaip_dd_merge_qma
