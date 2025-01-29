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
namespace vaip_dd_merge_qmhagrpb {
using namespace vaip_core;
struct DdMergeQmhagrpbProcessor {
public:
  DdMergeQmhagrpbProcessor(
      IPass& self, onnxruntime::Graph* graph, binder_t* binder,
      const std::unordered_map<std::string, std::string>& binder_params);
  std::vector<NodeArg*> process(int output_pat_id);

private:
  NodeInput get_matched_node(const std::string& pattern_name);

private:
  IPass& self_;
  onnxruntime::Graph* graph_;
  binder_t* binder_;
  const std::unordered_map<std::string, std::string>& binder_params_;
};
} // namespace vaip_dd_merge_qmhagrpb
