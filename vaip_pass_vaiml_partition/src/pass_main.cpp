/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

#include "vaiml_subgraph_processor.h"
#include "vaip/vaip.hpp"
#include "vitis/ai/env_config.hpp"
#include <glog/logging.h>

using VaimlTensorShape = std::vector<int64_t>;
using VaimlShapeDict = std::map<std::string, VaimlTensorShape>;
using VaimlShapeVec = std::vector<VaimlTensorShape>;

namespace {
using namespace vaip_core;
using namespace vaip_vaiml_subgraph_processor;
struct VaimlPartition {

  VaimlPartition(IPass& self) : self_{self} {}

  bool process(IPass& self, Graph& graph) {
    MY_LOG(1) << "Hello from vaip_pass_vaiml_parition";
    // auto& vaiml_proto = self.get_pass_proto().vaiml_config();

    auto subgraph_processor =
        std::make_unique<VaimlSubgraphProcessor>(graph, self);
    subgraph_processor->process();

    return true;
  }

  IPass& self_;
};
} // namespace
DEFINE_VAIP_PASS(VaimlPartition, vaip_pass_vaiml_partition)
