/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

/**
 * GraphOutputAddNode pattern pass
 * X : wildcard()
 * From : X,
 *
 * To  : X + identity, if X is graph output and X's consumers >= 1
 *
 */
#pragma once
#include "vaip/vaip.hpp"
namespace vaip_pass_graph_output_add_node {
using namespace vaip_core;
class GraphOutputAddNodeRule : public Rule {
public:
  GraphOutputAddNodeRule();

private:
  virtual const Pattern* pattern() const override;
  virtual bool action(onnxruntime::Graph* graph,
                      binder_t& binder) const override;

private:
  std::shared_ptr<Pattern> output_; // wildcard
};
} // namespace vaip_pass_graph_output_add_node
