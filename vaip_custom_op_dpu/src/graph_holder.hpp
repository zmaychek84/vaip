/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

#pragma once
#include "vaip/vaip.hpp"
#include <memory>
#include <string>
#include <vaip/my_ort.h>
#include <vector>
namespace xir {
class Graph;
class Subgraph;
} // namespace xir
namespace vaip_core {

class GraphHolder {
public:
  explicit GraphHolder(const vaip_core::PassContext& pass_context,
                       const std::string& filename,
                       std::string& decryption_key);

  ~GraphHolder() = default;

public:
  const xir::Graph* get_graph() const;
  std::vector<const xir::Subgraph*> get_subgraphs() const;
  const xir::Graph* release() { return graph_.release(); }

private:
  void init_subgraph();

private:
  std::unique_ptr<const xir::Graph> graph_;
  std::vector<const xir::Subgraph*> subgraphs_;
};

} // namespace vaip_core
