/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */
#pragma once
#include <fstream>
#include <glog/logging.h>
#include <iostream>

#include "vaip/vaip.hpp"
#include "vitis/ai/env_config.hpp"
using namespace vaip_core;
namespace vaip_vaiml_subgraph_processor {

inline void collect_subgraph_nodes(std::set<std::string>& sg_node_set,
                                   binder_t& binder,
                                   const std::set<int>& excludes = {}) {
  for (auto iter = binder.begin(); iter != binder.end(); iter++) {
    if (excludes.count(iter->first)) {
      // std::cout << "skipping " <<
      // VAIP_ORT_API(node_get_name)(*iter->second.node) << std::endl;
      continue;
    }
    if (iter->second.node == nullptr) {
      // std::cout << "could be input or constant: "
      // <<node_arg_get_name(*iter->second.node_arg) << std::endl;
    } else {
      // std::cout << "could be a node: " << iter->first << " " <<
      // VAIP_ORT_API(node_get_name)(*iter->second.node) << std::endl;
      sg_node_set.insert(VAIP_ORT_API(node_get_name)(*iter->second.node));
    }
  }
}

template <typename... Args>
std::string str_fmt(const std::string& format, Args... args) {
  int size_s = std::snprintf(nullptr, 0, format.c_str(), args...) +
               1; // Extra space for '\0'
  if (size_s <= 0) {
    throw std::runtime_error("Error during str formatting.");
  }
  auto size = static_cast<size_t>(size_s);
  std::unique_ptr<char[]> buf(new char[size]);
  std::snprintf(buf.get(), size, format.c_str(), args...);
  return std::string(buf.get(),
                     buf.get() + size - 1); // We don't want the '\0' inside
}
} // namespace vaip_vaiml_subgraph_processor
