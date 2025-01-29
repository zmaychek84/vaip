/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

#pragma once

#include "vitis/ai/env_config.hpp"
#include <filesystem>
#include <pybind11/cast.h>
#include <pybind11/embed.h>
#include <pybind11/eval.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <unordered_map>
DEF_ENV_PARAM(DEBUG_VAIP_PY_EXT, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_VAIP_PY_EXT) >= n)
#include "vaip/vaip.hpp"
#include <glog/logging.h>
namespace py = pybind11;
namespace vaip_core {
struct GraphWrapper {
  GraphWrapper(Graph& g) : graph{g} {}
  Graph& graph;
};
struct NodeWrapper {
  Node& node;
};
struct NodeArgWrapper {
  NodeArg& node_arg;
};

struct ModelWrapper {
  ModelPtr model;
};

static std::shared_ptr<Pattern> parse_pattern0(PatternBuilder& builder,
                                               py::object pattern_def) {
  auto ret = std::shared_ptr<Pattern>();
  auto m = py::module::import("voe.pattern");
  auto is_pattern_f = m.attr("is_pattern");
  CHECK(py::cast<bool>(is_pattern_f(pattern_def)))
      << "parse error: invalid pattern ; pattern=" << pattern_def;
  auto pattern_json = py::cast<std::string>(pattern_def);
  MY_LOG(1) << "create pattern by json : " << pattern_json;
  return builder.create_by_json(pattern_json);
}

} // namespace vaip_core
