/*
 *     The Xilinx Vitis AI Vaip in this distribution are provided under the
 * following free and permissive binary-only license, but are not provided in
 * source code form.  While the following free and permissive license is similar
 * to the BSD open source license, it is NOT the BSD open source license nor
 * other OSI-approved open source license.
 *
 *      Copyright (C) 2022 Xilinx, Inc. All rights reserved.
 *      Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights
 * reserved.
 *
 *      Redistribution and use in binary form only, without modification, is
 * permitted provided that the following conditions are met:
 *
 *      1. Redistributions must reproduce the above copyright notice, this list
 * of conditions and the following disclaimer in the documentation and/or other
 * materials provided with the distribution.
 *
 *      2. The name of Xilinx, Inc. may not be used to endorse or promote
 * products redistributed with this software without specific prior written
 * permission.
 *
 *      THIS SOFTWARE IS PROVIDED BY XILINX, INC. "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL XILINX, INC. BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 *      PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE
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
