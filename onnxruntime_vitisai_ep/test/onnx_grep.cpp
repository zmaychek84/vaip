/*
 *     The Xilinx Vitis AI Vaip in this distribution are provided under the
 * following free and permissive binary-only license, but are not provided in
 * source code form.  While the following free and permissive license is similar
 * to the BSD open source license, it is NOT the BSD open source license nor
 * other OSI-approved open source license.
 *
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
// clang-format off
#include <glog/logging.h>
#include <exception>
#include <limits>
#include "initialize_vaip.hpp"

#include <vaip/vaip.hpp>
#include <vaip/vaip_ort.hpp>
#include <vaip/pattern_zoo.hpp>
#include "../include/onnxruntime_vitisai_ep/onnxruntime_vitisai_ep.hpp"
#include "node.hpp"
extern "C" {
#include "./getopt.h"
}

#ifdef CREATE_DUMMY_SESSION
#include <codecvt>
#include <locale>
using convert_t = std::codecvt_utf8<wchar_t>;
std::wstring_convert<convert_t, wchar_t> strconverter;
#endif// CREATE_DUMMY_SESSION

static bool endsWith(const std::string &fullString, const std::string &ending) {
  if (fullString.size() >= ending.size()) {
      return (fullString.compare(fullString.length() - ending.size(), ending.size(), ending) == 0);
  } else {
      return false;
  }
}

static std::shared_ptr<vaip_core::Pattern> get_pattern(const std::string& file) {
    auto builder = vaip_core::PatternBuilder ();
    auto ret = std::shared_ptr<vaip_core::Pattern>();
  // see test_conv_pattern.py as an example
  if (endsWith(file, std::string(".py"))) {
#ifdef ENABLE_PYTHON
      ret =  builder.create_by_py(vaip_core::slurp(file));
#else
    throw std::runtime_error("Unsupported pattern data type");
#endif
  } else if (endsWith(file, std::string(".json"))) {
      ret =  builder.create_by_json(vaip_core::slurp(file));
  } else {
#include "./onnx_grep_cxx_pattern.h.inc"
      auto find_pattern = vaip::pattern_zoo::get_pattern(file);
      if(find_pattern) {
          LOG(INFO) << "use predfined pattern: " << file;
          ret = find_pattern;
      } else {
          LOG(WARNING) << "cannot pattern: " << file << " fallback to default pattern";
      }
  }
  return ret;
}

// clang-format on
static void list_all_predefined_pattern() {
  auto list = vaip::pattern_zoo::pattern_list();
  std::cout << "available predefined patterns:" << std::endl;
  for (auto& x : list) {
    std::cout << "    " << x << std::endl;
  }
  return;
}

int main(int argc, char* argv[]) {
  std::cout << "- ONNX Grep utility ..." << std::endl;
  try {
    auto file = std::string();
    auto pattern = std::string();
    auto node_arg = std::string();
    auto opt_verbose = false;
    int opt = 0;
    while ((opt = getopt(argc, argv, "p:f:n:lv")) != -1) {
      switch (opt) {
      case 'f': {
        file = std::string(optarg);
        break;
      }
      case 'p': {
        pattern = std::string(optarg);
        break;
      }
      case 'n': {
        node_arg = std::string(optarg);
        break;
      }
      case 'l': {
        list_all_predefined_pattern();
        exit(0);
        break;
      }
      case 'v': {
        opt_verbose = true;
        break;
      }
      }
    }
    Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "onnx_grep");
    initialize_vaip();

    CHECK_NE(file, "");

    auto p = get_pattern(pattern);
    if (p == nullptr) {
      LOG(ERROR) << "no pattern";
      return 1;
    }
    if (opt_verbose) {
      std::cout << "pattern is " << (void*)p.get() << std::endl;
      std::cout << "pattern is " << p->debug_string() << std::endl;
    }
    if (!node_arg.empty()) {
      vaip_core::Pattern::enable_trace(1);
    }
    auto model = vaip_core::model_load(std::filesystem::path(file).u8string());
    auto& graph = VAIP_ORT_API(model_main_graph)(*model);
    vaip_core::graph_resolve(graph, true);
    if (!node_arg.empty()) {
      auto node_found = VAIP_ORT_API(graph_producer_node)(graph, node_arg);
      CHECK(node_found != nullptr)
          << "cannot find node arg. node_arg=" << node_arg;
    }
    for (auto index : vaip_core::graph_get_node_in_topoligical_order(graph)) {
      auto node = VAIP_ORT_API(graph_get_node)(graph, index);
      CHECK(node != nullptr);
      auto this_node_arg_name = vaip_core::node_get_first_output_name(*node);
      // node_arg.empty() means user does not specify `-n` for
      // tracing, we try to search for all possible matched node.
      //
      // if it is not empty, we only trace the node whose name is
      // `node_arg`, i.e. this_node_arg_name == node_arg.
      if (node_arg.empty() || (this_node_arg_name == node_arg)) {
        auto bind = p->match(graph, *node);
        if (bind != nullptr) {
          LOG(INFO) << "find node: " << vaip_core::node_as_string(*node);
          if (opt_verbose) {
            for (auto ni : *bind) {
              LOG(INFO) << "pattern id: " << ni.first << " node_arg: "
                        << vaip_core::node_arg_as_string(*ni.second.node_arg);
            }
          }
        }
      }
    }
  } catch (const std::exception& e) {
    std::cerr << "exception occurs : " << e.what() << "\n";
  }

  return 0;
}

#include "./getopt.c"
