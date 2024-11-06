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

#include "vaip/vaip.hpp"
#include <glog/logging.h>

#include "../vaip_pass_py_ext/src/common.hpp"

DEF_ENV_PARAM(DEBUG_GRAPH_LABEL_EXAMPLE, "0")

namespace py = pybind11;
namespace {
using namespace vaip_core;

template <template <typename...> class V, typename T>
std::string vector_to_string(const V<T>& nodes) {
  std::stringstream ss;
  std::copy(nodes.begin(), nodes.end(), std::ostream_iterator<T>(ss, " "));
  return ss.str();
}

static bool as_boolean(py::object value) {
  bool ret = false;
  if (py::isinstance<py::bool_>(value)) {
    ret = py::cast<bool>(value);
  } else if (value.is_none()) {
    ret = false;
  } else {
    ret = true;
  }
  return ret;
}
struct NodeInputComparator {
  bool operator()(const NodeInput& a, const NodeInput& b) const {
    return int64_t(a.node) > int64_t(b.node) ||
           int64_t(a.node_arg) > int64_t(b.node_arg);
  }
};
typedef std::set<NodeInput, NodeInputComparator> NodeInputSet;

struct GraphFuse {
  std::vector<size_t> nodes;
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
  std::vector<std::string> constant_initializers;
};

static bool is_all_consumer_in_nodes(const std::vector<size_t>& nodes,
                                     const std::string& name,
                                     const Graph& graph) {
  auto consumers = graph_get_consumer_nodes(graph, name);
  std::vector<size_t> consumers_ids;
  std::for_each(consumers.begin(), consumers.end(), [&](const Node* n) {
    consumers_ids.push_back(VAIP_ORT_API(node_get_index)(*n));
  });
  return std::includes(nodes.begin(), nodes.end(), consumers_ids.begin(),
                       consumers_ids.end());
}

static GraphFuse NodeInputSet2GraphFuse(const Graph& graph,
                                        const NodeInputSet& nis) {
  GraphFuse gf;
  std::set<std::string> all_node_arg_names;
  std::set<const Node*> all_node;
  for (auto&& ni : nis) {
    all_node_arg_names.insert(node_arg_get_name(*ni.node_arg));
    if (ni.node == nullptr) {
      if (VAIP_ORT_API(node_arg_is_constant)(graph, *ni.node_arg)) {
        gf.constant_initializers.push_back(node_arg_get_name(*ni.node_arg));
      } else {
        gf.inputs.push_back(node_arg_get_name(*ni.node_arg));
      }
      continue;
    } else {
      all_node.insert(ni.node);
    }
  }
  std::vector<const Node*> my_node;
  for (auto&& n : all_node) {
    std::set<std::string> input_arg_names;
    for (auto&& na : node_get_input_node_args(*n)) {
      input_arg_names.insert(node_arg_get_name(*na));
    }
    if (std::includes(all_node_arg_names.begin(), all_node_arg_names.end(),
                      input_arg_names.begin(), input_arg_names.end())) {
      gf.nodes.push_back(VAIP_ORT_API(node_get_index)(*n));
      my_node.push_back(n);
    }
  }
  for (auto&& n : my_node) {
    for (auto&& ni : node_get_inputs(*n)) {
      auto name = node_arg_get_name(*ni.node_arg);
      if (std::find(gf.constant_initializers.begin(),
                    gf.constant_initializers.end(),
                    name) == gf.constant_initializers.end() &&
          std::find(gf.nodes.begin(), gf.nodes.end(),
                    VAIP_ORT_API(node_get_index)(*ni.node)) == gf.nodes.end()) {
        gf.inputs.push_back(name);
      }
    }
    for (auto&& na : node_get_output_node_args(*n)) {
      auto name = node_arg_get_name(*na);
      if (!is_all_consumer_in_nodes(gf.nodes, name, graph)) {
        gf.outputs.push_back(name);
      }
    }
  }
  return gf;
}

static std::vector<GraphFuse>
NodeInputSet2GraphFuse(const Graph& graph,
                       const std::vector<NodeInputSet>& nis) {
  std::vector<GraphFuse> gfs;
  std::for_each(nis.begin(), nis.end(), [&](const NodeInputSet& n) {
    gfs.push_back(NodeInputSet2GraphFuse(graph, n));
  });
  return gfs;
}

static bool need_marge(GraphFuse gf1, GraphFuse gf2) {
  std::sort(gf1.inputs.begin(), gf1.inputs.end());
  std::sort(gf1.outputs.begin(), gf1.outputs.end());
  std::sort(gf2.inputs.begin(), gf2.inputs.end());
  std::sort(gf2.outputs.begin(), gf2.outputs.end());
  std::vector<std::string> intersection;
  std::set_intersection(gf1.inputs.begin(), gf1.inputs.end(),
                        gf2.outputs.begin(), gf2.outputs.end(),
                        std::back_inserter(intersection));
  std::set_intersection(gf2.inputs.begin(), gf2.inputs.end(),
                        gf1.outputs.begin(), gf1.outputs.end(),
                        std::back_inserter(intersection));
  return !intersection.empty();
}

static void marge(GraphFuse& gf1, GraphFuse gf2, const Graph& graph) {
  gf1.constant_initializers.insert(gf1.constant_initializers.end(),
                                   gf2.constant_initializers.begin(),
                                   gf2.constant_initializers.end());
  gf1.nodes.insert(gf1.nodes.end(), gf2.nodes.begin(), gf2.nodes.end());
  gf1.inputs.insert(gf1.inputs.end(), gf2.inputs.begin(), gf2.inputs.end());
  gf1.outputs.insert(gf1.outputs.end(), gf2.outputs.begin(), gf2.outputs.end());
  std::sort(gf1.constant_initializers.begin(), gf1.constant_initializers.end());
  std::sort(gf1.nodes.begin(), gf1.nodes.end());
  std::sort(gf1.inputs.begin(), gf1.inputs.end());
  std::sort(gf1.outputs.begin(), gf1.outputs.end());
  gf1.constant_initializers.erase(std::unique(gf1.constant_initializers.begin(),
                                              gf1.constant_initializers.end()),
                                  gf1.constant_initializers.end());
  gf1.nodes.erase(std::unique(gf1.nodes.begin(), gf1.nodes.end()),
                  gf1.nodes.end());
  gf1.inputs.erase(std::unique(gf1.inputs.begin(), gf1.inputs.end()),
                   gf1.inputs.end());
  gf1.outputs.erase(std::unique(gf1.outputs.begin(), gf1.outputs.end()),
                    gf1.outputs.end());

  auto it = std::remove_if(
      gf1.inputs.begin(), gf1.inputs.end(), [&gf1](std::string name) {
        return std::find(gf1.outputs.begin(), gf1.outputs.end(), name) !=
               gf1.outputs.end();
      });
  gf1.inputs.erase(it, gf1.inputs.end());
  it = std::remove_if(gf1.outputs.begin(), gf1.outputs.end(),
                      [&](std::string name) {
                        return is_all_consumer_in_nodes(gf1.nodes, name, graph);
                      });
  gf1.outputs.erase(it, gf1.outputs.end());
}

struct GraphLabelExample {
  GraphLabelExample(IPass& self) : self_(self) {
    self.add_context_resource("py_ext.interpreter",
                              vaip_core::init_interpreter());
  }

  void process(IPass& pass, Graph& graph) {
    auto module_vaip_pass_py_ext = py::module::import("voe.voe_cpp2py_export");
    module_vaip_pass_py_ext.attr("_init_vaip_ort_api")(
        py::capsule(vaip_core::api()));
    std::vector<NodeInputSet> nodeinput_label;
    auto& pass_proto = self_.get_pass_proto();
    for (auto& py_ext_proto : pass_proto.graph_label_param().py_ext()) {
      auto m = py::module::import(py_ext_proto.module_name().c_str());
      py::object rules1 = m.attr(py_ext_proto.method_name().c_str())();
      CHECK(py::isinstance<py::list>(rules1))
          << py_ext_proto.module_name() << "." << py_ext_proto.method_name()
          << " must return a list";
      py::list rules = rules1;
      std::vector<std::unique_ptr<BaseRule>> cpp_rules;
      auto is_rule = py::module::import("voe.rule_ext").attr("is_rule");
      for (auto i = 0u; i < rules.size(); ++i) {
        auto rule = rules[i];
        CHECK(is_rule(rule)) << " rule[" << i << "] must be a vaip.rule.Rule";
        auto builder = PatternBuilder();
        auto ppass = &pass;
        cpp_rules.push_back(Rule::create_rule(
            parse_pattern0(builder, rule.attr("pattern")()),
            [rule, builder, ppass, &nodeinput_label](onnxruntime::Graph* graph,
                                                     binder_t& binder) -> bool {
              py::dict py_binder;
              NodeInputSet binder_nodeinputs;
              for (auto& binding : builder.bindings()) {
                py_binder[binding.first.c_str()] =
                    py::cast(binder[binding.second]);
                binder_nodeinputs.insert(binder[binding.second]);
              }
              rule.attr("initialize")(GraphWrapper{*graph}, py::cast(ppass),
                                      py_binder);
              if (as_boolean(rule.attr("_where")(**py_binder))) {
                nodeinput_label.push_back(binder_nodeinputs);
              }
              return false;
            }));
      }
      BaseRule::create_rule_chain(std::move(cpp_rules))->apply(&graph);
    }
    // graph fusion
    auto gfs = NodeInputSet2GraphFuse(graph, nodeinput_label);
    std::vector<GraphFuse> graph_fusion;
    for (; graph_fusion.size() != gfs.size(); gfs = graph_fusion) {
      graph_fusion.clear();
      for (const auto& gf : gfs) {
        bool need_break = false;
        for (auto& gf2 : graph_fusion) {
          if (need_marge(gf, gf2)) {
            need_break = true;
            marge(gf2, gf, graph);
            break;
          }
        }
        if (!need_break)
          graph_fusion.push_back(gf);
      }
    }
    int i = 0;
    if (ENV_PARAM(DEBUG_GRAPH_LABEL_EXAMPLE)) {
      for (auto&& gf : graph_fusion) {
        LOG(INFO) << "subgraph index: " << i++;
        LOG(INFO) << " nodes: " << vector_to_string(gf.nodes);
        LOG(INFO) << " inputs: " << vector_to_string(gf.inputs);
        LOG(INFO) << " outputs: " << vector_to_string(gf.outputs);
        LOG(INFO) << " constant_initializers: "
                  << vector_to_string(gf.constant_initializers) << "\n";
      }
    }

    for (auto&& gf : graph_fusion) {
      VAIP_ORT_API(graph_fuse)
      (graph, gf.outputs[0], "None", gf.nodes, gf.inputs, gf.outputs,
       gf.constant_initializers);
    }
  }

private:
  IPass& self_;
};
} // namespace
DEFINE_VAIP_PASS(GraphLabelExample, vaip_pass_graph_label_example)
