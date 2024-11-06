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

#include <glog/logging.h>

#include <iostream>
#include <unordered_set>

#include "vaip/xir_headers.hpp"
using namespace std;

struct SubgraphInfo {
  std::string name;
  std::unordered_set<std::string> ops;
  std::vector<SubgraphInfo> children;
  std::unique_ptr<xir::Attrs> attrs;
};
// static std::ostream& operator<<(std::ostream& s,
//                                 const std::unordered_set<std::string>& v) {
//   s << "[";
//   for (auto c : v) {
//     s << c << ", ";
//   }
//   s << "]";
//   return s;
// }
static std::ostream& operator<<(std::ostream& s,
                                const std::vector<std::string>& v) {
  s << "[";
  for (auto c = 0u; c < v.size(); ++c) {
    if (c != 0) {
      s << ",";
    }
    s << v[c];
  }
  s << "]";
  return s;
}
static std::ostream& operator<<(std::ostream& s, const SubgraphInfo& v) {
  s << "{";
  s << " name : " << v.name << " ,\n";

  if (!v.children.empty()) {
    s << " children : size " << v.children.size() << " : [";
    for (auto& child : v.children) {
      s << child.name << ",";
    }
    s << "] , \n";
  }
  s << " ops_size: " << v.ops.size() << " \n";
  s << "} \n";
  for (auto& child : v.children) {
    s << child;
  }
  return s;
}
// namespace
static void clone_op(xir::Graph* graph, const xir::Op* op) {
  auto name = op->get_name();
  auto type = op->get_type();
  auto attrs = op->get_attrs();
  auto output_tensor_attrs = op->get_output_tensor()->get_attrs();

  auto input_ops_map = std::map<std::string, std::vector<xir::Op*>>();
  auto args = op->get_input_ops();
  for (const auto& input_arg : args) {
    auto ops = std::vector<xir::Op*>();
    for (const auto& arg_op : input_arg.second) {
      auto op_tmp = graph->get_op(arg_op->get_name());
      if (nullptr == op_tmp) {
        // create data-fix op for input op
        auto data_attrs = xir::Attrs::create();
        data_attrs->set_attr<std::string>("device", "USER");
        data_attrs->set_attr<std::vector<int>>(
            "shape", arg_op->get_output_tensor()->get_shape());
        data_attrs->set_attr<std::string>(
            "data_type",
            arg_op->get_output_tensor()->get_data_type().to_string());

        auto new_op = graph->add_op(arg_op->get_name(), "data-fix",
                                    std::move(data_attrs), {});
        new_op->get_output_tensor()->set_attrs(
            std::move(arg_op->get_output_tensor()->get_attrs()));
        ops.push_back(new_op);
      } else {
        ops.push_back(op_tmp);
      }
    }
    input_ops_map.insert(std::make_pair(input_arg.first, ops));
  }
  auto new_op = graph->add_op(name, type, std::move(attrs), input_ops_map);
  new_op->get_output_tensor()->set_attrs(std::move(output_tensor_attrs));
}

std::unique_ptr<xir::Graph> clone_dpu_ops(const xir::Subgraph& dpu_subgraph) {
  auto graph = xir::Graph::create(dpu_subgraph.get_name());
  auto ops = dpu_subgraph.topological_sort();
  for (auto op : ops) {
    clone_op(graph.get(), op);
  }

  return graph;
}

using subgraph_path_t = std::vector<std::string>;
/// std::unique_ptr<SubgraphInfo> subgraph = create_subgraph_info(graph);

/// we assume that all ops belong to its own subgraph, the subgrahp contains
/// only one op, i.e. just after calling Subgraph::create_children()
static xir::Subgraph*
create_child_subgraph(xir::Subgraph* parent,
                      const std::unordered_set<std::string>& ops) {
  std::set<xir::Subgraph*> children;
  auto graph = parent->get_graph();
  for (auto& op : ops) {
    auto xir_op = graph->get_op(op);
    CHECK(xir_op != nullptr);
    children.insert(graph->get_leaf_subgraph(xir_op));
  }
  return parent->merge_children(children);
}

static void create_subgraph_tree1(xir::Subgraph* current_subgraph,
                                  SubgraphInfo& subgraph_info) {
  bool is_leaf_subgraph = subgraph_info.children.empty();
  if (!is_leaf_subgraph) {
    if (current_subgraph->is_leaf()) {
      LOG(INFO) << "create_children: " << current_subgraph->get_name();
      current_subgraph->create_children();
    }
    for (auto& child : subgraph_info.children) {
      auto child_subgraph = create_child_subgraph(current_subgraph, child.ops);
      create_subgraph_tree1(child_subgraph, child);
    }
    // we need to do nothing, and all child subgraphs are created.
  }
  LOG_IF(INFO, false) << "hello :" << (void*)current_subgraph << " attrs.get() "
                      << (void*)subgraph_info.attrs.get();
  current_subgraph->set_attrs(std::move(subgraph_info.attrs));
  current_subgraph->set_name(subgraph_info.name);
  return;
}
void create_subgraph_tree(xir::Graph& graph, SubgraphInfo& subgraph_info) {
  LOG(INFO) << "create subgraph tree:  " << subgraph_info.name
            << ", children size " << subgraph_info.children.size()
            << ", ops size " << subgraph_info.ops.size();
  create_subgraph_tree1(graph.get_root_subgraph(), subgraph_info);
}

static void extract_subgraph_info1(const std::string& op_name,
                                   const subgraph_path_t& subgraph_path,
                                   SubgraphInfo& current_info,
                                   size_t subgraph_path_index) {
  // add op_name anyway
  current_info.ops.insert(op_name);
  // if leaf subgraph, return
  if (subgraph_path_index == subgraph_path.size()) {
    return;
  }
  // find next level subgraph info
  auto current_subgraph_name = subgraph_path[subgraph_path_index];
  auto begin = current_info.children.begin();
  auto end = current_info.children.end();
  auto child_info = std::find_if(
      begin, end, [&current_subgraph_name](const SubgraphInfo& info) {
        return info.name == current_subgraph_name;
      });
  SubgraphInfo* pinfo = nullptr;

  // create next level subgraph info if not found.
  if (child_info == end) {
    current_info.children.emplace_back();
    pinfo = &current_info.children.back();
    pinfo->name = current_subgraph_name;
  } else {
    pinfo = &(*child_info);
  }
  extract_subgraph_info1(op_name, subgraph_path, *pinfo,
                         subgraph_path_index + 1);
}

/// we assume that the graph has no subgraphs
/// and each ops has a attr, "subgraph_path", and we extract subgrahp info from
/// the "subgraph_path'
static SubgraphInfo extract_subgraph_info(const xir::Graph& graph) {
  auto ops = graph.topological_sort();
  auto ret = SubgraphInfo{};
  ret.name = "root";
  ret.attrs = xir::Attrs::create();
  for (auto& op : ops) {
    auto subgraph_path = op->get_attr<subgraph_path_t>("subgraph_path");
    extract_subgraph_info1(op->get_name(), subgraph_path, ret, 0u);
  }
  return ret;
}

static subgraph_path_t get_subgraph_path(const xir::Graph& graph,
                                         const xir::Op& op) {
  auto ret = subgraph_path_t{};
  auto subgraph = graph.get_leaf_subgraph(&op);
  while (subgraph != nullptr &&
         !subgraph->is_root()) { // root is not include in subgraph path.
    ret.push_back(subgraph->get_name());
    subgraph = subgraph->get_parent();
  }
  std::reverse(ret.begin(), ret.end());
  return ret;
}

static void denotate_subgraph_path(xir::Graph& dst_graph,
                                   const xir::Graph& src_graph) {
  auto ops = dst_graph.topological_sort();
  for (auto& op : ops) {
    auto src_op = src_graph.get_op(op->get_name());
    CHECK(src_op != nullptr);
    auto subgraph_path = get_subgraph_path(src_graph, *src_op);
    op->set_attr<subgraph_path_t>("subgraph_path", subgraph_path);
  }
}

static void fill_in_subgraph_attrs(const xir::Graph& src_graph,
                                   SubgraphInfo& info) {
  // fix subgraph name same with someone supler_layer name
  if (info.children.empty()) {
    auto src_subgraph =
        src_graph.get_leaf_subgraph(src_graph.get_op(*info.ops.begin()));
    info.attrs = src_subgraph->get_attrs();
  } else {
    auto src_subgraph = src_graph.get_subgraph(info.name);
    info.attrs = src_subgraph->get_attrs();
    for (auto& child : info.children) {
      fill_in_subgraph_attrs(src_graph, child);
    }
  }
}
static void clone_subgraphs(xir::Graph& dst_graph,
                            const xir::Graph& src_graph) {
  denotate_subgraph_path(dst_graph, src_graph);

  for (auto& op : dst_graph.topological_sort()) {
    LOG(INFO) << "op name : " << op->get_name() << "  subgraph_path : "
              << op->get_attr<subgraph_path_t>("subgraph_path");
  }
  auto info = extract_subgraph_info(dst_graph);
  LOG(INFO) << "extract_subgraph_info : " << info;
  fill_in_subgraph_attrs(src_graph, info);

  create_subgraph_tree(dst_graph, info);
}

int main(int argc, char* argv[]) {
  auto model = argv[1];
  LOG(INFO) << "xmodel : " << model;
  auto src_graph = xir::Graph::deserialize(model);
  auto root = src_graph->get_root_subgraph();
  auto children = root->children_topological_sort();

  auto subgraph_idx = 0;
  for (auto c : children) {
    auto device = c->get_attr<std::string>("device");
    if ("DPU" == device) {
      LOG(INFO) << device << " " << c->get_name();
      auto dst_graph = clone_dpu_ops(*c);
      dst_graph->serialize("/tmp/" + std::to_string(subgraph_idx) + "b.xmodel");
      clone_subgraphs(*dst_graph, *src_graph);
      dst_graph->serialize("/tmp/" + std::to_string(subgraph_idx) + ".xmodel");
    }
    subgraph_idx = subgraph_idx + 1;
  }

  return 0;
}
