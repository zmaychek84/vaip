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
#include "dd.hpp"
#include <algorithm>
#include <iostream>

namespace dd {

void print_node_list(std::string str, node_list& nodes) {
  std::cout << str << "[";
  for (auto node : nodes)
    std::cout << node << ", ";
  std::cout << "]" << std::endl;
}
void print_property_map(std::string str, property_map& node_labels) {
  std::cout << str << "{";
  for (auto l : node_labels) {
    std::cout << l.first << ": " << l.second << ", ";
  }
  std::cout << "}" << std::endl;
}
void print_label_map(std::string str, label_map& labels) {
  std::cout << str << "{";
  for (auto l : labels) {
    std::cout << l.first << ": " << l.second << ", ";
  }
  std::cout << "}" << std::endl;
}
void print_graph(std::string str, Graph& g) {
  std::cout << str << "{";
  for (auto node : g) {
    std::cout << node.first << ": [";
    for (auto ind : node.second)
      std::cout << " " << ind;
    std::cout << "], ";
  }
  std::cout << "}" << std::endl;
}
node_list unique(node_list& in_nodes) {
  node_set temp_set(in_nodes.begin(), in_nodes.end());
  node_list out_nodes(temp_set.begin(), temp_set.end());
  return out_nodes;
}
node_list stable_unique(node_list& in_nodes) {
  std::map<node_ind_t, node_ind_t> temp_map;
  node_list out_nodes;
  for (auto node : in_nodes) {
    auto it = temp_map.insert({node, 0});
    if (it.second)
      out_nodes.emplace_back(node);
  }
  return out_nodes;
}
CompositeGraph::CompositeGraph(Graph adj_graph) {
  child_graph_ = adj_graph;
  // print_graph("Child_graph : ", child_graph_);
  child_graph_to_parent_graph(adj_graph);
  for (auto node : child_graph_) {
    labels_.insert(std::make_pair(node.first, node.first));
  }
  for (auto l : labels_) {
    clusters_.insert(std::make_pair(l.second, node_list()));
    clusters_[l.second].emplace_back(l.first);
  }
  get_input_nodes();
}

void CompositeGraph::child_graph_to_parent_graph(Graph adj_graph) {
  for (auto node : child_graph_) {
    parent_graph_.insert(std::make_pair(node.first, node_list()));
    for (auto child : node.second) {
      parent_graph_.insert(std::make_pair(child, node_list()));
      parent_graph_[child].emplace_back(node.first);
    }
  }
  // print_graph("Parent_graph : ", parent_graph_);
}
void CompositeGraph::get_input_nodes() {
  for (auto node : parent_graph_) {
    if (node.second.size() == 0)
      input_nodes_.emplace_back(node.first);
  }
}

node_ind_t CompositeGraph::get_subgraph_label_of_node(node_ind_t node) {
  return labels_[node];
}

node_ind_t CompositeGraph::get_subgraph_for_node(node_ind_t node) {
  auto label = get_subgraph_label_of_node(node);
  return label;
}

node_list CompositeGraph::get_next_nodes_of_node(node_ind_t node) {
  return child_graph_[node];
}

node_list CompositeGraph::get_input_subgraphs() {
  node_list subgs;
  for (auto node : input_nodes_)
    subgs.emplace_back(get_subgraph_label_of_node(node));
  auto unique_subgs = stable_unique(subgs);
  return unique_subgs;
}

node_list CompositeGraph::get_nodes_in_subgraph(node_ind_t subg) {
  auto nodes = clusters_[subg];
  return stable_unique(nodes);
}

node_list CompositeGraph::get_next_subgraphs(node_ind_t subg) {
  auto nodes = get_nodes_in_subgraph(subg);
  node_list next_nodes;
  for (auto node : nodes) {
    auto child_nodes = child_graph_[node];
    for (auto child_node : child_nodes) {
      if (std::find(nodes.begin(), nodes.end(), child_node) == nodes.end())
        next_nodes.emplace_back(child_node);
    }
  }
  node_list next_subgs;
  for (auto node : next_nodes) {
    next_subgs.emplace_back(get_subgraph_label_of_node(node));
  }
  next_subgs = stable_unique(next_subgs);
  return next_subgs;
}

bool CompositeGraph::is_cycle_detected(node_ind_t subg, node_set& subgs_visited,
                                       node_set& subgs_in_stack) {
  subgs_visited.insert(subg);
  subgs_in_stack.insert(subg);
  auto next_subgs = get_next_subgraphs(subg);
  for (auto next_subg : next_subgs) {
    if (subgs_in_stack.find(next_subg) != subgs_in_stack.end())
      return true;
    if (subgs_visited.find(next_subg) == subgs_visited.end())
      if (is_cycle_detected(next_subg, subgs_visited, subgs_in_stack))
        return true;
  }
  subgs_in_stack.erase(subg);
  return false;
}

bool CompositeGraph::is_dag() {
  node_set subgs_visited;
  node_set subgs_in_stack;

  auto input_subgs = get_input_subgraphs();
  for (auto subg : input_subgs) {
    if (is_cycle_detected(subg, subgs_visited, subgs_in_stack))
      return false;
  }
  return true;
}

void CompositeGraph::fuse(node_ind_t subgraph1, node_ind_t subgraph2) {
  if (subgraph1 < subgraph2) {
    auto subgraph2_nodes = clusters_[subgraph2];
    clusters_[subgraph1].insert(clusters_[subgraph1].end(),
                                subgraph2_nodes.begin(), subgraph2_nodes.end());
    for (auto node : subgraph2_nodes) {
      labels_[node] = subgraph1;
    }
  } else if (subgraph2 < subgraph1) {
    auto subgraph1_nodes = clusters_[subgraph1];
    clusters_[subgraph2].insert(clusters_[subgraph2].end(),
                                subgraph1_nodes.begin(), subgraph1_nodes.end());
    for (auto node : subgraph1_nodes) {
      labels_[node] = subgraph2;
    }
  }
}

void CompositeGraph::fuse_all(node_list subgraphs) {}

bool CompositeGraph::try_fuse(node_ind_t subgraph1, node_ind_t subgraph2) {
  label_map old_labels(labels_);
  auto subg1_nodes = clusters_[subgraph1];
  auto subg2_nodes = clusters_[subgraph2];

  fuse(subgraph1, subgraph2);
  if (is_dag()) {
    return true;
  } else {
    labels_ = old_labels;
    clusters_[subgraph1] = subg1_nodes;
    clusters_[subgraph2] = subg2_nodes;
    return false;
  }
}

node_list CompositeGraph::topsort() {
  std::vector<node_ind_t> s; //(input_nodes_.begin(), input_nodes_.end());
  node_list result;

  for (auto node : input_nodes_)
    s.push_back(node);
  node_set visited_nodes(input_nodes_.begin(), input_nodes_.end());
  while (!s.empty()) {
    auto node = s.back();
    auto child_nodes = get_next_nodes_of_node(node);
    for (auto child_node : child_nodes) {
      if (visited_nodes.find(child_node) == visited_nodes.end()) {
        s.push_back(child_node);
        visited_nodes.insert(child_node);
      }
    }
    if (s.back() == node) {
      result.emplace_back(node);
      s.pop_back();
    }
  }
  node_list res(result.rbegin(), result.rend());
  return res;
}

label_map partition_graph(Graph adj_graph, property_map property,
                          std::string optimization_flag,
                          const std::vector<size_t>& sorted_node_ids) {
  std::vector<std::string> optim_flags = {"L0", "L1", "L2"};
  assert(std::find(optim_flags.begin(), optim_flags.end(), optimization_flag) !=
         optim_flags.end());

  auto graph = CompositeGraph(adj_graph);

  if (optimization_flag == "L0")
    return graph.labels_;

  node_list nodes;
  bool sorted = !sorted_node_ids.empty();
  if (sorted) {
    for (auto& item : sorted_node_ids)
      nodes.push_back((int32_t)item);
  } else
    nodes = graph.topsort();

  for (auto i = 0; i < ((int32_t)nodes.size() - 1); i++) {
    auto node = nodes[i];
    auto next_node = nodes[i + 1];
    auto node_property = property[node];
    if (node_property != "CPU" && (node_property == property[next_node])) {
      auto subg_i = graph.get_subgraph_for_node(node);
      auto subg_in = graph.get_subgraph_for_node(next_node);
      graph.fuse(subg_i, subg_in);
    }
  }
  std::queue<node_ind_t> q;
  node_set visited_nodes;
  // L1 Partition
  for (auto node : graph.input_nodes_) {
    q.push(node);
    visited_nodes.insert(node);
  }

  while (!q.empty()) {
    auto node = q.front();
    q.pop();
    auto try_fuse = property[node] != "CPU" ? true : false;
    auto sg = graph.get_subgraph_for_node(node);
    auto next_nodes = graph.get_next_nodes_of_node(node);
    for (auto next_node : next_nodes) {
      if (try_fuse && (property[node] == property[next_node])) {
        auto next_sg = graph.get_subgraph_for_node(next_node);
        // auto status = graph.try_fuse(sg, next_sg);
        graph.try_fuse(sg, next_sg);
      }
      if (visited_nodes.find(next_node) == visited_nodes.end()) {
        q.push(next_node);
        visited_nodes.insert(next_node);
      }
    }
  }
  if (optimization_flag == "L1")
    return graph.labels_;

  if (optimization_flag == "L2")
    throw std::runtime_error(
        "L2 optimization is not implmented in Graph Partitioner");

  return graph.labels_;
}

Graph subgraph_labels_to_clusters(label_map subgraphs) {
  std::set<node_ind_t> cluster_inds;
  for (auto node : subgraphs)
    cluster_inds.insert(node.second);
  Graph clusters;
  for (auto c : cluster_inds)
    clusters.insert(std::make_pair(c, node_list()));
  for (auto node : subgraphs)
    clusters[node.second].emplace_back(node.first);
  return clusters;
}

} // namespace dd