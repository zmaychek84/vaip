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
#pragma once

#include <cassert>
#include <map>
#include <numeric>
#include <queue>
#include <set>
#include <stack>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace dd {

using node_ind_t = int32_t;
using node_list = std::vector<node_ind_t>;
using node_set = std::set<node_ind_t>;
using Graph = std::map<node_ind_t, node_list>;
using label_map = std::map<node_ind_t, node_ind_t>;
using property_map = std::map<node_ind_t, std::string>;

class CompositeGraph {
public:
  Graph child_graph_;
  Graph parent_graph_;
  label_map labels_;
  Graph clusters_;

  std::vector<node_ind_t> input_nodes_;

  CompositeGraph(Graph adj_graph);
  void get_input_nodes();

  node_ind_t get_subgraph_label_of_node(node_ind_t node);
  node_ind_t get_subgraph_for_node(node_ind_t node);
  node_list get_next_nodes_of_node(node_ind_t node);
  node_list get_input_subgraphs();
  node_list get_nodes_in_subgraph(node_ind_t subg);
  node_list get_next_subgraphs(node_ind_t subg);
  bool is_cycle_detected(node_ind_t subg, node_set& subgs_visited,
                         node_set& subgs_in_stack);
  bool is_dag();
  void fuse(node_ind_t subgraph1, node_ind_t subgraph2);
  void fuse_all(node_list subgraphs);
  bool try_fuse(node_ind_t subgraph1, node_ind_t subgraph2);
  node_list topsort();

  void child_graph_to_parent_graph(Graph adj_graph);
};

void print_node_list(std::string str, node_list& nodes);
void print_property_map(std::string str, property_map& labels);
void print_label_map(std::string str, label_map& labels);
void print_graph(std::string str, Graph& g);

label_map partition_graph(Graph adj_graph, property_map property,
                          std::string optimization_flag,
                          const std::vector<size_t>& sorted_nodes);
Graph subgraph_labels_to_clusters(label_map subgraphs);

} // namespace dd