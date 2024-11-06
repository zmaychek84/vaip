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
#include <fstream>
#include <iostream>

#include <dd.hpp>

using namespace dd;

struct test {
  Graph adj_graph;
  property_map property;
  Graph result;
};

std::vector<test> tests =
    {/* Test 0*/
     {{{{0, {1, 2}},
        {1, {3, 4}},
        {2, {5, 6}},
        {3, {7}},
        {4, {7}},
        {5, {8}},
        {6, {8}},
        {7, {9}},
        {8, {9}},
        {9, {}}}},
      {{{0, "true"},
        {1, "true"},
        {2, "true"},
        {3, "true"},
        {4, "false"},
        {5, "true"},
        {6, "false"},
        {7, "true"},
        {8, "true"},
        {9, "true"}}},
      {{{0, {0, 1, 2, 3, 5}}, {1, {4}}, {2, {6}}, {3, {7, 8, 9}}}}},
     /* Test 1 */
     {{{{0, {1, 2, 3}}, {1, {4}}, {2, {4}}, {3, {4}}, {4, {}}}},
      {{{0, "true"}, {1, "true"}, {2, "true"}, {3, "false"}, {4, "true"}}},
      {{{0, {0, 1, 2}}, {1, {3}}, {2, {4}}}}},
     /* Test 2 */
     {{{{0, {1, 2, 3}}, {1, {3}}, {2, {3}}, {3, {}}}},
      {{{0, "AIE"}, {1, "AIE"}, {2, "CPU"}, {3, "AIE"}}},
      {{{0, {0, 1}}, {1, {2}}, {2, {3}}}}},
     /* Test 3 */
     {{{{0, {1, 2, 3}}, {1, {3}}, {2, {3}}, {3, {}}}},
      {{{0, "true"}, {1, "true"}, {2, "true"}, {3, "true"}}},
      {{{0, {0, 1, 2, 3}}}}},
     /* Test ps-attn */
     {
         {{{0, {1, 2, 3, 6}},
           {1, {4}},
           {2, {4}},
           {3, {4}},
           {4, {5}},
           {5, {6}},
           {6, {7}},
           {7, {8, 10}},
           {8, {9}},
           {9, {10}},
           {10, {}}}},
         {{{0, "true"},
           {1, "true"},
           {2, "true"},
           {3, "true"},
           {4, "false"},
           {5, "true"},
           {6, "true"},
           {7, "true"},
           {8, "true"},
           {9, "true"},
           {10, "true"}}},
         {{{0, {0, 1, 2, 3}}, {1, {4}}, {2, {5, 6, 7, 8, 9, 10}}}},
     },
     /* Test MatMulAddGelu */
     {{{{2, {6}},
        {0, {3}},
        {1, {3}},
        {3, {4}},
        {4, {5}},
        {5, {6}},
        {6, {7}},
        {7, {8}},
        {8, {9}},
        {9, {10}},
        {10, {}}}},
      {{{0, "CPU"},
        {1, "CPU"},
        {2, "CPU"},
        {3, "AIE"},
        {4, "CPU"},
        {5, "CPU"},
        {6, "AIE"},
        {7, "CPU"},
        {8, "CPU"},
        {9, "CPU"},
        {10, "CPU"}}},
      {{{0, {0}},
        {1, {1}},
        {2, {2}},
        {3, {3}},
        {4, {4, 5}},
        {6, {6}},
        {7, {7, 8, 9, 10}}}}}};

void test(Graph adj_graph) {
  auto CGraph = CompositeGraph(adj_graph);
  // std::cout << "parent_graph_[0] : " << CGraph.parent_graph_[0].empty() <<
  // std::endl;
  assert(CGraph.parent_graph_[0] == {});
  // std::cout << CGraph.input_nodes_.size() << " " << CGraph.input_nodes_[0] <<
  // std::endl;
  assert(CGraph.input_nodes_ == {0});
  // std::cout << CGraph.get_subgraph_label_of_node(1) << std::endl;
  assert(CGraph.get_subgraph_label_of_node(1) == 1);
  auto nodes = CGraph.get_next_nodes_of_node(0);
  // std::cout << "next_nodes of 0 : ";
  // for(auto ind : nodes)
  //     std::cout << ind << " ";
  // std::cout << std::endl;
  assert(CGraph.get_next_nodes_of_node(0) == {1, 2});
  // assert(CGraph.get_input_subgraphs() == {0});
  auto subgs = CGraph.get_next_subgraphs(0);
  // for(auto node : subgs)
  //     std::cout << node << " ";
  // std::cout << std::endl;
  assert(CGraph.get_next_subgraphs(0) == {1, 2});
  auto nod = CGraph.get_nodes_in_subgraph(1);
  // std::cout << "nodes in subg 1 : " << nod[0] << std::endl;
  assert(CGraph.get_nodes_in_subgraph(1) == {1});
  // std::cout << CGraph.is_dag() << std::endl;
  assert(CGraph.is_dag() == true);

  CGraph.fuse(0, 1);
  // std::cout << "After fusing 0 & 1 : {";
  print_label_map("After fusing 0 & 1 : {", CGraph.labels_);

  CGraph.fuse(0, 3);
  assert(CGraph.is_dag() == true);
  print_label_map("After fusing 0 & 3 : {", CGraph.labels_);
  // std::cout << "After fusing 0 & 3 : {";

  assert(CGraph.try_fuse(0, 7) == false);
  CGraph.fuse(0, 7);
  assert(CGraph.is_dag() == false);
  print_label_map("After fusing 0 & 7 : {", CGraph.labels_);

  std::cout << "TEST PASSED" << std::endl;
}
int main(int argc, char* argv[]) {
  test(tests[0].adj_graph);

  for (auto test : tests) {
    auto subgraphs = partition_graph(test.adj_graph, test.property);
    auto clusters = subgraph_labels_to_clusters(subgraphs);
    print_graph("Expected Result: ", test.result);
    print_graph("Result Obtained: ", clusters);
    /*std::cout << "Expected Result: ";
    for(auto res : test.result) {
        std::cout << res.first << ": [";
        for(auto ind : res.second)
            std::cout << " " << ind;
        std::cout << "], ";
    }
    std::cout << "\nSubgraphs: {";
    for(auto node : subgraphs) {
        std::cout << node.first << ": " << node.second << ",";
    }
    std::cout << std::endl;
    std::cout << "Result Obtained: ";
    for(auto res : clusters) {
        std::cout << res.first << ": [";
        for(auto ind : res.second)
            std::cout << " " << ind;
        std::cout << "], ";
    }*/
    std::cout << std::endl << " ################ " << std::endl;
  }
  std::ifstream adj_file(argv[1]);
  node_ind_t node1, node2;
  Graph adj_list;
  while (adj_file >> node1 >> node2) {
    auto it = (node2 == -1) ? adj_list.insert({node1, node_list()})
                            : adj_list.insert({node1, {node2}});
    if (it.second == false) {
      adj_list[node1].emplace_back(node2);
    }
  }
  print_graph("adj_list :", adj_list);
  std::string prop;
  std::ifstream prop_file(argv[2]);
  property_map node_labels;
  while (prop_file >> node1 >> prop) {
    auto it = node_labels.insert({node1, prop});
  }
  print_property_map("labels :", node_labels);

  auto subgraphs = partition_graph(adj_list, node_labels);
  auto clusters = subgraph_labels_to_clusters(subgraphs);
  print_graph("Result Obtained: ", clusters);
}