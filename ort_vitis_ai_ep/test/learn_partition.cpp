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

#include "./sample_models.hpp"
#include "./test_main.hpp"
#include "core/graph/node_arg.h"
#include "vaip/pattern/pattern.hpp"

using namespace std;
using namespace onnxruntime;

class LearnPartition : public WithLogger {
protected:
  LearnPartition() : WithLogger() {}
};

NodeArg* to_node_arg(const NodeArg* n) { return const_cast<NodeArg*>(n); }
TEST_F(LearnPartition, LearnGraphApi) {
  std::shared_ptr<onnxruntime::Model> model = load_model(MATMUL_ADD_SOFTMAX());
  auto& graph = model->MainGraph();
  LOGS(*logger_, INFO) << "graph.Name() = " << graph.Name();
  LOGS(*logger_, INFO) << "graph.Description() = " << graph.Description();
  LOGS(*logger_, INFO) << "graph.StrictShapeTypeInference() "
                       << graph.StrictShapeTypeInference();
  auto& initializers = graph.GetAllInitializedTensors();
  LOGS(*logger_, INFO) << "graph.GetAllInitializedTensors(): begin";
  for (auto& i : initializers) {
    LOGS(*logger_, INFO) << "i "
                         << i.first; //  << " " << i.second // TensorProto;
  }
  LOGS(*logger_, INFO) << "graph.GetAllInitializedTensors(): end";
  auto& inputs = graph.GetInputs();
  LOGS(*logger_, INFO) << "graph.GetInputs(): begin";
  for (auto& i : inputs) {
    LOGS(*logger_, INFO) << "\ti=" << i->Name() << " exists=" << i->Exists()
                         << " type " << *i->Type() << *i;
  }
  LOGS(*logger_, INFO) << "graph.MaxNodeIndex(): " << graph.MaxNodeIndex();
  LOGS(*logger_, INFO) << "graph.NumberOfNodes(): " << graph.NumberOfNodes();
  LOGS(*logger_, INFO) << "graph.GraphResolveNeeded(): "
                       << graph.GraphResolveNeeded();
  auto& node_refererence = graph.Nodes();
  for (auto it = node_refererence.begin(); it != node_refererence.end(); ++it) {
    it->AddAttribute("origin", std::vector<std::string>{"hello", "world"});
  }
  std::vector<const Node*> nodes((size_t)graph.NumberOfNodes(), nullptr);
  std::transform(node_refererence.begin(), node_refererence.end(),
                 nodes.begin(), [](const Node& n) { return &n; });
  LOGS(*logger_, INFO) << "node list:";
  for (auto n : nodes) {
    LOGS(*logger_, INFO) << "\t" << n->Index() << ":" << *n;
  }
  auto add_node = nodes[1];
  graph.RemoveEdge(1, 2, 0, 0);
  graph.RemoveNode(add_node->Index());
  graph.AddNode("newAdd", "Add", "newly added op",
                {to_node_arg(&vaip::node_get_output_node_arg(*nodes[0])),
                 to_node_arg(graph.GetInputs()[2])},
                {to_node_arg(nodes[2]->InputDefs()[0])});
  LOGS(*logger_, INFO) << "node `add` is removed";
  LOGS(*logger_, INFO) << "graph.GraphResolveNeeded(): "
                       << graph.GraphResolveNeeded();
  ASSERT_STATUS_OK(graph.Resolve());
  LOGS(*logger_, INFO) << "graph= " << graph.ToGraphProto().DebugString();
  GraphViewer graph_viewer(graph);
  // Nodes must be sorted in Topological Order in the GraphProto per ONNX spec.
  auto node_indices = graph_viewer.GetNodesInTopologicalOrder();
  for (auto it = node_indices.rbegin(); it != node_indices.rend(); ++it) {
    auto node_idx = *it;
    auto n = graph_viewer.GetNode(node_idx);
    LOGS(*logger_, INFO) << "\t" << n->Index() << ":" << *n;
  }
}
