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

#include "vaip/pattern/pattern.hpp"

#include "./sample_models.hpp"
#include "./test_main.hpp"

using namespace std;
using namespace onnxruntime;

class PatternTest : public WithLogger {
protected:
  PatternTest() : WithLogger() {}
};

TEST_F(PatternTest, LearnGraphApi) {
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

  ASSERT_STATUS_OK(graph.Resolve());
}

TEST_F(PatternTest, Wildcard0) {
  std::shared_ptr<onnxruntime::Model> model = load_model(MATMUL_ADD_SOFTMAX());
  auto& graph = model->MainGraph();
  auto& node_refererence = graph.Nodes();
  std::vector<const Node*> nodes((size_t)graph.NumberOfNodes(), nullptr);
  std::transform(node_refererence.begin(), node_refererence.end(),
                 nodes.begin(), [](const Node& n) { return &n; });

  auto pattern_builder = vaip::PatternBuilder();
  auto is_wildcard = pattern_builder.wildcard();
  for (auto n : nodes) {
    LOGS(*logger_, INFO) << "pattern : " << is_wildcard->debug_string()
                         << " node : " << n->Domain() << ":" << n->OpType()
        // << " match : " << is_wildcard->match(*n)
        ;
  }
}

TEST_F(PatternTest, Node) {
  std::shared_ptr<onnxruntime::Model> model = load_model(MATMUL_ADD_SOFTMAX());
  auto& graph = model->MainGraph();
  auto& node_refererence = graph.Nodes();
  std::vector<const Node*> nodes((size_t)graph.NumberOfNodes(), nullptr);
  std::transform(node_refererence.begin(), node_refererence.end(),
                 nodes.begin(), [](const Node& n) { return &n; });

  auto pattern_builder = vaip::PatternBuilder();

  // pattern_node test
  auto inputs_pattern = std::vector<std::shared_ptr<vaip::Pattern>>();
  inputs_pattern.push_back(pattern_builder.wildcard());
  auto is_softmax_op =
      pattern_builder.node2("Softmax", std::move(inputs_pattern));
  for (auto n : nodes) {
    LOGS(*logger_, INFO) << "pattern : " << is_softmax_op->debug_string()
                         << " node : " << n->Domain() << ":" << n->OpType()
                         << " match : "
                         << (is_softmax_op->match(graph, *n) != nullptr);
  }

  // pattern_or test
  auto or_args = std::vector<std::shared_ptr<vaip::Pattern>>();
  or_args.push_back(std::move(is_softmax_op));

  auto add_pattern = std::vector<std::shared_ptr<vaip::Pattern>>();
  add_pattern.push_back(pattern_builder.wildcard());
  or_args.push_back(pattern_builder.node2("Add", std::move(add_pattern)));
  auto softmax_or_add = pattern_builder.Or(std::move(or_args));
  LOGS(*logger_, INFO) << softmax_or_add->debug_string();

  for (auto n : nodes) {
    LOGS(*logger_, INFO) << "pattern : " << softmax_or_add->debug_string()
                         << " node : " << n->Domain() << ":" << n->OpType()
                         << " match : "
                         << (softmax_or_add->match(graph, *n) != nullptr);
  }

  // pattern where

  std::function<bool(const vaip::NodeInput&)> test =
      [](const vaip::NodeInput& node_input) {
        if (node_input.node == nullptr) {
          return false;
        }
        auto pb = vaip::PatternBuilder();
        auto add_pattern = std::vector<std::shared_ptr<vaip::Pattern>>();
        add_pattern.push_back(pb.wildcard());
        auto is_add = pb.node2("Add", std::move(add_pattern));
        // todo
        return true;
      };

  auto in_pattern = std::vector<std::shared_ptr<vaip::Pattern>>();
  in_pattern.push_back(pattern_builder.wildcard());
  auto is_softmax_node =
      pattern_builder.node2("Softmax", std::move(in_pattern));
  auto softmax_where_add =
      pattern_builder.where(std::move(is_softmax_node), test);

  for (auto n : nodes) {
    LOGS(*logger_, INFO) << "pattern : " << softmax_where_add->debug_string()
                         << " node : " << n->Domain() << ":" << n->OpType()
                         << " match : "
                         << (softmax_where_add->match(graph, *n) != nullptr);
  }
}

TEST_F(PatternTest, Chain) {
  auto b = vaip::PatternBuilder();
  std::vector<std::shared_ptr<vaip::Pattern>> p1 = b.chain2(
      {"r_first", {"opt_first", true}, "r_second", {"opt_second", true}},
      {b.wildcard()});
  auto [first, opt_first, second, opt_second, pat_or] =
      std::make_tuple(p1[0], p1[1], p1[2], p1[3], p1[4]);
  LOG(INFO) << "first->get_id() " << first->get_id() << " "           //
            << "opt_first->get_id() " << opt_first->get_id() << " "   //
            << "second->get_id() " << second->get_id() << " "         //
            << "opt_second->get_id() " << opt_second->get_id() << " " //
            << "pat_or->get_id() " << pat_or->get_id() << " "         //
      ;
  CHECK(!p1.empty());
}
