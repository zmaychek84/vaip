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

#include "vaip/rewrite_rule.hpp"

#include <glog/logging.h>

#include "./sample_models.hpp"
#include "./test_main.hpp"
#include "vaip/node.hpp"
#include "vaip/pattern/pattern.hpp"

using namespace std;
using namespace onnxruntime;
using namespace vaip;

class RewriteRuleTest : public WithLogger {
protected:
  RewriteRuleTest() : WithLogger() {}
};

class ReplaceSoftmaxWithRelu : public Rule {
public:
  ReplaceSoftmaxWithRelu();

private:
  virtual const Pattern* pattern() const override;
  virtual bool action(onnxruntime::Graph* graph,
                      binder_t& binder) const override;

private:
  std::shared_ptr<Pattern> input_;
  std::unique_ptr<Pattern> pattern_;
};

ReplaceSoftmaxWithRelu::ReplaceSoftmaxWithRelu() : Rule() {
  auto builder = PatternBuilder();
  input_ = builder.wildcard();
  pattern_ = builder.node2("Softmax", {input_});
}

const Pattern* ReplaceSoftmaxWithRelu::pattern() const {
  return pattern_.get();
}

static NodeArg* to_node_arg(const NodeArg* n) {
  return const_cast<NodeArg*>(n);
}

bool ReplaceSoftmaxWithRelu::action(onnxruntime::Graph* graph,
                                    binder_t& binder) const {
  // for (auto& b : binder) {
  //   LOG(INFO) << "b.first " << b.first << " "     //
  //             << "b.second " << *b.second << " "  //
  //       ;
  // }
  auto input = binder[input_->get_id()];
  CHECK(input.node != nullptr);
  auto softmax = binder[pattern_->get_id()];
  CHECK(softmax.node != nullptr);
  graph->AddNode("Reluhere", "Relu", "replace softmax",
                 {to_node_arg(&node_get_output_node_arg(*input.node))},
                 {to_node_arg(&node_get_output_node_arg(*softmax.node))});
  graph->RemoveEdge(input.node->Index(), softmax.node->Index(), 0, 0);
  graph->RemoveNode(softmax.node->Index());
  return false;
}

TEST_F(RewriteRuleTest, relace_softmax_with_relu) {
  std::shared_ptr<onnxruntime::Model> model = load_model(MATMUL_ADD_SOFTMAX());
  auto& graph = model->MainGraph();
  ReplaceSoftmaxWithRelu rule;
  rule.apply(&graph);
  LOG(INFO) << "graph= " << graph.ToGraphProto().DebugString();
  ASSERT_STATUS_OK(graph.Resolve());
}
