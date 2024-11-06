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

#include "../src/imp/config.hpp"
#include "./test_main.hpp"
#include "core/graph/model.h"
#include "vaip/pattern/pattern.hpp"
#include "vaip/xir_ops/xir_ops_defs.hpp"
using namespace std;
using namespace onnxruntime;

class XirOpsTest : public WithLogger {
protected:
  XirOpsTest() : WithLogger() {}
};

#include "../src/imp/util.hpp"
#include "./xir_ops_test_samples.txt"
TEST_F(XirOpsTest, Fix) {
  vaip::register_xir_ops();
  std::shared_ptr<onnxruntime::Model> model = load_model(XIR_OPS_FIX());
  // auto& graph = model->MainGraph();
  // auto& node_refererence = graph.Nodes();
  // std::vector<const Node*> nodes((size_t)graph.NumberOfNodes(), nullptr);
  // std::transform(node_refererence.begin(), node_refererence.end(),
  //                nodes.begin(), [](const Node& n) { return &n; });

  // auto pattern_builder = vaip::PatternBuilder();
  // auto is_wildcard = pattern_builder.wildcard();
  // for (auto n : nodes) {
  //  LOGS(*logger_, INFO) << "pattern : " << is_wildcard->debug_string()
  //                       << " node : " << n->Domain() << ":" << n->OpType();
  //}
}

TEST_F(XirOpsTest, SuperLayer) {
  vaip::register_xir_ops();
  std::shared_ptr<onnxruntime::Model> model = load_model(XIR_OPS_SUPER_LAYER());
  LOGS(*logger_, INFO) << "Graph : "
                       << vaip::graph_as_string(model->MainGraph());
}

namespace vaip {
void show_all_op_defs();
void generate_all_op_defs();
} // namespace vaip

TEST_F(XirOpsTest, ShowAllOpDef) { vaip::show_all_op_defs(); }
TEST_F(XirOpsTest, GenerateXirOpDefs) { vaip::generate_all_op_defs(); }

// TEST_F(XirOpsTest, Config) {
//   auto config = vaip::Config::new_config();
//   CHECK(config != nullptr);
// }
