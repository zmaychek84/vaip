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

#include <core/graph/graph.h>

#include <fstream>
#include <iostream>
#include <streambuf>
#include <string>

#include "vaip/pattern/pattern.hpp"

using namespace std;
#include "onnx/checker.h"
#include "onnx/defs/parser.h"

using namespace ONNX_NAMESPACE;

class PatternTest : public ::testing::Test {
protected:
  PatternTest() { logger_ = DefaultLoggingManager().CreateLogger("GraphTest"); }

  std::unique_ptr<logging::Logger> logger_;
};

int main_test_wild_card() {
  const char* code = R"ONNX(
<
  ir_version: 7,
  opset_import: [ "" : 10 ]
>
agraph (float[N, 128] X, float[128, 10] W, float[10] B) => (float[N, 10] C)
{
    T = MatMul(X, W)
    S = Add(T, B)
    C = Softmax(S)
}
)ONNX";

  ModelProto model;
  OnnxParser::Parse(model, code);

  checker::check_model(model);
  cout << model.DebugString() << endl;
  onnxruntime::Graph::GetInitializedTensor(const std::string& tensor_name,
                                           const onnx::TensorProto*& value)
      vaip::PatternBuilder builder;
  auto pattern = builder.wildcard();
  for (auto& node : model.graph().node()) {
    auto match = pattern->match(node);
    cout << "node: " << node.name() << " "                //
         << "pattern: " << pattern->debug_string() << " " //
         << "match: " << match << " "                     //
         << endl;
  }
  return 0;
}
int main(int argc, char* argv[]) { return main_test_wild_card(); }
