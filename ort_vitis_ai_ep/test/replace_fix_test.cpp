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

#include "../src/imp/replace_fix.hpp"

#include <functional>

#include "../src/imp/provider/vitisai_compile_model.hpp"
#include "../src/imp/util.hpp"
#include "./test_main.hpp"
#include "core/graph/node_arg.h"
#include "vaip/pattern/pattern.hpp"
#include "vaip/type_denotation.hpp"
#include "vaip/xir_ops/xir_ops_defs.hpp"
#include "vitis/ai/profiling.hpp"

using namespace std;
using namespace onnxruntime;

class ReplaceFixTest : public WithLogger {
protected:
  ReplaceFixTest() : WithLogger() {}
};

NodeArg* to_node_arg(const NodeArg* n) { return const_cast<NodeArg*>(n); }
TEST_F(ReplaceFixTest, ReplaceFix0) {
  vaip::register_xir_ops();
  auto env_model_path = getenv("MODEL_PATH");
  auto model_path = std::string(
      "/workspace/aisw/onnx_models/quantized_resnet50/ResNet_int.onnx");
  if (env_model_path != nullptr) {
    model_path = env_model_path;
  }
  LOG(INFO) << "start loading  " << model_path << " ...";
  std::ifstream t(model_path, std::ifstream::binary);
  t.seekg(0, t.end);
  size_t length = t.tellg();
  t.seekg(0, t.beg);
  std::string buffer;
  buffer.resize(length);
  ASSERT_TRUE(t.read(&buffer[0], buffer.size()).good());
  t.close();
  LOG(INFO) << "start parsing  " << model_path << " ...";
  onnx::ModelProto model_proto;
  ASSERT_TRUE(ParseProtoFromBytes(&model_proto, &buffer[0], buffer.size()));
  LOG(INFO) << "start creating model.";
  auto model = load_model(std::move(model_proto));
  LOG(INFO) << "model is created." //
            << "\n\tproducer=" << model->ProducerName() << " @"
            << model->ProducerVersion()
            << "\n\tir_version=" << model->IrVersion() //
      ;

  auto& graph = model->MainGraph();

  for (auto i : graph.DomainToVersionMap()) {
    LOG(INFO) << "i.first " << i.first << " "   //
              << "i.second " << i.second << " " //
        ;
  }
  LOG(INFO) << "onnx::OpSchemaRegistry::Schema(key, version, domain); "
            << ONNX_NAMESPACE::OpSchemaRegistry::Schema("matmul", "com.xilinx");
  LOG(INFO) << "onnx::OpSchemaRegistry::Schema(key, version, domain); "
            << ONNX_NAMESPACE::OpSchemaRegistry::Schema("conv2d", 1,
                                                        "com.xilinx");
  auto all = onnx::OpSchemaRegistry::get_all_schemas();
  for (auto& r : all) {
    LOG(INFO) << r.Name() << " " << r.domain();
  }

  GraphViewer graph_viewer(graph);
  vaip::compile_onnx_model(graph_viewer, *logger_);
}
