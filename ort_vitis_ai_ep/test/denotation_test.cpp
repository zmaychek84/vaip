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
#include "vaip/type_denotation.hpp"
using namespace std;
using namespace onnxruntime;

class DenotationTest : public WithLogger {
protected:
  DenotationTest() : WithLogger() {}
};
NodeArg* to_node_arg(const NodeArg* n) { return const_cast<NodeArg*>(n); }
TEST_F(DenotationTest, DonationTest1) {
  auto env_model_path = getenv("MODEL_PATH");
  auto model_path = std::string(
      "/workspace/aisw/onnx/models/pytorch/quantized_resnet50/ResNet_int.onnx");
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
  vaip::fill_in_onnx_standard_meta_props(*model);
  vaip::denotation_pass(model->MainGraph());
  // ORT_THROW_IF_ERROR(onnxruntime::Model::SaveWithExternalInitializers(
  //     *model, "/tmp/a.onnx", "/tmp/a.data", 128));
}
