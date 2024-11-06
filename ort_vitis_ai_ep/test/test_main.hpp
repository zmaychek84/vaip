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

#pragma once
#include <glog/logging.h>

#include <fstream>
#include <iostream>
#include <streambuf>
#include <string>

#include "core/graph/model.h"
#include "core/session/environment.h"
#include "core/session/onnxruntime_c_api.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/ort_env.h"
#include "onnx/checker.h"
#include "onnx/defs/parser.h"
#include "test/util/include/asserts.h"
#include "gtest/gtest.h"
using Status = onnxruntime::Status; // for ASSERT_STATUS_OK
namespace {

const OrtApi* g_ort = nullptr;
std::unique_ptr<Ort::Env> ort_env;

void CheckStatus(OrtStatus* status) {
  if (status != NULL) {
    const char* msg = g_ort->GetErrorMessage(status);
    fprintf(stderr, "%s\n", msg);
    g_ort->ReleaseStatus(status);
    exit(1);
  }
}

static void ortenv_setup() {
  g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  OrtThreadingOptions* tpo;
  CheckStatus(g_ort->CreateThreadingOptions(&tpo));
  ort_env.reset(new Ort::Env(tpo, ORT_LOGGING_LEVEL_VERBOSE, "Default"));
}

static ::onnxruntime::logging::LoggingManager& DefaultLoggingManager() {
  return *((OrtEnv*)*ort_env.get())->GetEnvironment().GetLoggingManager();
}

struct WithLogger : public ::testing::Test {
  WithLogger() : logger_{DefaultLoggingManager().CreateLogger("VAIP")} {}
  std::shared_ptr<onnxruntime::Model>
  load_model(onnx::ModelProto&& model_proto) {
    std::shared_ptr<onnxruntime::Model> model;
    auto status = onnxruntime::Model::Load(std::move(model_proto), model,
                                           nullptr, *logger_);
    if (!status.IsOK()) {
      LOGS(*logger_, FATAL) << status;
    }
    return model;
  }

  std::shared_ptr<onnxruntime::Model> load_model(const char* code) {
    return load_model(str_to_model_proto(code));
  }

  std::shared_ptr<onnxruntime::Model> load_model_from_env_model_path() { //
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
    CHECK(t.read(&buffer[0], buffer.size()).good());
    t.close();
    LOG(INFO) << "start parsing  " << model_path << " ...";
    onnx::ModelProto model_proto;
    CHECK(ParseProtoFromBytes(&model_proto, &buffer[0], buffer.size()));
    LOG(INFO) << "start creating model.";
    auto model = load_model(std::move(model_proto));
    LOG(INFO) << "model is created." //
              << "\n\tproducer=" << model->ProducerName() << " @"
              << model->ProducerVersion()
              << "\n\tir_version=" << model->IrVersion() //
        ;
    return model;
  }

  static onnx::ModelProto str_to_model_proto(const char* code) {
    onnx::ModelProto model_proto;
    onnx::OnnxParser::Parse(model_proto, code);

    onnx::checker::check_model(model_proto);
    return model_proto;
  }

  std::unique_ptr<onnxruntime::logging::Logger> logger_;
};
} // namespace

int main(int argc, char* argv[]) {
  testing::InitGoogleTest(&argc, argv);
  ortenv_setup();
  auto ret = RUN_ALL_TESTS();
  ort_env.reset(nullptr);
  return ret;
}
