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

#include "core/framework/execution_provider.h"

#include "../src/imp/util.hpp"
#include "./test_main.hpp"
#include "core/framework/compute_capability.h"
#include "core/graph/node_arg.h"
#include "core/providers/vitisai/vitisai_provider_factory.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "vaip/export_to_xir.hpp"
#include "vaip/pattern/pattern.hpp"
#include "vaip/type_denotation.hpp"
#include "vaip/xir_ops/xir_ops_defs.hpp"

using namespace std;
using namespace onnxruntime;

class ExecutionProviderTest : public WithLogger {
protected:
  ExecutionProviderTest() : WithLogger() {}
};

NodeArg* to_node_arg(const NodeArg* n) { return const_cast<NodeArg*>(n); }
TEST_F(ExecutionProviderTest, Partition) {
  vaip::register_xir_ops();
  auto model = load_model_from_env_model_path();
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
  const auto& api = Ort::GetApi();
  OrtSessionOptions* session_options = nullptr;
  CheckStatus(api.CreateSessionOptions(&session_options));
  const char* export_runtime_module = "abc";
  const char* load_runtime_module = "abc";
  CheckStatus(OrtSessionOptionsAppendExecutionProvider_VITISAI(
      session_options, "VITISAI", 0 /*device id*/, export_runtime_module,
      load_runtime_module));
  auto ep = session_options->provider_factories.back()->CreateProvider();
  GraphViewer graph_viewer(model->MainGraph());
  auto sub_graph = ep->GetCapability(graph_viewer, {});
  LOG(INFO) << "sub_graph.size() " << sub_graph.size() << " " //
      ;
  api.ReleaseSessionOptions(session_options);
  return;
}
