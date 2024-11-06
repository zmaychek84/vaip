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

#include <assert.h>
#include <core/providers/vitisai/vitisai_provider_factory.h>
#include <core/session/experimental_onnxruntime_cxx_api.h>
#include <glog/logging.h>

#include <algorithm> // std::generate
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#if _WIN32
#  include <codecvt>
#  include <locale>
using convert_t = std::codecvt_utf8<wchar_t>;
std::wstring_convert<convert_t, wchar_t> strconverter;
#endif

#define CHECK_STATUS_OK(expr)                                                  \
  do {                                                                         \
    Status _tmp_status = (expr);                                               \
    CHECK(_tmp_status.IsOK()) << _tmp_status;                                  \
  } while (0)

static void CheckStatus(OrtStatus* status) {
  if (status != NULL) {
    const char* msg = Ort::GetApi().GetErrorMessage(status);
    fprintf(stderr, "%s\n", msg);
    Ort::GetApi().ReleaseStatus(status);
    exit(1);
  }
}

// pretty prints a shape dimension vector
static std::string print_shape(const std::vector<int64_t>& v) {
  std::stringstream ss("");
  for (size_t i = 0; i < v.size() - 1; i++)
    ss << v[i] << "x";
  ss << v[v.size() - 1];
  return ss.str();
}

static int calculate_product(const std::vector<int64_t>& v) {
  int total = 1;
  for (auto& i : v)
    total *= (int)i;
  return total;
}

std::string replace(const std::string& s) {
  const std::string pat = ".";
  std::ostringstream str;
  for (auto c : s) {
    if (pat.find(c) != std::string::npos) {
      str << "_";
    } else {
      str << c;
    }
  }
  return str.str();
}
using namespace std;
int main(int argc, char* argv[]) {
  for (auto i = 0; i < argc; ++i) {
    std::cout << "arg[" << i << "]=" << argv[i] << std::endl;
  }
  if (argc < 2) {
    std::cerr << "usage :" << argv[0] << " <model_name>"
              << " [<input_file>] [<input_file> ...]" << std::endl;
    abort();
  }

#if _WIN32
  auto model_name = strconverter.from_bytes(std::string(argv[1]));
#else
  auto model_name = std::string(argv[1]);
#endif
  std::vector<std::string> g_input_files;
  if (argc >= 2) {
    for (auto i = 2; i < argc; i++) {
      g_input_files.push_back(std::string(argv[i]));
    }
  }

  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test-onnx-runner");
  // auto model = load_model(std::string(argv[1]), *logger);
  auto session_options = Ort::SessionOptions();
  const char* export_runtime_module = "abc";
  const char* load_runtime_module = "abc";
  CheckStatus(OrtSessionOptionsAppendExecutionProvider_VITISAI(
      session_options, "VITISAI", 0 /*device id*/, export_runtime_module,
      load_runtime_module));
  auto session = Ort::Experimental::Session(env, model_name, session_options);

  // print name/shape of inputs
  std::vector<std::string> input_names = session.GetInputNames();
  std::vector<std::vector<int64_t>> input_shapes = session.GetInputShapes();
  cout << "Input Node Name/Shape (" << input_names.size() << "):" << endl;
  for (size_t i = 0; i < input_names.size(); i++) {
    cout << "\t" << input_names[i] << " : " << print_shape(input_shapes[i])
         << endl;
  }

  // print name/shape of outputs
  std::vector<std::string> output_names = session.GetOutputNames();
  std::vector<std::vector<int64_t>> output_shapes = session.GetOutputShapes();
  cout << "Output Node Name/Shape (" << output_names.size() << "):" << endl;
  for (size_t i = 0; i < output_names.size(); i++) {
    cout << "\t" << output_names[i] << " : " << print_shape(output_shapes[i])
         << endl;
  }

  // Create a single Ort tensor from input_file or random numbers
  std::vector<Ort::Value> input_tensors;
  input_tensors.reserve(input_names.size());
  auto input_tensor_values =
      std::vector<std::vector<float>>(input_names.size());
  for (auto i = 0u; i < input_names.size(); i++) {
    auto input_shape = input_shapes[i];
    int total_number_elements = calculate_product(input_shape);
    input_tensor_values[i].resize(total_number_elements);
    if (!g_input_files.empty() && g_input_files.size() == input_names.size()) {
      auto batch = input_shape[0];
      auto size = total_number_elements / batch * sizeof(float);
      for (int index = 0; index < batch; ++index) {
        CHECK(
            std::ifstream(g_input_files[i])
                .read((char*)input_tensor_values[i].data() + size * index, size)
                .good())
            << "fail to read! filename=" << g_input_files[i];
      }
    } else {
      std::generate(input_tensor_values[i].begin(),
                    input_tensor_values[i].end(), [&] {
                      return ((float)(rand() % 255)) / 255.0f;
                    }); // generate random numbers in the range float[0, 1)
    }

    for (auto k = 0; k < 64; k++) {
      if (k % 16 == 0)
        std::cout << "\n";
      std::cout << input_tensor_values[i][k] << "  ";
    }

    input_tensors.push_back(Ort::Experimental::Value::CreateTensor<float>(
        input_tensor_values[i].data(), input_tensor_values[i].size(),
        input_shape));

    // double-check the dimensions of the input tensor
    auto input_tensor_shape =
        input_tensors[i].GetTensorTypeAndShapeInfo().GetShape();
    LOG(INFO) << "input_tensor_shape[" << i
              << "]: " << print_shape(input_tensor_shape);
    assert(input_tensors[i].IsTensor() && input_tensor_shape == input_shape);
    if (false) {
      auto input_tensor_ptr = input_tensors[i].GetTensorData<float>();
      auto batch = input_tensor_shape[0];
      auto total_number_elements =
          input_tensors[i].GetTensorTypeAndShapeInfo().GetElementCount();
      auto size = total_number_elements / batch * sizeof(float);
      for (int index = 0; index < batch; ++index) {
        auto filename = "/tmp/onnx_input_" + std::to_string(i) + "_" +
                        replace(input_names[i]) + "_batch_" +
                        std::to_string(index) + "_float.bin";
        auto mode =
            std::ios_base::out | std::ios_base::binary | std::ios_base::trunc;
        CHECK(std::ofstream(filename, mode)
                  .write((char*)input_tensor_ptr + size * index, size)
                  .good())
            << " faild to write to " << filename;
      }
    }
  }

  // pass data through model
  cout << "Running model...";
  try {
    auto output_tensors = session.Run(session.GetInputNames(), input_tensors,
                                      session.GetOutputNames());
    cout << "done" << endl;

    // double-check the dimensions of the output tensors
    // NOTE: the number of output tensors is equal to the number of output nodes
    // specifed in the Run() call
    assert(output_tensors.size() == session.GetOutputNames().size());
    for (auto i = 0u; i < output_tensors.size(); ++i) {
      CHECK(output_tensors[i].IsTensor()) << session.GetOutputNames()[i];
      auto output_tensor_shape =
          output_tensors[i].GetTensorTypeAndShapeInfo().GetShape();
      LOG(INFO) << "output_tensor_shape[" << i
                << "]: " << print_shape(output_tensor_shape);
      if (false) {
        auto output_tensor_ptr =
            output_tensors[i].GetTensorMutableData<float>();
        auto batch = output_tensor_shape[0];
        auto total_number_elements =
            output_tensors[i].GetTensorTypeAndShapeInfo().GetElementCount();
        auto size = total_number_elements / batch * sizeof(float);
        for (int index = 0; index < batch; ++index) {
          auto filename = "/tmp/onnx_output_" + std::to_string(i) + "_" +
                          replace(output_names[i]) + "_batch_" +
                          std::to_string(index) + "_float.bin";
          auto mode =
              std::ios_base::out | std::ios_base::binary | std::ios_base::trunc;
          CHECK(std::ofstream(filename, mode)
                    .write((char*)output_tensor_ptr + size * index, size)
                    .good())
              << " faild to write to " << filename;
        }
      }
    }
  } catch (const Ort::Exception& exception) {
    cout << "ERROR running model inference: " << exception.what() << endl;
    exit(-1);
  }
  return 0;
}
