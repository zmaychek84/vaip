/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 **/

#include <exception>
#include <filesystem> // Forfilesystem::path
#include <iostream>
#ifdef __GNUC__
#  pragma GCC diagnostic ignored "-Wpedantic"
#  pragma GCC diagnostic ignored "-Wconversion"
#  pragma GCC diagnostic ignored "-Wsign-compare"
#  pragma GCC diagnostic ignored "-Wunused-variable"
#  pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#endif
#include <onnxruntime_cxx_api.h>
#include <string>
#include <unordered_map>

using namespace std;

void usage() {
  cout << "usage: generate_test_cases <onnx model> <json_config>" << endl;
}
int main(int argc, char* argv[]) {
  if (argc < 2) {
    usage();
  }
#if _WIN32
  auto model_path = filesystem::path(argv[1]).wstring();
#else
  auto model_path = filesystem::path(argv[1]).u8string();
#endif
  try {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "generate_test_cases");
    auto session_options = Ort::SessionOptions();
    string config_file(argv[2]);
    string config_key{"config_file"};
    unordered_map<string, string> options{{config_key, config_file}};
    session_options.AppendExecutionProvider_VitisAI(options);
    Ort::Session(env, model_path.data(), session_options);
  } catch (const std::exception& e) {
    std::cerr << "exception occurs : " << e.what() << "\n";
  }
  return 0;
}
