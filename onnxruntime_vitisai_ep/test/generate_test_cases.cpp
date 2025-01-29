/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

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
