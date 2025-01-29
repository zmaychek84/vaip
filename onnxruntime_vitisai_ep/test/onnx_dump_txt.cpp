/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

#include <exception>
#include <filesystem>
#include <fstream>
#include <glog/logging.h>

#include "initialize_vaip.hpp"
#include <limits>

//
#include "vaip/vaip.hpp"
#include "vaip/vaip_ort.hpp"
//
#include "../include/onnxruntime_vitisai_ep/onnxruntime_vitisai_ep.hpp"
//

extern "C" {
#include "./getopt.h"
}

using namespace std;
/**
 * Prints the usage information for the command line tool.
 * This function should be called when the user passes the '-h' option or
 * provides invalid arguments.
 */
static void usage() {
  std::cout
      << "Usage: onnx_dump_txt -i <input_file> -o <output_file>\n"
      << "Options:\n"
      << "  -i <input_file>    Path to the input ONNX model file.\n"
      << "  -o <output_file>   Path where the output text file will be saved.\n"
      << "  -h                 Display this help message and exit.\n";
}
int main(int argc, char* argv[]) {
  try {
    auto opt_input_file = std::string();
    auto opt_output_txt_file = std::string();
    int opt = 0;
    while ((opt = getopt(argc, argv, "i:o:h")) != -1) {
      switch (opt) {
      case 'i': {
        opt_input_file = std::string(optarg);
        break;
      }
      case 'o': {
        opt_output_txt_file = std::string(optarg);
        break;
      }
      case 'h': {
        usage();
        return 0;
      }
      }
    }

    Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "onnx_dump_txt");
    initialize_vaip();
    auto model = vaip_cxx::Model::load(opt_input_file);
    auto graph = model->main_graph();
    graph.resolve();

    if (!opt_output_txt_file.empty()) {
      LOG(INFO) << "write output file to " << opt_output_txt_file;
      std::ofstream ofs(opt_output_txt_file, std::ios::out | std::ios::trunc);
      ofs << graph << std::endl;
    } else {
      LOG(WARNING) << "no -o is specified write to std::cout";
      std::cout << graph << std::endl;
    }
  } catch (const std::exception& e) {
    std::cerr << "exception occurs : " << e.what() << "\n";
  }
  return 0;
}

#include "./getopt.c"
