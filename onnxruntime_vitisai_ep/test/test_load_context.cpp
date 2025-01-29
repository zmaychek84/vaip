/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

// must include glog/logging before vaip.hpp
#include <exception>
#include <filesystem>
#include <glog/logging.h>
//
#include "vaip/vaip.hpp"

using namespace vaip_core;

int main(int argc, char* argv[]) {
  try {
    auto cache_dir = std::filesystem::path(argv[1]);
    auto context = load_context(cache_dir);
    LOG(INFO) << "context =" << (void*)context.get();
  } catch (const std::exception& e) {
    std::cerr << "exception occurs : " << e.what() << "\n";
  }
  return 0;
}
