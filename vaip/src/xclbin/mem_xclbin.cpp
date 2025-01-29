/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

#include <glog/logging.h>
#include <iostream>
#include <vaip/mem_xclbin.hpp>
#include <vaip/vaip.hpp>
namespace vaip_core {

std::vector<char> get_mem_xclbin(const std::string& filename) {
  std::vector<char> mem_xclbin;
  auto ret = VAIP_ORT_API(vaip_get_mem_xclbin)(
      filename.data(), reinterpret_cast<void*>(&mem_xclbin),
      [](void* env, void* data, size_t size) {
        auto* ret = static_cast<std::vector<char>*>(env);
        std::swap(*ret, std::vector<char>(static_cast<char*>(data),
                                          static_cast<char*>(data) + size));
      });
  return mem_xclbin;
}

bool has_mem_xclbin(const std::string& filename) {
  return VAIP_ORT_API(vaip_has_mem_xclbin)(filename.c_str());
}

} // namespace vaip_core
