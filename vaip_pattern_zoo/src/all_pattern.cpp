/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

#include "vaip/pattern_zoo.hpp"
#include <functional>
#include <glog/logging.h>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace vaip {
namespace pattern_zoo {
VAIP_DLL_SPEC std::vector<std::string> pattern_list() {
  auto patterns = std::vector<std::string>();
  VAIP_ORT_API(vaip_get_pattern_list)
  (reinterpret_cast<void*>(&patterns), [](void* env, void* data, size_t size) {
    auto ret = reinterpret_cast<std::vector<std::string>*>(env);
    ret->emplace_back((const char*)data, size);
  });
  return patterns;
}
VAIP_DLL_SPEC std::shared_ptr<vaip_core::Pattern>
get_pattern(const std::string& name) {

  std::shared_ptr<vaip_core::Pattern> pattern;
  auto ret = VAIP_ORT_API(vaip_get_pattern_as_binary)(
      name.data(), reinterpret_cast<void*>(&pattern),
      [](void* env, void* data, size_t size) {
        auto ret = reinterpret_cast<std::shared_ptr<vaip_core::Pattern>*>(env);
        *ret = vaip_core::PatternBuilder().create_from_binary((const char*)data,
                                                              size);
      });
  if (ret != 0) {
    LOG(WARNING) << "Dose not exist pattern : " << name;
    pattern = nullptr;
  }
  return pattern;
}
} // namespace pattern_zoo
} // namespace vaip
