/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

#include <cstdlib>
#include <string>
#include <vector>

std::vector<std::string> split_path(const char* env_name) {
  std::string path;
#ifdef _WIN32
#  pragma warning(push)
#  pragma warning(disable : 4996)
#endif
  auto env_value = getenv(env_name);
  path = env_value != nullptr ? env_value : "";
#ifdef _WIN32
#  pragma warning(pop)
#endif
  auto ret = std::vector<std::string>();
#ifdef _MSC_VER
  char sep = ';';
#else
  char sep = ':';
#endif
  std::string::size_type pos0 = 0u;
  for (auto pos = path.find(sep, pos0); pos != std::string::npos;
       pos = path.find(sep, pos + 1)) {
    ret.push_back(path.substr(path[pos0] == sep ? pos0 + 1 : pos0,
                              pos - pos0 - (path[pos0] == sep ? 1 : 0)));
    pos0 = pos;
  }
  if (pos0 == 0 && !path.empty()) {
    ret.push_back(path);
  } else if (pos0 != std::string::npos && path[pos0] == sep) {
    ret.push_back(path.substr(pos0 + 1));
  }
  return ret;
}
