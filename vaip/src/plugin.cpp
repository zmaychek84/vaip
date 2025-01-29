/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

#include "vaip/plugin.hpp"

#include <glog/logging.h>

#include <iostream>

#include "vitis/ai/env_config.hpp"
DEF_ENV_PARAM(DEBUG_VART_PLUGIN, "0")
#include <sstream>
#include <string>
#include <unordered_map>
namespace vaip_core {

#if _WIN32
// clang-format off
#  include <windows.h>
// clang-format on
#  include <libloaderapi.h>

static std::unordered_map<std::string, std::unordered_map<std::string, void*>>&
get_store() {
  static std::unordered_map<std::string, std::unordered_map<std::string, void*>>
      store_;
  return store_;
}

static std::wstring s2ws(const std::string& s) {
  int len;
  int slength = (int)s.length() + 1;
  len = MultiByteToWideChar(CP_ACP, 0, s.c_str(), slength, 0, 0);
  wchar_t* buf = new wchar_t[len];
  MultiByteToWideChar(CP_ACP, 0, s.c_str(), slength, buf, len);
  std::wstring r(buf);
  delete[] buf;
  return r;
}

vitis::ai::plugin_t open_plugin(const std::string& name,
                                vitis::ai::scope_t scope) {
  static_assert(sizeof(vitis::ai::plugin_t) == sizeof(HMODULE));
  return LoadLibraryW(s2ws(name).c_str());
}

void* plugin_sym(vitis::ai::plugin_t plugin, const std::string& name) {
  return GetProcAddress(static_cast<HMODULE>(plugin), name.c_str());
}

void close_plugin(vitis::ai::plugin_t plugin) {
  FreeLibrary(static_cast<HMODULE>(plugin));
}
#else
#  include <dlfcn.h>
vitis::ai::plugin_t open_plugin(const std::string& name,
                                vitis::ai::scope_t scope) {
  auto flag_public = (RTLD_LAZY | RTLD_GLOBAL);
  auto flag_private = (RTLD_LAZY | RTLD_LOCAL);
  return dlopen(name.c_str(), scope == vitis::ai::scope_t::PUBLIC
                                  ? flag_public
                                  : flag_private);
}
void* plugin_sym(vitis::ai::plugin_t plugin, const std::string& name) {
  dlerror(); // clean up error;
  return dlsym(plugin, name.c_str());
}
std::string plugin_error(vitis::ai::plugin_t plugin) {
  std::ostringstream str;
  str << "ERROR CODE: " << dlerror();
  return str.str();
}
void close_plugin(vitis::ai::plugin_t plugin) { dlclose(plugin); }
#endif

} // namespace vaip_core