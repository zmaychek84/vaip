/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

#include <glog/logging.h>
// include glog/logging.h to define CHECK before include vaip_plugin.hpp
#include "vaip/plugin.hpp"
#include "vaip/vaip_plugin.hpp"
#include "vitis/ai/weak.hpp"

namespace vaip_core {
std::string Plugin::guess_name(const char* name) {
#ifdef _WIN32
  return std::string("") + name + ".dll";
#else
  return std::string("lib") + name + ".so";
#endif
}
struct Tag_Plugin_Func_Set {
  vitis::ai::plugin_t (*open_plugin)(const std::string& name,
                                     vitis::ai::scope_t scope);
  void* (*plugin_sym)(vitis::ai::plugin_t plugin, const std::string& name);
  void (*close_plugin)(vitis::ai::plugin_t plugin);
};

Plugin_Func_Set g_static_plugin_func_set = {
    vitis::ai::open_plugin, vitis::ai::plugin_sym, vitis::ai::close_plugin};

Plugin_Func_Set g_dynamic_plugin_func_set = {
    vaip_core::open_plugin, vaip_core::plugin_sym, vaip_core::close_plugin};

Plugin_Func_Set* g_static_plugin_func_set_ptr = &g_static_plugin_func_set;
Plugin_Func_Set* g_dynamic_plugin_func_set_ptr = &g_dynamic_plugin_func_set;

Plugin::Plugin(const char* name, Plugin_Func_Set* func_set)
    : name_{name}, so_name_{guess_name(name)}, func_set_{func_set},
      plugin_{func_set->open_plugin(so_name_, vitis::ai::scope_t::PUBLIC)} {
  CHECK(plugin_ != nullptr) << "cannot open plugin: "
                            << "name_ " << name_ << " "       //
                            << "so_name_ " << so_name_ << " " //
      ;
}

Plugin::~Plugin() { func_set_->close_plugin((vitis::ai::plugin_t)plugin_); }

std::unordered_map<std::string, std::shared_ptr<Plugin>> Plugin::store_;

Plugin* Plugin::get(const std::string& plugin_name, Plugin_Func_Set* func_set) {
  auto it = store_.find(plugin_name);
  if (it == store_.end()) {
    store_[plugin_name] = vitis::ai::WeakStore<std::string, Plugin>::create(
        plugin_name, plugin_name.c_str(), func_set);
  }
  it = store_.find(plugin_name);
  CHECK(it != store_.end())
      << "cannot load plugin. plugin_name=" << plugin_name;
  return it->second.get();
}
void* Plugin::my_plugin_sym(void* handle, const char* name) {
  return func_set_->plugin_sym((vitis::ai::plugin_t)handle, name);
}
} // namespace vaip_core
