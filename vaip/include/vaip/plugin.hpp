/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

#pragma once
#include "vitis/ai/plugin.hpp"
#include <string>

namespace vaip_core {
vitis::ai::plugin_t open_plugin(const std::string& name,
                                vitis::ai::scope_t scope);
void* plugin_sym(vitis::ai::plugin_t plugin, const std::string& name);
void close_plugin(vitis::ai::plugin_t plugin);
} // namespace vaip_core
