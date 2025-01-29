/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

#pragma once

#include <stdint.h>
#include <string>

namespace vaip_core {
const std::string get_lib_name();
const std::string get_lib_id();
uint32_t get_vaip_version_major();
uint32_t get_vaip_version_minor();
uint32_t get_vaip_version_patch();
} // namespace vaip_core
