/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

#pragma once
#include <memory>
#include <string>
#include <vaip/vaip.hpp>
namespace vaip_core {
class Pattern;
}
namespace vaip {
namespace pattern_zoo {
VAIP_DLL_SPEC std::vector<std::string> pattern_list();
VAIP_DLL_SPEC std::shared_ptr<vaip_core::Pattern>
get_pattern(const std::string& name);
} // namespace pattern_zoo
} // namespace vaip
