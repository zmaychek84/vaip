/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

#pragma once
#include <vaip/vaip.hpp>

namespace vaip_pass_fuse_xmodel {
using namespace vaip_core;
void fuse_xmodel(Graph& graph);
} // namespace vaip_pass_fuse_xmodel
