/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

#pragma once

#include <memory>

#include "vaip/vaip.hpp"
#include "vaip/xir_headers.hpp"

namespace vaip_core {
std::string get_xclbin_fullpath(const std::string& xclbin);
std::string get_xcompiler_fingerprint(const PassContext& pass_context,
                                      const PassDpuParamProto& dpu_param);
std::unique_ptr<xir::Graph>
compiler_xir_model(std::unique_ptr<xir::Graph> graph,
                   const PassContext& pass_context,
                   const PassDpuParamProto& dpu_param);

} // namespace vaip_core
