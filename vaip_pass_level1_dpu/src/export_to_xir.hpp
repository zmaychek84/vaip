/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

#pragma once
//
#include <vaip/vaip.hpp>

#include "vaip/xir_headers.hpp"
#include <memory>

namespace vaip_core {

std::unique_ptr<xir::Graph> export_to_xir(IPass& pass,
                                          onnxruntime::Graph& graph);

} // namespace vaip_core
