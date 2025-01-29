/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */
#include "vaip/vaip.hpp"

namespace vaip_core {
StatProto& get_stat_proto();
void clean_stat();
void collect_stat(const onnxruntime::Graph& graph, const ContextProto& context);
} // namespace vaip_core
