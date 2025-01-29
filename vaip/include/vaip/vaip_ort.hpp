/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

#pragma once
#include "./_sanity_check.hpp"
#if __has_include(<filesystem>)  && __cplusplus > 201700
#  include <filesystem>
#else
#  error "must enable c++17"
#endif

#include <vaip/custom_op.h>
#include <vaip/export.h>
#include <vaip/vaip_ort_api.h>
/// header file used by ort VITISAI execution providers.

using EventInfo = std::tuple<std::string, // name
                             int,         // pid
                             int,         // tid
                             long long,   // timestamp
                             long long    // duration
                             >;

namespace vaip_core {
class PassContextImp;

VAIP_DLL_SPEC void
initialize_onnxruntime_vitisai_ep(OrtApiForVaip* api,
                                  std::vector<OrtCustomOpDomain*>& ret_domain);

VAIP_DLL_SPEC void set_the_global_api(OrtApiForVaip* api);

VAIP_DLL_SPEC const OrtApiForVaip* api();

VAIP_DLL_SPEC std::vector<std::unique_ptr<ExecutionProvider>>
compile_onnx_model_3(const std::string& model_path, const Graph& graph,
                     const char* json_config);

int optimize_onnx_model(const std::filesystem::path& model_path_in,
                        const std::filesystem::path& model_path_out,
                        const char* json_config);

void initialize_graph_optimizer(const std::string& json_path);

std::shared_ptr<PassContextImp>
initialize_context(const std::string& model_path, const Graph& onnx_graph,
                   const char* json_config);

VAIP_DLL_SPEC void
get_compilation_cache(const std::string& model_path, const Graph& graph,
                      const char* json_config, uint8_t compiler_codes,
                      std::string& cache_dir, std::string& cache_key,
                      std::string& cache_data);

VAIP_DLL_SPEC void restore_compilation_cache(const std::string& cache_dir,
                                             const std::string& cache_key,
                                             const std::string& cache_data,
                                             const std::string& model_path);
} // namespace vaip_core
