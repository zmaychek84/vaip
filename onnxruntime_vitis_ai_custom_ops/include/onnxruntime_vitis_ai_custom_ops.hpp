/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */
#pragma once
#define ORT_API_MANUAL_INIT
#include <onnxruntime_cxx_api.h>
#undef ORT_API_MANUAL_INIT

#ifdef __cplusplus
extern "C" {
#endif
ORT_EXPORT OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options,
                                                     const OrtApiBase* api);

// alternative name to test registration by function name
ORT_EXPORT OrtStatus* ORT_API_CALL
RegisterCustomOpsAltName(OrtSessionOptions* options, const OrtApiBase* api);

#ifdef __cplusplus
}
#endif
