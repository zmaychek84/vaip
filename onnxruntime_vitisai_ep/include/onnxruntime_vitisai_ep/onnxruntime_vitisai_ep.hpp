/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */
#pragma once
#include <filesystem>
#include <vector>
#if defined(_WIN32)
#  if ONNXRUNTIME_VITISAI_EP_EXPORT_DLL == 1
#    define ONNXRUNTIME_VITISAI_EP_DLL_SPEC __declspec(dllexport)
#  else
#    define ONNXRUNTIME_VITISAI_EP_DLL_SPEC __declspec(dllimport)
#  endif
#else
#  define ONNXRUNTIME_VITISAI_EP_DLL_SPEC __attribute__((visibility("default")))
#endif

#ifndef USE_VITISAI
#  define USE_VITISAI /* mimic VITISAI EP in ORT */
#endif
