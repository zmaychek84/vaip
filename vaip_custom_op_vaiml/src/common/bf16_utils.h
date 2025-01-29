/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */
#pragma once

#include <intrin.h>
#include <mmintrin.h>
#include <xmmintrin.h>

#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <iterator>
#include <memory>
#include <utility>
#include <vector>

#if defined(_WIN32)
#  pragma warning(disable : 4996)
#endif
namespace vaip_vaiml_custom_op {
// Function to convert float to bfloat16
uint16_t float_to_bfloat16_scalar(float value);
void float_to_bfloat16_avx512_unrolled(const float* v, uint16_t* out,
                                       size_t size);

float bfloat16_to_float_single(uint16_t v);
void bfloat16_to_float_full(uint16_t* s, float* d, int n);

void bfloat16_to_float_avx512_unrolled(uint16_t* s, float* d, int n);

float dequant(int64_t x, int64_t zp, double scale);

uint16_t float_to_bfloat16(float x);

float bfloat16_to_float(uint16_t x);

float bfloat16_rnd_even(float x);

} // namespace vaip_vaiml_custom_op
