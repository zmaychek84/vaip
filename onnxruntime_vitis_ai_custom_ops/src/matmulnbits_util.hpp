/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

/*

**/

#ifndef GUARD_ONNX_CUSTOM_OPS_HYBRID_LLM_MATMULNBITS_UTIL
#define GUARD_ONNX_CUSTOM_OPS_HYBRID_LLM_MATMULNBITS_UTIL

#include <immintrin.h>
#if defined(_WIN32)
#  include <intrin.h>
#else
#  include <x86intrin.h>
#endif
#include <mmintrin.h>
#include <xmmintrin.h>

#if defined(_WIN32)
#  pragma warning(disable : 4996)
#endif

namespace ryzenai::onnx_utils {

inline float bfloat16_to_float_single(uint16_t v) {
  union {
    uint32_t i;
    float f;
  } u;
  u.i = (uint32_t(v)) << 16;
  return u.f;
}
void bfloat16_to_float_full(uint16_t* s, float* d, int n) {
  uint16_t* s1 = s;
  float* d1 = d;
  int i = 0;

#define L 8

  union {
    uint32_t i[L];
    float f[L];
  } u;
  for (i = 0; i < n / L; ++i) {
    u.i[0] = (uint32_t(s1[0])) << 16;
    u.i[1] = (uint32_t(s1[1])) << 16;
    u.i[2] = (uint32_t(s1[2])) << 16;
    u.i[3] = (uint32_t(s1[3])) << 16;
    u.i[4] = (uint32_t(s1[4])) << 16;
    u.i[5] = (uint32_t(s1[5])) << 16;
    u.i[6] = (uint32_t(s1[6])) << 16;
    u.i[7] = (uint32_t(s1[7])) << 16;
    d1[0] = u.f[0];
    d1[1] = u.f[1];
    d1[2] = u.f[2];
    d1[3] = u.f[3];
    d1[4] = u.f[4];
    d1[5] = u.f[5];
    d1[6] = u.f[6];
    d1[7] = u.f[7];
    s1 += L;
    d1 += L;
  }
  for (; i < n; ++i) {
    d[i] = bfloat16_to_float_single(s[i]);
  }
}

} // namespace ryzenai::onnx_utils

#endif // GUARD_ONNX_CUSTOM_OPS_HYBRID_LLM_MATMULNBITS_UTIL
