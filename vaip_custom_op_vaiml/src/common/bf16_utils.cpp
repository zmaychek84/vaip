/*
 *     The Xilinx Vitis AI Vaip in this distribution are provided under the
 * following free and permissive binary-only license, but are not provided in
 * source code form.  While the following free and permissive license is similar
 * to the BSD open source license, it is NOT the BSD open source license nor
 * other OSI-approved open source license.
 *
 *      Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights
 * reserved.
 *
 *      Redistribution and use in binary form only, without modification, is
 * permitted provided that the following conditions are met:
 *
 *      1. Redistributions must reproduce the above copyright notice, this list
 * of conditions and the following disclaimer in the documentation and/or other
 * materials provided with the distribution.
 *
 *      2. The name of Xilinx, Inc. may not be used to endorse or promote
 * products redistributed with this software without specific prior written
 * permission.
 *
 *      THIS SOFTWARE IS PROVIDED BY XILINX, INC. "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL XILINX, INC. BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 *      PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE
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
uint16_t float_to_bfloat16_scalar(float value) {
  // Reinterpret the float as an unsigned 32-bit integer
  uint32_t int_value;
  std::memcpy(&int_value, &value, sizeof(value));

  // Extract the least significant bit (lsb) of the original float's lower 16
  // bits
  uint32_t lsb = (int_value >> 16) & 1;

  // Calculate the bias
  uint32_t bias = 0x7FFF + lsb;

  // Add the bias to the original integer value
  uint32_t new_value = int_value + bias;

  // Shift the new value right by 16 bits to get the bfloat16 representation
  uint16_t bfloat16_value = (uint16_t)(new_value >> 16);

  return bfloat16_value;
}
void float_to_bfloat16_avx512_unrolled(const float* v, uint16_t* out,
                                       size_t size) {
  constexpr size_t nelems_in_vector = sizeof(__m512) / sizeof(float);
  constexpr size_t unroll_factor = 4;
  constexpr size_t nelems_per_loop = nelems_in_vector * unroll_factor;

  static const __m512i ones = _mm512_set1_epi32(0x1);           // 1
  static const __m512i round_value = _mm512_set1_epi32(0x7fff); // 1

  const uint32_t* v32 = reinterpret_cast<const uint32_t*>(v);
  size_t i = 0;
  for (; i < (size / nelems_per_loop) * nelems_per_loop; i += nelems_per_loop) {
    // std::cout << "compute vector : " << i << std::endl;
    __m512i a0 = _mm512_loadu_epi32(v32 + i + nelems_in_vector * 0);
    __m512i a1 = _mm512_loadu_epi32(v32 + i + nelems_in_vector * 1);
    __m512i a2 = _mm512_loadu_epi32(v32 + i + nelems_in_vector * 2);
    __m512i a3 = _mm512_loadu_epi32(v32 + i + nelems_in_vector * 3);

    _mm_prefetch((const char*)v32 +
                     (i + nelems_in_vector * 0 + nelems_per_loop) * 4,
                 _MM_HINT_T0);
    _mm_prefetch((const char*)v32 +
                     (i + nelems_in_vector * 1 + nelems_per_loop) * 4,
                 _MM_HINT_T0);
    _mm_prefetch((const char*)v32 +
                     (i + nelems_in_vector * 2 + nelems_per_loop) * 4,
                 _MM_HINT_T0);
    _mm_prefetch((const char*)v32 +
                     (i + nelems_in_vector * 3 + nelems_per_loop) * 4,
                 _MM_HINT_T0);

    __m512i c0 = _mm512_srli_epi32(a0, 16);
    __m512i c1 = _mm512_srli_epi32(a1, 16);
    __m512i c2 = _mm512_srli_epi32(a2, 16);
    __m512i c3 = _mm512_srli_epi32(a3, 16);

    __m512i lsb0 = _mm512_and_epi32(c0, ones);
    __m512i lsb1 = _mm512_and_epi32(c1, ones);
    __m512i lsb2 = _mm512_and_epi32(c2, ones);
    __m512i lsb3 = _mm512_and_epi32(c3, ones);

    __m512i bias0 = _mm512_add_epi32(lsb0, round_value);
    __m512i bias1 = _mm512_add_epi32(lsb1, round_value);
    __m512i bias2 = _mm512_add_epi32(lsb2, round_value);
    __m512i bias3 = _mm512_add_epi32(lsb3, round_value);

    __m512i d0 = _mm512_add_epi32(a0, bias0);
    __m512i d1 = _mm512_add_epi32(a1, bias1);
    __m512i d2 = _mm512_add_epi32(a2, bias2);
    __m512i d3 = _mm512_add_epi32(a3, bias3);

    __m512i e0 = _mm512_srli_epi32(d0, 16);
    __m512i e1 = _mm512_srli_epi32(d1, 16);
    __m512i e2 = _mm512_srli_epi32(d2, 16);
    __m512i e3 = _mm512_srli_epi32(d3, 16);

    __m256i z0 = _mm512_cvtusepi32_epi16(e0);
    __m256i z1 = _mm512_cvtusepi32_epi16(e1);
    __m256i z2 = _mm512_cvtusepi32_epi16(e2);
    __m256i z3 = _mm512_cvtusepi32_epi16(e3);

    //_mm256_storeu_epi16(out + i + nelems_in_vector * 0, z0);
    //_mm256_storeu_epi16(out + i + nelems_in_vector * 1, z1);
    //_mm256_storeu_epi16(out + i + nelems_in_vector * 2, z2);
    //_mm256_storeu_epi16(out + i + nelems_in_vector * 3, z3);

    _mm256_stream_si256((__m256i*)(out + i + nelems_in_vector * 0), z0);
    _mm256_stream_si256((__m256i*)(out + i + nelems_in_vector * 1), z1);
    _mm256_stream_si256((__m256i*)(out + i + nelems_in_vector * 2), z2);
    _mm256_stream_si256((__m256i*)(out + i + nelems_in_vector * 3), z3);
    // break;
  }
  for (; i < size; ++i) {
    out[i] = float_to_bfloat16_scalar(v[i]);
  }
  _mm_sfence();
}

float bfloat16_to_float_single(uint16_t v) {
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
void bfloat16_to_float_avx512_unrolled(uint16_t* s, float* d, int n) {
  // __m512 _mm512_cvtpbh_ps (__m256bh a)
  constexpr size_t nelems_in_vector = sizeof(__m256) / sizeof(uint16_t);
  constexpr size_t unroll_factor = 4;
  constexpr size_t nelems_per_loop = nelems_in_vector * unroll_factor;
  size_t i = 0;

  for (; i < (n / nelems_per_loop) * nelems_per_loop; i += nelems_per_loop) {
    __m256bh a0 = _mm256_loadu_ph(s + i + nelems_in_vector * 0);
    __m256bh a1 = _mm256_loadu_ph(s + i + nelems_in_vector * 1);
    __m256bh a2 = _mm256_loadu_ph(s + i + nelems_in_vector * 2);
    __m256bh a3 = _mm256_loadu_ph(s + i + nelems_in_vector * 3);

    _mm_prefetch((const char*)s +
                     (i + nelems_in_vector * 0 + nelems_per_loop) * 4,
                 _MM_HINT_T0);
    _mm_prefetch((const char*)s +
                     (i + nelems_in_vector * 1 + nelems_per_loop) * 4,
                 _MM_HINT_T0);
    _mm_prefetch((const char*)s +
                     (i + nelems_in_vector * 2 + nelems_per_loop) * 4,
                 _MM_HINT_T0);
    _mm_prefetch((const char*)s +
                     (i + nelems_in_vector * 3 + nelems_per_loop) * 4,
                 _MM_HINT_T0);

    __m512 b0 = _mm512_cvtpbh_ps(a0);
    __m512 b1 = _mm512_cvtpbh_ps(a1);
    __m512 b2 = _mm512_cvtpbh_ps(a2);
    __m512 b3 = _mm512_cvtpbh_ps(a3);

    _mm512_storeu_ps(d + i + nelems_in_vector * 0, b0);
    _mm512_storeu_ps(d + i + nelems_in_vector * 1, b1);
    _mm512_storeu_ps(d + i + nelems_in_vector * 2, b2);
    _mm512_storeu_ps(d + i + nelems_in_vector * 3, b3);
  }

  for (; i < n; ++i) {
    d[i] = bfloat16_to_float_single(s[i]);
  }
  _mm_sfence();
}

float dequant(int64_t x, int64_t zp, double scale) {
  double q = (double)(x - zp);
  return (float)(q * (float)scale);
}

uint16_t float_to_bfloat16(float x) {
  uint32_t i;
  uint8_t* src = (uint8_t*)&x;
  uint8_t* tmp = (uint8_t*)&i;
  // copy float to uint32_t
  tmp[0] = src[0];
  tmp[1] = src[1];
  tmp[2] = src[2];
  tmp[3] = src[3];
  // round to nearest even
  uint32_t lsb = (i >> 16) & 0x1;
  uint32_t bias = 0x7fff + lsb;
  i += bias;
  // extract upper half of input
  uint16_t y = uint16_t(i >> 16);
  return y;
}

float bfloat16_to_float(uint16_t x) {
  float y = 0.0;
  uint8_t* src = (uint8_t*)&x;
  uint8_t* dst = (uint8_t*)&y;
  dst[2] = src[0];
  dst[3] = src[1];
  return y;
}

float bfloat16_rnd_even(float x) {
  // Simulate rounding bfloat16 to nearest even
  return bfloat16_to_float(float_to_bfloat16(x));
}

} // namespace vaip_vaiml_custom_op
