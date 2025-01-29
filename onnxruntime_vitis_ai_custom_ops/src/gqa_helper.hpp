/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */
#pragma once
#if defined(_WIN32)
#  include <intrin.h>
#else
#  include "ryzenai/dynamic_dispatch/utils/instruction_cache.hpp"
#  include "ryzenai/dynamic_dispatch/utils/instruction_registry.hpp"
#  include <ryzenai/dynamic_dispatch/xrt_context/xrt_context.hpp>
#  include <x86intrin.h>
#endif
#include <immintrin.h>
#include <mmintrin.h>
#include <xmmintrin.h>

#include <cassert>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <vector>

// Update based on the max sequence length to be supported
#define MAX_SEQ_LENGTH 3072

static std::string shape2str(const std::vector<int64_t>& v) {
  std::stringstream ss("");
  for (size_t i = 0; i < v.size() - 1; i++)
    ss << v[i] << "x";
  ss << v[v.size() - 1];
  return ss.str();
}

static void saveDataToFile(const uint16_t* data, size_t size,
                           const std::string& filename) {
  std::ofstream outfile(filename, std::ios::binary);
  if (!outfile.is_open()) {
    std::cerr << "Can not open " << filename << " to write." << std::endl;
    return;
  }
  outfile.write(reinterpret_cast<const char*>(data), size * sizeof(int16_t));
  outfile.close();
}

static uint16_t float_to_bfloat16(float x) {
  uint32_t i;
  uint8_t* src = (uint8_t*)&x;
  uint8_t* tmp = (uint8_t*)&i;
  std::memcpy(tmp, src, sizeof(float));
  uint32_t lsb = (i >> 16) & 0x1;
  uint32_t bias = 0x7fff + lsb;
  i += bias;
  uint16_t y = uint16_t(i >> 16);
  return y;
}

static float bfloat16_to_float_single(uint16_t v) {
  union {
    uint32_t i;
    float f;
  } u;
  u.i = (uint32_t(v)) << 16;
  return u.f;
}

static void float_to_bfloat16_avx512_unrolled(const float* v, uint16_t* out,
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

    _mm256_stream_si256((__m256i*)(out + i + nelems_in_vector * 0), z0);
    _mm256_stream_si256((__m256i*)(out + i + nelems_in_vector * 1), z1);
    _mm256_stream_si256((__m256i*)(out + i + nelems_in_vector * 2), z2);
    _mm256_stream_si256((__m256i*)(out + i + nelems_in_vector * 3), z3);
  }
  for (; i < size; ++i) {
    out[i] = float_to_bfloat16(v[i]);
  }
  _mm_sfence();
}
#ifdef _WIN32
static void bfloat16_to_float_avx512_unrolled(const uint16_t* s, float* d,
                                              int n) {
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
#else
inline __m512 bf16_to_fp32(__m256i bf16_data) {
  // Convert 16-bit bfloat16 to 32-bit by shifting left by 16 bits (move to
  // higher half of FP32)
  __m512i temp = _mm512_slli_epi32(_mm512_cvtepu16_epi32(bf16_data), 16);

  // Cast the 32-bit integer as 32-bit floating-point values
  return _mm512_castsi512_ps(temp);
}

static void bfloat16_to_float_avx512_unrolled(const uint16_t* s, float* d,
                                              int n) {
  // __m512 _mm512_cvtpbh_ps (__m256bh a)
  constexpr size_t nelems_in_vector = sizeof(__m256) / sizeof(uint16_t);
  constexpr size_t unroll_factor = 4;
  constexpr size_t nelems_per_loop = nelems_in_vector * unroll_factor;
  size_t i = 0;

  for (; i < (n / nelems_per_loop) * nelems_per_loop; i += nelems_per_loop) {
    // __m256bh a0 = (__m256bh)_mm256_loadu_si256((__m256i*)ptr);
    // Load 16-bit integers (FP16 values)
    __m256i a0 = _mm256_loadu_si256((__m256i*)(s + i + nelems_in_vector * 0));
    __m256i a1 = _mm256_loadu_si256((__m256i*)(s + i + nelems_in_vector * 1));
    __m256i a2 = _mm256_loadu_si256((__m256i*)(s + i + nelems_in_vector * 2));
    __m256i a3 = _mm256_loadu_si256((__m256i*)(s + i + nelems_in_vector * 3));

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

    __m512 b0 = bf16_to_fp32(a0);
    __m512 b1 = bf16_to_fp32(a1);
    __m512 b2 = bf16_to_fp32(a2);
    __m512 b3 = bf16_to_fp32(a3);

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
#endif

static void vec_float32_to_bf16(uint16_t* dest, const float* src, size_t size) {
  assert(src != nullptr);
  assert(dest != nullptr);
  assert(size > 0);
  float_to_bfloat16_avx512_unrolled(src, dest, size);
}

static void vec_bf16_to_float(float* dest, const uint16_t* src, size_t size) {
  assert(src != nullptr);
  assert(dest != nullptr);
  assert(size > 0);

  bfloat16_to_float_avx512_unrolled(src, dest, size);
}

static void fill_attn_mask_impl(uint16_t* attn_mask, int S) {
  std::memset(attn_mask, 0, S * S * sizeof(uint16_t));
  const uint16_t neg_inf_ui16 = float_to_bfloat16(-3.389e38f);
  for (int i = 0; i < S; i++) {
    uint16_t* start_ptr = attn_mask + i * S + (i + 1);
    size_t count = S - (i + 1);
#ifdef _WIN32
    __stosw(start_ptr, neg_inf_ui16, count);
#else
    std::fill(start_ptr, start_ptr + count, neg_inf_ui16);
#endif
  }
}

static void fill_attn_mask_3d(uint16_t* attn_mask, int S, int S_pad, int past_S,
                              int kv_len) {
  const int all_sequence_length = past_S + kv_len;

  std::memset(attn_mask, 0, S_pad * all_sequence_length * sizeof(uint16_t));
  const uint16_t neg_inf_ui16 = float_to_bfloat16(-3.389e38f);
  for (int i = 0; i < S; i++) {
    uint16_t* start_ptr = attn_mask + i * all_sequence_length + i + 1 + past_S;
    size_t count = all_sequence_length - (i + 1 + past_S);
#ifdef _WIN32
    __stosw(start_ptr, neg_inf_ui16, count);
#else
    std::fill(start_ptr, start_ptr + count, neg_inf_ui16);
#endif
  }
  if (S < S_pad) {
    uint16_t* start_ptr = attn_mask + S * all_sequence_length;
#ifdef _WIN32
    __stosw(start_ptr, neg_inf_ui16, ((S_pad - S) * all_sequence_length));
#else
    std::fill(start_ptr, start_ptr + ((S_pad - S) * all_sequence_length),
              neg_inf_ui16);
#endif
  }
}

/*
Note(chuanliang & ltp):
This is LUT constructor for attentation mask.
We will create as many LUT(128 as step) for different S as possble, it's up to
the invoker to decide which LUT to use.
*/
class AttnMaskLUTSingleton {

private:
  // Add more in the future if required.
  static std::set<int32_t> get_seqs() {
    return {128, 256, 512, 1024, 2048, 3072};
  }

public:
  static AttnMaskLUTSingleton& getInstance() {
    static AttnMaskLUTSingleton instance;
    return instance;
  }

  bool hasLut(int32_t S) { return idx_map_.count(S) != 0; }

  /// Should use const.
  uint16_t* getLut(int32_t S) {
    if (!hasLut(S)) {
      throw std::invalid_argument("There is not LUT for " + std::to_string(S));
    }

    auto idx = idx_map_.at(S);
    return attn_masks_lut_[idx].data();
  }

private:
  AttnMaskLUTSingleton() {
    // construct the LUT
    auto seqs = get_seqs();
    int32_t idx = 0;
    for (auto& s : seqs) {
      std::vector<uint16_t> lut;
      lut.resize(s * s);
      fill_attn_mask_impl(lut.data(), s);
      attn_masks_lut_.emplace_back(std::move(lut));
      idx_map_[s] = idx++;
    }
  }

  // index for LUT given a S
  std::unordered_map<uint16_t, int32_t> idx_map_;
  std::vector<std::vector<uint16_t>> attn_masks_lut_;

  AttnMaskLUTSingleton(const AttnMaskLUTSingleton&) = delete;
  void operator=(const AttnMaskLUTSingleton&) = delete;
};

class AttnMask3DLUTSingleton {

private:
  // Add more in the future if required.
  static std::set<int32_t> get_seqs() {
    return {64, 128, 192, 256, 320, 384, 448, 512};
  }

public:
  static AttnMask3DLUTSingleton& getInstance(int32_t chunk_size) {
    static AttnMask3DLUTSingleton instance(chunk_size);
    return instance;
  }

  bool hasLut(int32_t S) { return idx_map_.count(S) != 0; }

  /// Should use const.
  uint16_t* getLut(int32_t S) {
    if (!hasLut(S)) {
      throw std::invalid_argument("There is not LUT for " + std::to_string(S));
    }

    auto idx = idx_map_.at(S);
    return attn_masks_lut_[idx].data();
  }

private:
  AttnMask3DLUTSingleton(int32_t chunk_size) {
    // construct the LUT
    auto seqs = get_seqs();
    int32_t idx = 0;
    for (auto& s : seqs) {
      std::vector<uint16_t> lut;
      lut.resize(chunk_size * s);
      fill_attn_mask_3d(lut.data(), chunk_size, chunk_size, s - chunk_size,
                        chunk_size);
      attn_masks_lut_.emplace_back(std::move(lut));
      idx_map_[s] = idx++;
    }
  }

  // index for LUT given a S
  std::unordered_map<uint16_t, int32_t> idx_map_;
  std::vector<std::vector<uint16_t>> attn_masks_lut_;

  AttnMask3DLUTSingleton() = delete;
  void operator=(const AttnMask3DLUTSingleton&) = delete;
};
