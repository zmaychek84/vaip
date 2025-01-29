/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

#include <glog/logging.h>

#include "vaip/op_def.hpp"
#include "vitis/ai/env_config.hpp"
using namespace vaip_core;

#include <immintrin.h>
#include <iostream>
#include <map>
#include <memory>

namespace cpu_support {

struct CPUSupport {
  bool avx = true;
  bool avx2 = true;
  bool avx512f = true;
  bool avx512bw = true;
  bool avx512vbmi2 = true;
  bool avx512vl = true;
};

const CPUSupport cpuSupport;

} // namespace cpu_support

namespace {
template <typename T>
static std::ostream& operator<<(std::ostream& s, const std::vector<T>& v) {
  int index = 0;
  s << "[";
  for (auto c : v) {
    if (index++ != 0) {
      s << ",";
    }
    s << c;
  }
  s << "]";
  return s;
}

template <typename T> inline T rounder(float data, int data_min, int data_max) {
  T rlt = 0;
  if (data > data_max) {
    rlt = data_max;
  } else if (data < data_min) {
    rlt = data_min;
  } else if ((data - floor(data)) == 0.5) {
    rlt = T(std::round(data * 0.5f) * 2.0f);
  } else {
    rlt = static_cast<T>(round(data));
  }
  return rlt;
}

template <typename T> inline T rounder(float data) {
  static const int data_max = std::numeric_limits<T>::max();
  static const int data_min = std::numeric_limits<T>::min();
  T rlt = 0;
  if (data > data_max) {
    rlt = data_max;
  } else if (data < data_min) {
    rlt = data_min;
  } else if ((data - floor(data)) == 0.5) {
    rlt = T(std::round(data * 0.5f) * 2.0f);
  } else {
    rlt = static_cast<T>(round(data));
  }
  return rlt;
}

#ifdef _WIN32
bool isAligned(void* data, int alignment) {
  return ((uintptr_t)data & (alignment - 1)) == 0;
}
inline void float2int16_avx512_stream(const float* src, __m256i* dst,
                                      std::size_t num_elements,
                                      const float scale,
                                      const int16_t zero_point) {
  constexpr std::size_t VECTOR_SIZE_BYTES = sizeof(__m512);
  constexpr std::size_t FLOAT_SIZE_BYTES = sizeof(float);
  constexpr std::size_t FLOATS_PER_VECTOR =
      VECTOR_SIZE_BYTES / FLOAT_SIZE_BYTES;

  static_assert(FLOAT_SIZE_BYTES == 4, "Unexpected float size!");

  const std::size_t num_iter = num_elements / FLOATS_PER_VECTOR;
  const std::size_t remainder = num_elements - (num_iter * FLOATS_PER_VECTOR);

  const __m512 scale_vector = _mm512_set1_ps(scale);
  // const __m512 round_vector = _mm512_set1_ps(0.5f);
  const __m256i zero_point_vector = _mm256_set1_epi16(zero_point);
  for (std::size_t i = 0; i < num_iter; ++i) {
    _mm_prefetch((const char*)src + 64, _MM_HINT_T0);
    __m512 in = _mm512_loadu_ps(src);
    __m256i int16s =
        _mm512_cvtsepi32_epi16(_mm512_cvtps_epi32(_mm512_roundscale_ps(
            _mm512_div_ps(in, scale_vector), _MM_FROUND_TO_NEAREST_INT)));
    // print_m512i32(scaled);
    // print_m256i16(_mm512_cvtsepi32_epi16(scaled));
    int16s = _mm256_add_epi16(int16s, zero_point_vector);

    // if(num_elements >= 2048)
    _mm256_stream_si256(dst, int16s);
    // else
    // _mm256_storeu_si256((__m256i*)dst, int16s);
    // print_m256i16(int16s);
    src += FLOATS_PER_VECTOR;
    dst += 1;
  }
  _mm_sfence();
  if (remainder > 0) {
    std::transform(src, src + remainder, (std::int16_t*)dst,
                   [&](const float& src) {
                     return rounder<std::int16_t>(((src / scale) + zero_point));
                   });
  }
}
#endif

#ifdef _WIN32
inline void float2int16_avx512(const float* src, std::int16_t* dst,
                               std::size_t num_elements, const float scale,
                               const int16_t zero_point) {
  constexpr std::size_t VECTOR_SIZE_BYTES = sizeof(__m512);
  constexpr std::size_t FLOAT_SIZE_BYTES = sizeof(float);
  constexpr std::size_t FLOATS_PER_VECTOR =
      VECTOR_SIZE_BYTES / FLOAT_SIZE_BYTES;

  static_assert(FLOAT_SIZE_BYTES == 4, "Unexpected float size!");

  const std::size_t num_iter = num_elements / FLOATS_PER_VECTOR;
  const std::size_t remainder = num_elements - (num_iter * FLOATS_PER_VECTOR);

  const __m512 scale_vector = _mm512_set1_ps(scale);
  // const __m512 round_vector = _mm512_set1_ps(0.5f);
  const __m256i zero_point_vector = _mm256_set1_epi16(zero_point);
  for (std::size_t i = 0; i < num_iter; ++i) {
    _mm_prefetch((const char*)src + 64, _MM_HINT_T0);
    __m512 in = _mm512_loadu_ps(src);
    __m256i int16s =
        _mm512_cvtsepi32_epi16(_mm512_cvtps_epi32(_mm512_roundscale_ps(
            _mm512_div_ps(in, scale_vector), _MM_FROUND_TO_NEAREST_INT)));
    // print_m512i32(scaled);
    // print_m256i16(_mm512_cvtsepi32_epi16(scaled));
    int16s = _mm256_add_epi16(int16s, zero_point_vector);

    // if(num_elements >= 2048)
    // _mm256_stream_si256((__m256i*)dst, int16s);
    // else
    _mm256_storeu_si256((__m256i*)dst, int16s);
    // print_m256i16(int16s);
    src += FLOATS_PER_VECTOR;
    dst += FLOATS_PER_VECTOR;
  }
  if (remainder > 0) {
    std::transform(src, src + remainder, dst, [&](const float& src) {
      return rounder<std::int16_t>(((src / scale) + zero_point));
    });
  }
}
#endif

// To-do: MNDBG: confirm if const needed
void float2int16(const float* src, std::int16_t* dst, std::size_t num_elements,
                 const float scale, const int16_t zero_point) {
#ifdef _WIN32 // TODO _mm_storeu_epi8 is not defined on Linux
  static const bool avxSupport = cpu_support::cpuSupport.avx512f &&
                                 cpu_support::cpuSupport.avx512bw &&
                                 cpu_support::cpuSupport.avx512vl;
  if (avxSupport) {
    if (isAligned((void*)dst, sizeof(__m256i))) {
      float2int16_avx512_stream(src, (__m256i*)dst, num_elements, scale,
                                zero_point);
    } else {
      float2int16_avx512(src, dst, num_elements, scale, zero_point);
    }
  } else {
    std::transform(src, src + num_elements, dst, [&](const float& src) {
      return rounder<std::int16_t>(((src / scale) + zero_point));
    });
  }
#else
  std::transform(src, src + num_elements, dst, [&](const float& src) {
    return rounder<std::int16_t>(((src / scale) + zero_point));
  });
#endif
}

//  float -> uint16
#ifdef _WIN32
inline void float2uint16_avx512_stream(const float* src, __m256i* dst,
                                       std::size_t num_elements,
                                       const float scale,
                                       const uint16_t zero_point) {
  constexpr std::size_t VECTOR_SIZE_BYTES = sizeof(__m512);
  constexpr std::size_t FLOAT_SIZE_BYTES = sizeof(float);
  constexpr std::size_t FLOATS_PER_VECTOR =
      VECTOR_SIZE_BYTES / FLOAT_SIZE_BYTES;

  static_assert(FLOAT_SIZE_BYTES == 4, "Unexpected float size!");

  const std::size_t num_iter = num_elements / FLOATS_PER_VECTOR;
  const std::size_t remainder = num_elements - (num_iter * FLOATS_PER_VECTOR);

  const __m512 scale_vector = _mm512_set1_ps(scale);
  // const __m512 round_vector = _mm512_set1_ps(0.5f);
  const __m512i zero_point_vector = _mm512_set1_epi32(zero_point);
  __m512i max =
      _mm512_set1_epi32(std::numeric_limits<uint16_t>::max() - zero_point);
  __m512i min =
      _mm512_set1_epi32(std::numeric_limits<uint16_t>::min() - zero_point);
  for (std::size_t i = 0; i < num_iter; ++i) {
    _mm_prefetch((const char*)src + 64, _MM_HINT_T0);
    __m512 in = _mm512_loadu_ps(src);
    __m512i int32s = _mm512_cvtps_epi32(_mm512_roundscale_ps(
        _mm512_div_ps(in, scale_vector), _MM_FROUND_TO_NEAREST_INT));
    __m512i saturated = _mm512_min_epi32(int32s, max);
    saturated = _mm512_max_epi32(saturated, min);

    __m512i add_result = _mm512_add_epi32(saturated, zero_point_vector);
    __m128i x0 = _mm512_extracti32x4_epi32(add_result, 0);
    __m128i x1 = _mm512_extracti32x4_epi32(add_result, 1);
    __m128i x2 = _mm512_extracti32x4_epi32(add_result, 2);
    __m128i x3 = _mm512_extracti32x4_epi32(add_result, 3);
    __m128i pack0 = _mm_packus_epi32(x0, x1);
    __m128i pack1 = _mm_packus_epi32(x2, x3);
    __m256i packed = _mm256_set_m128i(pack1, pack0);

    _mm256_stream_si256(dst, packed);
    src += FLOATS_PER_VECTOR;
    dst += 1;
  }
  _mm_sfence();
  if (remainder > 0) {
    std::transform(
        src, src + remainder, (std::uint16_t*)dst, [&](const float& src) {
          return rounder<std::uint16_t>(
                     src / scale,
                     int32_t(std::numeric_limits<uint16_t>::min()) - zero_point,
                     int32_t(std::numeric_limits<uint16_t>::max()) -
                         zero_point) +
                 zero_point;
        });
  }
}
#endif

#ifdef _WIN32
inline void float2uint16_avx512(const float* src, std::uint16_t* dst,
                                std::size_t num_elements, const float scale,
                                const uint16_t zero_point) {
  constexpr std::size_t VECTOR_SIZE_BYTES = sizeof(__m512);
  constexpr std::size_t FLOAT_SIZE_BYTES = sizeof(float);
  constexpr std::size_t FLOATS_PER_VECTOR =
      VECTOR_SIZE_BYTES / FLOAT_SIZE_BYTES;

  static_assert(FLOAT_SIZE_BYTES == 4, "Unexpected float size!");

  const std::size_t num_iter = num_elements / FLOATS_PER_VECTOR;
  const std::size_t remainder = num_elements - (num_iter * FLOATS_PER_VECTOR);

  const __m512 scale_vector = _mm512_set1_ps(scale);
  // const __m512 round_vector = _mm512_set1_ps(0.5f);
  const __m512i zero_point_vector = _mm512_set1_epi32(zero_point);
  __m512i max =
      _mm512_set1_epi32(std::numeric_limits<uint16_t>::max() - zero_point);
  __m512i min =
      _mm512_set1_epi32(std::numeric_limits<uint16_t>::min() - zero_point);
  for (std::size_t i = 0; i < num_iter; ++i) {
    _mm_prefetch((const char*)src + 64, _MM_HINT_T0);
    __m512 in = _mm512_loadu_ps(src);
    __m512i int32s = _mm512_cvtps_epi32(_mm512_roundscale_ps(
        _mm512_div_ps(in, scale_vector), _MM_FROUND_TO_NEAREST_INT));
    __m512i saturated = _mm512_min_epi32(int32s, max);
    saturated = _mm512_max_epi32(saturated, min);

    __m512i add_result = _mm512_add_epi32(saturated, zero_point_vector);
    __m128i x0 = _mm512_extracti32x4_epi32(add_result, 0);
    __m128i x1 = _mm512_extracti32x4_epi32(add_result, 1);
    __m128i x2 = _mm512_extracti32x4_epi32(add_result, 2);
    __m128i x3 = _mm512_extracti32x4_epi32(add_result, 3);
    __m128i pack0 = _mm_packus_epi32(x0, x1);
    __m128i pack1 = _mm_packus_epi32(x2, x3);
    __m256i packed = _mm256_set_m128i(pack1, pack0);

    _mm256_storeu_si256((__m256i*)dst, packed);
    src += FLOATS_PER_VECTOR;
    dst += FLOATS_PER_VECTOR;
  }
  if (remainder > 0) {
    std::transform(src, src + remainder, dst, [&](const float& src) {
      return rounder<std::uint16_t>(
                 src / scale,
                 int32_t(std::numeric_limits<uint16_t>::min()) - zero_point,
                 int32_t(std::numeric_limits<uint16_t>::max()) - zero_point) +
             zero_point;
    });
  }
}
#endif
// To-do: MNDBG: confirm if const needed
void float2uint16(const float* src, std::uint16_t* dst,
                  std::size_t num_elements, const float scale,
                  const uint16_t zero_point) {
#ifdef _WIN32 // TODO _mm_storeu_epi8 is not defined on Linux
  static const bool avxSupport = cpu_support::cpuSupport.avx512f &&
                                 cpu_support::cpuSupport.avx512bw &&
                                 cpu_support::cpuSupport.avx512vl;
  if (avxSupport) {
    if (isAligned((void*)dst, sizeof(__m256i))) {
      float2uint16_avx512_stream(src, (__m256i*)dst, num_elements, scale,
                                 zero_point);
    } else {
      float2uint16_avx512(src, dst, num_elements, scale, zero_point);
    }
  } else {
    std::transform(src, src + num_elements, dst, [&](const float& src) {
      return rounder<std::uint16_t>(
                 src / scale,
                 int32_t(std::numeric_limits<uint16_t>::min()) - zero_point,
                 int32_t(std::numeric_limits<uint16_t>::max()) - zero_point) +
             zero_point;
    });
  }
#else
  std::transform(src, src + num_elements, dst, [&](const float& src) {
    return rounder<std::uint16_t>(((src / scale) + zero_point));
  });
#endif
}

#ifdef _WIN32
inline void int162float_avx512_stream256(const std::int16_t* src, float* dst,
                                         std::size_t num_elements,
                                         const float scale,
                                         const int16_t zero_point) {
  constexpr std::size_t VECTOR_SIZE_BYTES = sizeof(__m256i);
  constexpr std::size_t INT16_SIZE_BYTES = sizeof(int16_t);
  constexpr std::size_t INT16_PER_VECTOR = VECTOR_SIZE_BYTES / INT16_SIZE_BYTES;

  static_assert(INT16_SIZE_BYTES == 2, "Unexpected int16_t size!");
  const std::size_t num_iter = num_elements / INT16_PER_VECTOR;
  const std::size_t remainder = num_elements - (num_iter * INT16_PER_VECTOR);
  const __m512 scale_vector = _mm512_set1_ps(scale);
  const __m256i zero_point_vector = _mm256_set1_epi16(zero_point);
  for (std::size_t i = 0; i < num_iter; ++i) {
    _mm_prefetch((const char*)(src + 32), _MM_HINT_T0);
    __m256i in = _mm256_load_si256((__m256i*)src);
    __m512 mul = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(
                                   _mm256_sub_epi16(in, zero_point_vector))),
                               scale_vector);
    // _mm512_storeu_ps(dst, mul);
    __m256 temp0 = _mm512_extractf32x8_ps(mul, 0);
    __m256 temp1 = _mm512_extractf32x8_ps(mul, 1);
    _mm256_stream_ps(dst, temp0);
    _mm256_stream_ps(dst + 8, temp1);
    src += INT16_PER_VECTOR;
    dst += INT16_PER_VECTOR;
  }

  if (remainder > 0) {
    std::transform(src, src + remainder, dst, [&](const std::int16_t& src) {
      return (std::int16_t((src - zero_point)) * scale);
    });
  }
}
#endif

#ifdef _WIN32
inline void int162float_avx512(const std::int16_t* src, float* dst,
                               std::size_t num_elements, const float scale,
                               const int16_t zero_point) {
  constexpr std::size_t VECTOR_SIZE_BYTES = sizeof(__m256i);
  constexpr std::size_t INT16_SIZE_BYTES = sizeof(int16_t);
  constexpr std::size_t INT16_PER_VECTOR = VECTOR_SIZE_BYTES / INT16_SIZE_BYTES;

  static_assert(INT16_SIZE_BYTES == 2, "Unexpected int16_t size!");
  const std::size_t num_iter = num_elements / INT16_PER_VECTOR;
  const std::size_t remainder = num_elements - (num_iter * INT16_PER_VECTOR);
  const __m512 scale_vector = _mm512_set1_ps(scale);
  const __m256i zero_point_vector = _mm256_set1_epi16(zero_point);
  for (std::size_t i = 0; i < num_iter; ++i) {
    _mm_prefetch((const char*)(src + 32), _MM_HINT_T0);
    __m256i in = _mm256_load_si256((__m256i*)src);
    __m512 mul = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(
                                   _mm256_sub_epi16(in, zero_point_vector))),
                               scale_vector);
    _mm512_storeu_ps(dst, mul);

    src += INT16_PER_VECTOR;
    dst += INT16_PER_VECTOR;
  }

  if (remainder > 0) {
    std::transform(src, src + remainder, dst, [&](const std::int16_t& src) {
      return (std::int16_t((src - zero_point)) * scale);
    });
  }
}
#endif

void int162float(const std::int16_t* src, float* dst, std::size_t num_elements,
                 const float scale, const int16_t zero_point) {
#ifdef _WIN32 // TODO _mm_storeu_epi8 is not defined on Linux
  static const bool avxSupport = cpu_support::cpuSupport.avx512f &&
                                 cpu_support::cpuSupport.avx512bw &&
                                 cpu_support::cpuSupport.avx512vl;
  if (avxSupport) {
    if (isAligned((void*)dst, sizeof(__m256))) {
      int162float_avx512_stream256(src, dst, num_elements, scale, zero_point);
    } else {
      int162float_avx512(src, dst, num_elements, scale, zero_point);
    }
  } else {
    std::transform(src, src + num_elements, dst, [&](const std::int16_t& src) {
      return (std::int16_t((src - zero_point)) * scale);
    });
  }
#else
  std::transform(src, src + num_elements, dst, [&](const std::int16_t& src) {
    return (std::int16_t((src - zero_point)) * scale);
  });
#endif
}

// uint16 -> float
#ifdef _WIN32
inline void uint162float_avx512_stream256(const std::uint16_t* src, float* dst,
                                          std::size_t num_elements,
                                          const float scale,
                                          const uint16_t zero_point) {
  constexpr std::size_t VECTOR_SIZE_BYTES = sizeof(__m256i);
  constexpr std::size_t UINT16_SIZE_BYTES = sizeof(uint16_t);
  constexpr std::size_t UINT16_PER_VECTOR =
      VECTOR_SIZE_BYTES / UINT16_SIZE_BYTES;

  static_assert(UINT16_SIZE_BYTES == 2, "Unexpected uint16_t size!");
  const std::size_t num_iter = num_elements / UINT16_PER_VECTOR;
  const std::size_t remainder = num_elements - (num_iter * UINT16_PER_VECTOR);
  const __m512 scale_vector = _mm512_set1_ps(scale);
  const __m512i zero_point_vector = _mm512_set1_epi32(zero_point);
  for (std::size_t i = 0; i < num_iter; ++i) {
    _mm_prefetch((const char*)(src + 32), _MM_HINT_T0);
    __m256i in = _mm256_load_si256((__m256i*)src);
    // convert u16 to i32
    __m512i in_i32 = _mm512_cvtepu16_epi32(in);
    __m512i sub1 = _mm512_sub_epi32(in_i32, zero_point_vector);
    __m512 mul = _mm512_mul_ps(_mm512_cvtepi32_ps(sub1), scale_vector);

    // _mm512_storeu_ps(dst, mul);
    __m256 temp0 = _mm512_extractf32x8_ps(mul, 0);
    __m256 temp1 = _mm512_extractf32x8_ps(mul, 1);
    _mm256_stream_ps(dst, temp0);
    _mm256_stream_ps(dst + 8, temp1);
    src += UINT16_PER_VECTOR;
    dst += UINT16_PER_VECTOR;
  }

  if (remainder > 0) {
    std::transform(src, src + remainder, dst, [&](const std::uint16_t& src) {
      return ((std::int32_t(src) - zero_point) * scale);
    });
  }
}
#endif

#ifdef _WIN32
inline void uint162float_avx512(const std::uint16_t* src, float* dst,
                                std::size_t num_elements, const float scale,
                                const uint16_t zero_point) {
  constexpr std::size_t VECTOR_SIZE_BYTES = sizeof(__m256i);
  constexpr std::size_t UINT16_SIZE_BYTES = sizeof(uint16_t);
  constexpr std::size_t UINT16_PER_VECTOR =
      VECTOR_SIZE_BYTES / UINT16_SIZE_BYTES;

  static_assert(UINT16_SIZE_BYTES == 2, "Unexpected uint16_t size!");
  const std::size_t num_iter = num_elements / UINT16_PER_VECTOR;
  const std::size_t remainder = num_elements - (num_iter * UINT16_PER_VECTOR);
  const __m512 scale_vector = _mm512_set1_ps(scale);
  const __m512i zero_point_vector = _mm512_set1_epi32(zero_point);
  for (std::size_t i = 0; i < num_iter; ++i) {
    _mm_prefetch((const char*)(src + 32), _MM_HINT_T0);
    __m256i in = _mm256_load_si256((__m256i*)src);
    __m512i in_i32 = _mm512_cvtepu16_epi32(in);
    __m512i sub1 = _mm512_sub_epi32(in_i32, zero_point_vector);
    __m512 mul = _mm512_mul_ps(_mm512_cvtepi32_ps(sub1), scale_vector);
    _mm512_storeu_ps(dst, mul);
    src += UINT16_PER_VECTOR;
    dst += UINT16_PER_VECTOR;
  }

  if (remainder > 0) {
    std::transform(src, src + remainder, dst, [&](const std::uint16_t& src) {
      return ((std::int32_t(src) - zero_point) * scale);
    });
  }
}
#endif

void uint162float(const std::uint16_t* src, float* dst,
                  std::size_t num_elements, const float scale,
                  const uint16_t zero_point) {
#ifdef _WIN32 // TODO _mm_storeu_epi8 is not defined on Linux
  static const bool avxSupport = cpu_support::cpuSupport.avx512f &&
                                 cpu_support::cpuSupport.avx512bw &&
                                 cpu_support::cpuSupport.avx512vl;
  if (avxSupport) {
    if (isAligned((void*)dst, sizeof(__m256))) {
      uint162float_avx512_stream256(src, dst, num_elements, scale, zero_point);
    } else {
      uint162float_avx512(src, dst, num_elements, scale, zero_point);
    }
  } else {
    std::transform(src, src + num_elements, dst, [&](const std::uint16_t& src) {
      return ((std::int32_t(src) - zero_point) * scale);
    });
  }
#else
  std::transform(src, src + num_elements, dst, [&](const std::int16_t& src) {
    return ((std::int32_t(src) - zero_point) * scale);
  });
#endif
}

#ifdef _WIN32
inline void float2fix_avx512_stream(const float* src, __m128i* dst,
                                    std::size_t num_elements, const float scale,
                                    const int8_t zero_point) {
  constexpr std::size_t VECTOR_SIZE_BYTES = sizeof(__m512);
  constexpr std::size_t FLOAT_SIZE_BYTES = sizeof(float);
  constexpr std::size_t FLOATS_PER_VECTOR =
      VECTOR_SIZE_BYTES / FLOAT_SIZE_BYTES;

  static_assert(FLOAT_SIZE_BYTES == 4, "Unexpected float size!");

  const std::size_t num_iter = num_elements / FLOATS_PER_VECTOR;
  const std::size_t remainder = num_elements - (num_iter * FLOATS_PER_VECTOR);

  const __m512 scale_vector = _mm512_set1_ps(scale);
  // const __m512 round_vector = _mm512_set1_ps(0.5f);
  const __m128i zero_point_vector = _mm_set1_epi8(zero_point);
  for (std::size_t i = 0; i < num_iter; ++i) {
    _mm_prefetch((const char*)src + 64, _MM_HINT_T0);
    __m512 in = _mm512_loadu_ps(src);
    in = _mm512_roundscale_ps(_mm512_div_ps(in, scale_vector),
                              _MM_FROUND_TO_NEAREST_INT);
    __m128i int8s = _mm_add_epi8(_mm512_cvtsepi32_epi8(_mm512_cvtps_epi32(in)),
                                 zero_point_vector);
    // _mm_storeu_epi8(dst, int8s);
    _mm_stream_si128((__m128i*)dst, int8s);
    src += FLOATS_PER_VECTOR;
    dst += 1;
  }
  _mm_sfence();

  if (remainder > 0) {
    std::transform(src, src + remainder, (std::int8_t*)dst,
                   [&](const float& src) {
                     return rounder<std::int8_t>(src / scale) + zero_point;
                   });
  }
}
#endif

#ifdef _WIN32
inline void float2fix_avx512(const float* src, std::int8_t* dst,
                             std::size_t num_elements, const float scale,
                             const int8_t zero_point) {
  constexpr std::size_t VECTOR_SIZE_BYTES = sizeof(__m512);
  constexpr std::size_t FLOAT_SIZE_BYTES = sizeof(float);
  constexpr std::size_t FLOATS_PER_VECTOR =
      VECTOR_SIZE_BYTES / FLOAT_SIZE_BYTES;

  static_assert(FLOAT_SIZE_BYTES == 4, "Unexpected float size!");

  const std::size_t num_iter = num_elements / FLOATS_PER_VECTOR;
  const std::size_t remainder = num_elements - (num_iter * FLOATS_PER_VECTOR);

  const __m512 scale_vector = _mm512_set1_ps(scale);
  // const __m512 round_vector = _mm512_set1_ps(0.5f);
  const __m128i zero_point_vector = _mm_set1_epi8(zero_point);
  for (std::size_t i = 0; i < num_iter; ++i) {
    _mm_prefetch((const char*)src + 64, _MM_HINT_T0);
    __m512 in = _mm512_loadu_ps(src);
    in = _mm512_roundscale_ps(_mm512_div_ps(in, scale_vector),
                              _MM_FROUND_TO_NEAREST_INT);
    __m128i int8s = _mm_add_epi8(_mm512_cvtsepi32_epi8(_mm512_cvtps_epi32(in)),
                                 zero_point_vector);
    _mm_storeu_epi8(dst, int8s);
    // _mm_stream_si128((__m128i*)dst, int8s);
    src += FLOATS_PER_VECTOR;
    dst += FLOATS_PER_VECTOR;
  }

  if (remainder > 0) {
    std::transform(src, src + remainder, dst, [&](const float& src) {
      return rounder<std::int8_t>(src / scale) + zero_point;
    });
  }
}
#endif

// To-do: MNDBG: confirm if const needed
void float2fix(const float* src, std::int8_t* dst, std::size_t num_elements,
               const float scale, const int zero_point) {
#ifdef _WIN32 // TODO _mm_storeu_epi8 is not defined on Linux
  static const bool avxSupport = cpu_support::cpuSupport.avx512f &&
                                 cpu_support::cpuSupport.avx512bw &&
                                 cpu_support::cpuSupport.avx512vl;
  if (avxSupport) {
    if (isAligned((void*)dst, sizeof(__m128i))) {
      float2fix_avx512_stream(src, (__m128i*)dst, num_elements, scale,
                              zero_point);
    } else {
      float2fix_avx512(src, dst, num_elements, scale, zero_point);
    }
  } else {
    std::transform(src, src + num_elements, dst, [&](const float& src) {
      return rounder<std::int8_t>(src / scale) + zero_point;
    });
  }
#else
  std::transform(src, src + num_elements, dst, [&](const float& src) {
    return rounder<std::int8_t>(src / scale) + zero_point;
  });
#endif
}

// float -> uint8
#ifdef _WIN32
inline void float2uint8_avx512_stream(const float* src, __m128i* dst,
                                      std::size_t num_elements,
                                      const float scale,
                                      const uint8_t zero_point) {
  constexpr std::size_t VECTOR_SIZE_BYTES = sizeof(__m512);
  constexpr std::size_t FLOAT_SIZE_BYTES = sizeof(float);
  constexpr std::size_t FLOATS_PER_VECTOR =
      VECTOR_SIZE_BYTES / FLOAT_SIZE_BYTES;

  static_assert(FLOAT_SIZE_BYTES == 4, "Unexpected float size!");

  const std::size_t num_iter = num_elements / FLOATS_PER_VECTOR;
  const std::size_t remainder = num_elements - (num_iter * FLOATS_PER_VECTOR);

  const __m512 scale_vector = _mm512_set1_ps(scale);
  const __m512i zero_point_vector = _mm512_set1_epi32(zero_point);
  __m512i max =
      _mm512_set1_epi32(std::numeric_limits<uint8_t>::max() - zero_point);
  __m512i min =
      _mm512_set1_epi32(std::numeric_limits<uint8_t>::min() - zero_point);
  for (std::size_t i = 0; i < num_iter; ++i) {
    _mm_prefetch((const char*)src + 64, _MM_HINT_T0);
    __m512 in = _mm512_loadu_ps(src);
    __m512i int32s = _mm512_cvtps_epi32(_mm512_roundscale_ps(
        _mm512_div_ps(in, scale_vector), _MM_FROUND_TO_NEAREST_INT));
    int32s = _mm512_min_epi32(int32s, max);
    int32s = _mm512_max_epi32(int32s, min);
    int32s = _mm512_add_epi32(int32s, zero_point_vector);
    // convert int32 to uint8
    __m128i low = _mm512_extracti32x4_epi32(int32s, 0);
    __m128i mid_low = _mm512_extracti32x4_epi32(int32s, 1);
    __m128i mid_high = _mm512_extracti32x4_epi32(int32s, 2);
    __m128i high = _mm512_extracti32x4_epi32(int32s, 3);

    __m128i packed_low = _mm_packus_epi32(low, mid_low);
    __m128i packed_high = _mm_packus_epi32(mid_high, high);
    __m128i packed_uint8 = _mm_packus_epi16(packed_low, packed_high);
    // _mm_storeu_epi8(dst, int8s);
    _mm_stream_si128((__m128i*)dst, packed_uint8);
    src += FLOATS_PER_VECTOR;
    dst += 1;
  }
  _mm_sfence();

  if (remainder > 0) {
    std::transform(
        src, src + remainder, (std::int8_t*)dst, [&](const float& src) {
          return rounder<std::uint8_t>(
                     src / scale,
                     int32_t(std::numeric_limits<std::uint8_t>::min()) -
                         zero_point,
                     int32_t(std::numeric_limits<std::uint8_t>::max()) -
                         zero_point) +
                 zero_point;
        });
  }
}
#endif

#ifdef _WIN32
inline void float2uint8_avx512(const float* src, std::uint8_t* dst,
                               std::size_t num_elements, const float scale,
                               const uint8_t zero_point) {
  constexpr std::size_t VECTOR_SIZE_BYTES = sizeof(__m512);
  constexpr std::size_t FLOAT_SIZE_BYTES = sizeof(float);
  constexpr std::size_t FLOATS_PER_VECTOR =
      VECTOR_SIZE_BYTES / FLOAT_SIZE_BYTES;

  static_assert(FLOAT_SIZE_BYTES == 4, "Unexpected float size!");

  const std::size_t num_iter = num_elements / FLOATS_PER_VECTOR;
  const std::size_t remainder = num_elements - (num_iter * FLOATS_PER_VECTOR);

  const __m512 scale_vector = _mm512_set1_ps(scale);
  // const __m512 round_vector = _mm512_set1_ps(0.5f);
  const __m512i zero_point_vector = _mm512_set1_epi32(zero_point);
  __m512i max =
      _mm512_set1_epi32(std::numeric_limits<uint8_t>::max() - zero_point);
  __m512i min =
      _mm512_set1_epi32(std::numeric_limits<uint8_t>::min() - zero_point);
  for (std::size_t i = 0; i < num_iter; ++i) {
    _mm_prefetch((const char*)src + 64, _MM_HINT_T0);
    __m512 in = _mm512_loadu_ps(src);
    __m512i int32s = _mm512_cvtps_epi32(_mm512_roundscale_ps(
        _mm512_div_ps(in, scale_vector), _MM_FROUND_TO_NEAREST_INT));
    int32s = _mm512_min_epi32(int32s, max);
    int32s = _mm512_max_epi32(int32s, min);
    int32s = _mm512_add_epi32(int32s, zero_point_vector);
    // convert int32 to uint8
    __m128i low = _mm512_extracti32x4_epi32(int32s, 0);
    __m128i mid_low = _mm512_extracti32x4_epi32(int32s, 1);
    __m128i mid_high = _mm512_extracti32x4_epi32(int32s, 2);
    __m128i high = _mm512_extracti32x4_epi32(int32s, 3);

    __m128i packed_low = _mm_packus_epi32(low, mid_low);
    __m128i packed_high = _mm_packus_epi32(mid_high, high);
    __m128i packed_uint8 = _mm_packus_epi16(packed_low, packed_high);

    _mm_storeu_epi8(dst, packed_uint8);
    // _mm_stream_si128((__m128i*)dst, int8s);
    src += FLOATS_PER_VECTOR;
    dst += FLOATS_PER_VECTOR;
  }

  if (remainder > 0) {
    std::transform(src, src + remainder, dst, [&](const float& src) {
      return rounder<std::uint8_t>(
                 src / scale,
                 int32_t(std::numeric_limits<std::uint8_t>::min()) - zero_point,
                 int32_t(std::numeric_limits<std::uint8_t>::max()) -
                     zero_point) +
             zero_point;
    });
  }
}
#endif
// To-do: MNDBG: confirm if const needed
void float2uint8(const float* src, std::uint8_t* dst, std::size_t num_elements,
                 const float scale, const std::uint8_t zero_point) {
#ifdef _WIN32 // TODO _mm_storeu_epi8 is not defined on Linux
  static const bool avxSupport = cpu_support::cpuSupport.avx512f &&
                                 cpu_support::cpuSupport.avx512bw &&
                                 cpu_support::cpuSupport.avx512vl;
  if (avxSupport) {
    if (isAligned((void*)dst, sizeof(__m128i))) {
      float2uint8_avx512_stream(src, (__m128i*)dst, num_elements, scale,
                                zero_point);
    } else {
      float2uint8_avx512(src, dst, num_elements, scale, zero_point);
    }
  } else {
    std::transform(src, src + num_elements, dst, [&](const float& src) {
      return rounder<std::uint8_t>(
                 src / scale,
                 int32_t(std::numeric_limits<std::uint8_t>::min()) - zero_point,
                 int32_t(std::numeric_limits<std::uint8_t>::max()) -
                     zero_point) +
             zero_point;
    });
  }
#else
  std::transform(src, src + num_elements, dst, [&](const float& src) {
    return rounder<std::uint8_t>(
               src / scale,
               int32_t(std::numeric_limits<std::uint8_t>::min()) - zero_point,
               int32_t(std::numeric_limits<std::uint8_t>::max()) - zero_point) +
           zero_point;
  });
#endif
}

// int8 -> float
#ifdef _WIN32
inline void fix2float_avx512_stream256(const std::int8_t* src, float* dst,
                                       std::size_t num_elements,
                                       const float scale,
                                       const int8_t zero_point) {
  constexpr std::size_t VECTOR_SIZE_BYTES = sizeof(__m128);
  constexpr std::size_t INT8_SIZE_BYTES = sizeof(int8_t);
  constexpr std::size_t INT8_PER_VECTOR = VECTOR_SIZE_BYTES / INT8_SIZE_BYTES;

  static_assert(INT8_SIZE_BYTES == 1, "Unexpected int8_t size!");
  const std::size_t num_iter = num_elements / INT8_PER_VECTOR;
  const std::size_t remainder = num_elements - (num_iter * INT8_PER_VECTOR);

  const __m512 scale_vector = _mm512_set1_ps(scale);
  const __m128i zero_point_vector = _mm_set1_epi8(zero_point);
  for (std::size_t i = 0; i < num_iter; ++i) {
    _mm_prefetch((const char*)src + 16, _MM_HINT_T0);
    __m128i in = _mm_loadu_epi8(src);
    __m512 mul = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(
                                   _mm_sub_epi8(in, zero_point_vector))),
                               scale_vector);
    __m256 temp0 = _mm512_extractf32x8_ps(mul, 0);
    __m256 temp1 = _mm512_extractf32x8_ps(mul, 1);
    _mm256_stream_ps(dst, temp0);
    _mm256_stream_ps(dst + 8, temp1);
    // _mm512_storeu_ps(dst, mul);
    src += INT8_PER_VECTOR;
    dst += INT8_PER_VECTOR;
  }
  _mm_sfence();
  if (remainder > 0) {
    std::transform(src, src + remainder, dst, [&](const std::int8_t& src) {
      return (std::int8_t((src - zero_point)) * scale);
    });
  }
}
#endif

#ifdef _WIN32
inline void fix2float_avx512(const std::int8_t* src, float* dst,
                             std::size_t num_elements, const float scale,
                             const int8_t zero_point) {
  constexpr std::size_t VECTOR_SIZE_BYTES = sizeof(__m128);
  constexpr std::size_t INT8_SIZE_BYTES = sizeof(int8_t);
  constexpr std::size_t INT8_PER_VECTOR = VECTOR_SIZE_BYTES / INT8_SIZE_BYTES;

  static_assert(INT8_SIZE_BYTES == 1, "Unexpected int8_t size!");
  const std::size_t num_iter = num_elements / INT8_PER_VECTOR;
  const std::size_t remainder = num_elements - (num_iter * INT8_PER_VECTOR);

  const __m512 scale_vector = _mm512_set1_ps(scale);
  const __m128i zero_point_vector = _mm_set1_epi8(zero_point);
  for (std::size_t i = 0; i < num_iter; ++i) {
    _mm_prefetch((const char*)src + 16, _MM_HINT_T0);
    __m128i in = _mm_loadu_epi8(src);
    __m512 mul = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(
                                   _mm_sub_epi8(in, zero_point_vector))),
                               scale_vector);
    _mm512_storeu_ps(dst, mul);
    src += INT8_PER_VECTOR;
    dst += INT8_PER_VECTOR;
  }

  if (remainder > 0) {
    std::transform(src, src + remainder, dst, [&](const std::int8_t& src) {
      return (std::int8_t((src - zero_point)) * scale);
    });
  }
}
#endif

void fix2float(const std::int8_t* src, float* dst, std::size_t num_elements,
               const float scale, const int8_t zero_point) {
#ifdef _WIN32 // TODO _mm_storeu_epi8 is not defined on Linux
  static const bool avxSupport = cpu_support::cpuSupport.avx512f &&
                                 cpu_support::cpuSupport.avx512bw &&
                                 cpu_support::cpuSupport.avx512vl;
  if (avxSupport) {
    if (isAligned((void*)dst, sizeof(__m256))) {
      fix2float_avx512_stream256(src, dst, num_elements, scale, zero_point);
    } else {
      fix2float_avx512(src, dst, num_elements, scale, zero_point);
    }
  } else {
    std::transform(src, src + num_elements, dst, [&](const std::int8_t& src) {
      return (std::int8_t((src - zero_point)) * scale);
    });
  }
#else
  std::transform(src, src + num_elements, dst, [&](const std::int8_t& src) {
    return (std::int8_t((src - zero_point)) * scale);
  });
#endif
}

// uint8 -> float
#ifdef _WIN32
inline void uint82float_avx512_stream256(const std::uint8_t* src, float* dst,
                                         std::size_t num_elements,
                                         const float scale,
                                         const uint8_t zero_point) {
  constexpr std::size_t VECTOR_SIZE_BYTES = sizeof(__m128);
  constexpr std::size_t UINT8_SIZE_BYTES = sizeof(uint8_t);
  constexpr std::size_t UINT8_PER_VECTOR = VECTOR_SIZE_BYTES / UINT8_SIZE_BYTES;

  static_assert(UINT8_SIZE_BYTES == 1, "Unexpected uint8_t size!");
  const std::size_t num_iter = num_elements / UINT8_PER_VECTOR;
  const std::size_t remainder = num_elements - (num_iter * UINT8_PER_VECTOR);

  const __m512 scale_vector = _mm512_set1_ps(scale);
  const __m512i zero_point_vector = _mm512_set1_epi32(zero_point);
  for (std::size_t i = 0; i < num_iter; ++i) {
    _mm_prefetch((const char*)src + 16, _MM_HINT_T0);
    __m128i in = _mm_loadu_epi8(src);
    __m512i sub_result =
        _mm512_sub_epi32(_mm512_cvtepu8_epi32(in),
                         zero_point_vector); // int r = int(x)- int(zeropoint);
    __m512 mul = _mm512_mul_ps(_mm512_cvtepi32_ps(sub_result), scale_vector);

    __m256 temp0 = _mm512_extractf32x8_ps(mul, 0);
    __m256 temp1 = _mm512_extractf32x8_ps(mul, 1);
    _mm256_stream_ps(dst, temp0);
    _mm256_stream_ps(dst + 8, temp1);
    // _mm512_storeu_ps(dst, mul);
    src += UINT8_PER_VECTOR;
    dst += UINT8_PER_VECTOR;
  }
  _mm_sfence();
  if (remainder > 0) {
    std::transform(src, src + remainder, dst, [&](const std::uint8_t& src) {
      return ((std::int32_t(src) - zero_point) * scale);
    });
  }
}
#endif

#ifdef _WIN32
inline void uint82float_avx512(const std::uint8_t* src, float* dst,
                               std::size_t num_elements, const float scale,
                               const uint8_t zero_point) {
  constexpr std::size_t VECTOR_SIZE_BYTES = sizeof(__m128);
  constexpr std::size_t UINT8_SIZE_BYTES = sizeof(uint8_t);
  constexpr std::size_t UINT8_PER_VECTOR = VECTOR_SIZE_BYTES / UINT8_SIZE_BYTES;

  static_assert(UINT8_SIZE_BYTES == 1, "Unexpected uint8_t size!");
  const std::size_t num_iter = num_elements / UINT8_PER_VECTOR;
  const std::size_t remainder = num_elements - (num_iter * UINT8_PER_VECTOR);

  const __m512 scale_vector = _mm512_set1_ps(scale);
  const __m512i zero_point_vector = _mm512_set1_epi32(zero_point);
  for (std::size_t i = 0; i < num_iter; ++i) {
    _mm_prefetch((const char*)src + 16, _MM_HINT_T0);
    __m128i in = _mm_loadu_epi8(src);
    __m512i sub_result =
        _mm512_sub_epi32(_mm512_cvtepu8_epi32(in),
                         zero_point_vector); // int r = int(x)- int(zeropoint);
    __m512 mul = _mm512_mul_ps(_mm512_cvtepi32_ps(sub_result), scale_vector);
    _mm512_storeu_ps(dst, mul);
    src += UINT8_PER_VECTOR;
    dst += UINT8_PER_VECTOR;
  }

  if (remainder > 0) {
    std::transform(src, src + remainder, dst, [&](const std::uint8_t& src) {
      return ((std::int32_t(src) - zero_point) * scale);
    });
  }
}
#endif

void uint82float(const std::uint8_t* src, float* dst, std::size_t num_elements,
                 const float scale, const uint8_t zero_point) {
#ifdef _WIN32 // TODO _mm_storeu_epi8 is not defined on Linux
  static const bool avxSupport = cpu_support::cpuSupport.avx512f &&
                                 cpu_support::cpuSupport.avx512bw &&
                                 cpu_support::cpuSupport.avx512vl;
  if (avxSupport) {
    if (isAligned((void*)dst, sizeof(__m256))) {
      uint82float_avx512_stream256(src, dst, num_elements, scale, zero_point);
    } else {
      uint82float_avx512(src, dst, num_elements, scale, zero_point);
    }
  } else {
    std::transform(src, src + num_elements, dst, [&](const std::uint8_t& src) {
      return ((std::int32_t(src) - zero_point) * scale);
    });
  }
#else
  std::transform(src, src + num_elements, dst, [&](const std::int8_t& src) {
    return ((std::int32_t(src) - zero_point) * scale);
  });
#endif
}
} // namespace

void QuantizeLinearInt8(OrtKernelContext* context,
                        const Ort::Custom::Tensor<float>& X,
                        const Ort::Custom::Tensor<float>& scale,
                        const Ort::Custom::Tensor<int8_t>& zp,
                        Ort::Custom::Tensor<int8_t>& Y) {
  auto X_raw = X.Data();
  auto Y_raw = Y.Allocate(X.Shape());
  CHECK_EQ(scale.NumberOfElement(), 1) << scale.NumberOfElement();
  auto zp_data = zp.NumberOfElement() == 1 ? *zp.Data() : 0;
  float2fix(X_raw, Y_raw, X.NumberOfElement(), *scale.Data(), zp_data);
}
void QuantizeLinearUint8(OrtKernelContext* context,
                         const Ort::Custom::Tensor<float>& X,
                         const Ort::Custom::Tensor<float>& scale,
                         const Ort::Custom::Tensor<uint8_t>& zp,
                         Ort::Custom::Tensor<uint8_t>& Y) {
  auto X_raw = X.Data();
  auto Y_raw = Y.Allocate(X.Shape());
  CHECK_EQ(scale.NumberOfElement(), 1) << scale.NumberOfElement();
  auto zp_data = zp.NumberOfElement() == 1 ? *zp.Data() : 0;
  float2uint8(X_raw, (uint8_t*)Y_raw, X.NumberOfElement(), *scale.Data(),
              (uint8_t)zp_data);
}
void QuantizeLinearInt16(OrtKernelContext* context,
                         const Ort::Custom::Tensor<float>& X,
                         const Ort::Custom::Tensor<float>& scale,
                         const Ort::Custom::Tensor<int16_t>& zp,
                         Ort::Custom::Tensor<int16_t>& Y) {
  auto X_raw = X.Data();
  auto Y_raw = Y.Allocate(X.Shape());
  CHECK_EQ(scale.NumberOfElement(), 1) << scale.NumberOfElement();
  auto zp_data = zp.NumberOfElement() == 1 ? *zp.Data() : 0;
  float2int16(X_raw, Y_raw, X.NumberOfElement(), *scale.Data(), zp_data);
}
void QuantizeLinearUint16(OrtKernelContext* context,
                          const Ort::Custom::Tensor<float>& X,
                          const Ort::Custom::Tensor<float>& scale,
                          const Ort::Custom::Tensor<uint16_t>& zp,
                          Ort::Custom::Tensor<uint16_t>& Y) {
  auto X_raw = X.Data();
  auto Y_raw = Y.Allocate(X.Shape());
  CHECK_EQ(scale.NumberOfElement(), 1) << scale.NumberOfElement();
  auto zp_data = zp.NumberOfElement() == 1 ? *zp.Data() : 0;
  float2uint16(X_raw, (uint16_t*)Y_raw, X.NumberOfElement(), *scale.Data(),
               (uint16_t)zp_data);
}
void DequantizeLinearInt8(OrtKernelContext* context,
                          const Ort::Custom::Tensor<int8_t>& X,
                          const Ort::Custom::Tensor<float>& scale,
                          const Ort::Custom::Tensor<int8_t>& zp,
                          Ort::Custom::Tensor<float>& Y) {
  const std::vector<int64_t>& shape = X.Shape();
  const int8_t* X_raw = X.Data();
  float* Y_raw = Y.Allocate(shape);
  CHECK_EQ(scale.NumberOfElement(), 1) << scale.NumberOfElement();
  auto zp_data = zp.NumberOfElement() == 1 ? *zp.Data() : 0;
  fix2float(X_raw, Y_raw, X.NumberOfElement(), *scale.Data(), zp_data);
}

void DequantizeLinearUint8(OrtKernelContext* context,
                           const Ort::Custom::Tensor<uint8_t>& X,
                           const Ort::Custom::Tensor<float>& scale,
                           const Ort::Custom::Tensor<uint8_t>& zp,
                           Ort::Custom::Tensor<float>& Y) {
  const std::vector<int64_t>& shape = X.Shape();
  const uint8_t* X_raw = X.Data();
  float* Y_raw = Y.Allocate(shape);
  CHECK_EQ(scale.NumberOfElement(), 1) << scale.NumberOfElement();
  uint8_t zp_data = zp.NumberOfElement() == 1 ? *(const uint8_t*)zp.Data() : 0;
  uint82float((const uint8_t*)X_raw, Y_raw, X.NumberOfElement(), *scale.Data(),
              zp_data);
}
void DequantizeLinearInt16(OrtKernelContext* context,
                           const Ort::Custom::Tensor<int16_t>& X,
                           const Ort::Custom::Tensor<float>& scale,
                           const Ort::Custom::Tensor<int16_t>& zp,
                           Ort::Custom::Tensor<float>& Y) {
  const std::vector<int64_t>& shape = X.Shape();
  auto X_raw = X.Data();
  float* Y_raw = Y.Allocate(shape);
  CHECK_EQ(scale.NumberOfElement(), 1) << scale.NumberOfElement();
  auto zp_data = zp.NumberOfElement() == 1 ? *zp.Data() : 0;
  int162float(X_raw, Y_raw, X.NumberOfElement(), *scale.Data(), zp_data);
}
void DequantizeLinearUint16(OrtKernelContext* context,
                            const Ort::Custom::Tensor<uint16_t>& X,
                            const Ort::Custom::Tensor<float>& scale,
                            const Ort::Custom::Tensor<uint16_t>& zp,
                            Ort::Custom::Tensor<float>& Y) {
  const std::vector<int64_t>& shape = X.Shape();
  auto X_raw = X.Data();
  float* Y_raw = Y.Allocate(shape);
  CHECK_EQ(scale.NumberOfElement(), 1) << scale.NumberOfElement();
  uint16_t zp_data =
      zp.NumberOfElement() == 1 ? *(const uint16_t*)zp.Data() : 0;
  uint162float((const uint16_t*)X_raw, Y_raw, X.NumberOfElement(),
               *scale.Data(), zp_data);
}
struct OpdefQDQ {
  static void process(std::vector<Ort::CustomOpDomain>& domains) {
    constexpr const char* kOnnxDomain = "";
    constexpr const char* kMSDomain = "com.microsoft";
    Ort::CustomOpDomain domain(kOnnxDomain);
    // DequantizeLinear-10 13
    domain.Add(Ort::Custom::CreateLiteCustomOp(
        "DequantizeLinear", "VitisAIExecutionProvider", DequantizeLinearUint8,
        {}, 10, 13));
    domain.Add(Ort::Custom::CreateLiteCustomOp(
        "DequantizeLinear", "VitisAIExecutionProvider", DequantizeLinearInt8,
        {}, 10, 13));
    // todo int32

    // T1 : tensor(int8), tensor(uint8), tensor(int32), tensor(float8e4m3fn),
    // tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz) T2 :
    // tensor(float), tensor(float16), tensor(bfloat16)
    domain.Add(Ort::Custom::CreateLiteCustomOp(
        "DequantizeLinear", "VitisAIExecutionProvider", DequantizeLinearInt8,
        {}, 19, 21));
    domain.Add(Ort::Custom::CreateLiteCustomOp(
        "DequantizeLinear", "VitisAIExecutionProvider", DequantizeLinearUint8,
        {}, 19, 21));
    domain.Add(Ort::Custom::CreateLiteCustomOp(
        "DequantizeLinear", "VitisAIExecutionProvider", DequantizeLinearInt16,
        {}, 21, 21));
    domain.Add(Ort::Custom::CreateLiteCustomOp(
        "DequantizeLinear", "VitisAIExecutionProvider", DequantizeLinearUint16,
        {}, 21, 21));

    domain.Add(Ort::Custom::CreateLiteCustomOp(
        "QuantizeLinear", "VitisAIExecutionProvider", QuantizeLinearUint8, {},
        10, 13));
    domain.Add(Ort::Custom::CreateLiteCustomOp("QuantizeLinear",
                                               "VitisAIExecutionProvider",
                                               QuantizeLinearInt8, {}, 10, 13));
    domain.Add(Ort::Custom::CreateLiteCustomOp(
        "QuantizeLinear", "VitisAIExecutionProvider", QuantizeLinearUint8, {},
        19, 21));
    domain.Add(Ort::Custom::CreateLiteCustomOp("QuantizeLinear",
                                               "VitisAIExecutionProvider",
                                               QuantizeLinearInt8, {}, 19, 21));
    domain.Add(Ort::Custom::CreateLiteCustomOp(
        "QuantizeLinear", "VitisAIExecutionProvider", QuantizeLinearUint16, {},
        21, 21));
    domain.Add(Ort::Custom::CreateLiteCustomOp(
        "QuantizeLinear", "VitisAIExecutionProvider", QuantizeLinearInt16, {},
        21, 21));
    domains.push_back(std::move(domain));
    domain = Ort::CustomOpDomain(kMSDomain);
    // DequantizeLinear-10 13
    domain.Add(Ort::Custom::CreateLiteCustomOp("DequantizeLinear",
                                               "VitisAIExecutionProvider",
                                               DequantizeLinearUint8, {}, 1));
    domain.Add(Ort::Custom::CreateLiteCustomOp("DequantizeLinear",
                                               "VitisAIExecutionProvider",
                                               DequantizeLinearInt8, {}, 1));
    domain.Add(Ort::Custom::CreateLiteCustomOp("DequantizeLinear",
                                               "VitisAIExecutionProvider",
                                               DequantizeLinearInt16, {}, 1));
    domain.Add(Ort::Custom::CreateLiteCustomOp("DequantizeLinear",
                                               "VitisAIExecutionProvider",
                                               DequantizeLinearUint16, {}, 1));

    domain.Add(Ort::Custom::CreateLiteCustomOp("QuantizeLinear",
                                               "VitisAIExecutionProvider",
                                               QuantizeLinearUint8, {}, 1));
    domain.Add(Ort::Custom::CreateLiteCustomOp("QuantizeLinear",
                                               "VitisAIExecutionProvider",
                                               QuantizeLinearInt8, {}, 1));
    domain.Add(Ort::Custom::CreateLiteCustomOp("QuantizeLinear",
                                               "VitisAIExecutionProvider",
                                               QuantizeLinearUint16, {}, 1));
    domain.Add(Ort::Custom::CreateLiteCustomOp("QuantizeLinear",
                                               "VitisAIExecutionProvider",
                                               QuantizeLinearInt16, {}, 1));

    domains.push_back(std::move(domain));
  }
};

// vaip_op_def_qdq__hook needs to be written to symbols.txt
DEFINE_VAIP_OPDEF(OpdefQDQ, vaip_op_def_qdq)
