/*
 *     The Xilinx Vitis AI Vaip in this distribution are provided under the
 * following free and permissive binary-only license, but are not provided in
 * source code form.  While the following free and permissive license is similar
 * to the BSD open source license, it is NOT the BSD open source license nor
 * other OSI-approved open source license.
 *
 *      Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc.
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
  float2fix(X_raw, (int8_t*)Y_raw, X.NumberOfElement(), *scale.Data(),
            (int8_t)zp_data);
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
  float2int16(X_raw, (int16_t*)Y_raw, X.NumberOfElement(), *scale.Data(),
              (int16_t)zp_data);
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
  int8_t zp_data = zp.NumberOfElement() == 1 ? *(const int8_t*)zp.Data() : 0;
  fix2float((const int8_t*)X_raw, Y_raw, X.NumberOfElement(), *scale.Data(),
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
  int16_t zp_data = zp.NumberOfElement() == 1 ? *(const int16_t*)zp.Data() : 0;
  int162float((const int16_t*)X_raw, Y_raw, X.NumberOfElement(), *scale.Data(),
              zp_data);
}
struct OpdefQDQ {
  static void process(std::vector<Ort::CustomOpDomain>& domains) {
    Ort::CustomOpDomain domain("");
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
  }
};

// vaip_op_def_qdq__hook needs to be written to symbols.txt
DEFINE_VAIP_OPDEF(OpdefQDQ, vaip_op_def_qdq)
