/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

#include "onnxruntime_api.hpp"

#include <glog/logging.h>
#if !defined(_WIN32)
#  include <immintrin.h>
#  include <mmintrin.h>
#  include <x86intrin.h>
#  include <xmmintrin.h>
#endif
#include <iostream>
#include <sstream>
#include <string>
//
#include "./custom_op.hpp"
#include "./reporter.hpp"
#include "vitis/ai/profiling.hpp"

DEF_ENV_PARAM(DRY_RUN, "0")
#ifdef _WIN32
DEF_ENV_PARAM(USE_ASYNC_WAIT, "1")
#else
DEF_ENV_PARAM(USE_ASYNC_WAIT, "0")
#endif

#define MLADF_VERSION "v1"
#define MAX_SEQ_LENGTH 3072

static float bfloat16_to_float(uint16_t x) {
  float y = 0.0;
  uint8_t* src = (uint8_t*)&x;
  uint8_t* dst = (uint8_t*)&y;
  std::memcpy(&dst[2], src, sizeof(uint16_t));
  return y;
}

static uint16_t float_to_bfloat16(float x) {
  uint32_t i;
  uint8_t* src = (uint8_t*)&x;
  uint8_t* tmp = (uint8_t*)&i;
  // copy float to uint32_t
  std::memcpy(tmp, src, sizeof(float));
  // round to nearest even
  uint32_t lsb = (i >> 16) & 0x1;
  uint32_t bias = 0x7fff + lsb;
  i += bias;
  // extract upper half of input
  uint16_t y = uint16_t(i >> 16);
  return y;
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
    out[i] = float_to_bfloat16(v[i]);
  }
  _mm_sfence();
}

static float bfloat16_to_float_single(uint16_t v) {
  union {
    uint32_t i;
    float f;
  } u;
  u.i = (uint32_t(v)) << 16;
  return u.f;
}
#if defined(_WIN32)
static void bfloat16_to_float_avx512_unrolled(uint16_t* s, float* d, int n) {
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
__m512 bf16_to_fp32(__m256i bf16_data) {
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
namespace vaip_mlp_custom_op {

template <typename T>
static void writeToFile(std::string filename, const T* data, size_t size) {

  std::ofstream file(filename, std::ios::binary);

  if (!file) {
    throw std::ios_base::failure("Failed to open file");
  }
  file.write((char*)(data), size * sizeof(T));

  if (!file) {
    throw std::ios_base::failure("Failed to write data to file");
  }
  file.close();
}

template <typename dtype>
std::vector<dtype> MyCustomOp::loadbin(std::string& fname) const {
  std::ifstream ifs(fname, std::ifstream::binary);
  std::vector<dtype> bindata;
  if (ifs) {
    ifs.seekg(0, ifs.end);
    auto size = ifs.tellg();
    ifs.seekg(0, ifs.beg);
    // Create space
    bindata.reserve(size / sizeof(dtype));
    // Read to buffer
    ifs.read((char*)(bindata.data()), size);
    return bindata;
  }
  return bindata;
}

MyCustomOp::MyCustomOp(std::shared_ptr<const PassContext> context,
                       const std::shared_ptr<MetaDefProto>& meta_def,
                       onnxruntime::Model* model)
    : CustomOpImp(context, meta_def, model) {
  cnt_ = stoi(meta_def->generic_param().at("cnt"));
  // Extracting the attribute information
  // Gate proj
  gp_k = stoi(meta_def->generic_param().at("gp_K"));
  gp_n = stoi(meta_def->generic_param().at("gp_N"));
  gp_bits = stoi(meta_def->generic_param().at("gp_bits"));
  gp_block_size = stoi(meta_def->generic_param().at("gp_block_size"));
  // Up proj
  up_k = stoi(meta_def->generic_param().at("up_K"));
  up_n = stoi(meta_def->generic_param().at("up_N"));
  up_bits = stoi(meta_def->generic_param().at("up_bits"));
  up_block_size = stoi(meta_def->generic_param().at("up_block_size"));
  // Down proj
  dp_k = stoi(meta_def->generic_param().at("dp_K"));
  dp_n = stoi(meta_def->generic_param().at("dp_N"));
  dp_bits = stoi(meta_def->generic_param().at("dp_bits"));
  dp_block_size = stoi(meta_def->generic_param().at("dp_block_size"));

  // Input size for token phase
#ifdef _WIN32
  input_data_ = (uint16_t*)_aligned_malloc(gp_k * sizeof(uint16_t), 64);
#else
  input_data_ = (uint16_t*)aligned_alloc(64, gp_k * sizeof(uint16_t));
#endif
  if (input_data_ == nullptr) {
    throw std::runtime_error("Unable to create memory for ryzenai-customop");
  }

  // Reading the constant weights and scales file
  // Gate proj
  std::string gp_wts_f = meta_def->generic_param().at("gp_wts_file");
  std::string gp_scl_f = meta_def->generic_param().at("gp_scl_file");
  std::string gp_zps_f;
  if (meta_def->generic_param().contains("gp_zps_file")) {
    gp_zps_f = meta_def->generic_param().at("gp_zps_file");
  }
  // Get weights / scales / zps
  auto gp_wts = loadbin<int8_t>(gp_wts_f);
  auto gp_scl = loadbin<float>(gp_scl_f);
  std::vector<int8_t> gp_zps;
  if (!gp_zps_f.empty())
    gp_zps = loadbin<int8_t>(gp_zps_f);

  // Up proj
  std::string up_wts_f = meta_def->generic_param().at("up_wts_file");
  std::string up_scl_f = meta_def->generic_param().at("up_scl_file");
  std::string up_zps_f;
  if (meta_def->generic_param().contains("up_zps_file")) {
    up_zps_f = meta_def->generic_param().at("up_zps_file");
  }
  // Get weights / scales / zps
  auto up_wts = loadbin<int8_t>(up_wts_f);
  auto up_scl = loadbin<float>(up_scl_f);
  std::vector<int8_t> up_zps;
  if (!up_zps_f.empty())
    up_zps = loadbin<int8_t>(up_zps_f);

  // Down proj
  std::string dp_wts_f = meta_def->generic_param().at("dp_wts_file");
  std::string dp_scl_f = meta_def->generic_param().at("dp_scl_file");
  std::string dp_zps_f;
  if (meta_def->generic_param().contains("dp_zps_file")) {
    dp_zps_f = meta_def->generic_param().at("dp_zps_file");
  }
  // Get weights / scales / zps
  auto dp_wts = loadbin<int8_t>(dp_wts_f);
  auto dp_scl = loadbin<float>(dp_scl_f);
  std::vector<int8_t> dp_zps;
  if (!dp_zps_f.empty())
    dp_zps = loadbin<int8_t>(dp_zps_f);

  dry_run_ = 0;
  if (ENV_PARAM(DRY_RUN) == 1)
    dry_run_ = 1;
  /////////////////////////// Gate /////////////////////////////////
  size_t gp_kblks = gp_k / gp_block_size;
  int64_t gzp_shape =
      (gp_n * std::floor((float)((gp_kblks + 1) * gp_bits) / 8.0f));

  std::vector<float> gp_bias(gp_n, 0); // fill with zeros
  std::vector<float> gp_scales(gp_k * gp_n / gp_block_size);
  std::vector<int8_t> gp_weights(gp_k * gp_n, 0);
  // fill this with zeros for Symmetric quantization
  // std::vector<int8_t> gp_zpoints(gp_k * gp_n / gp_block_size, 0);
  std::vector<int8_t> gp_zpoints(gzp_shape * 2, 0);

  // Original weights are in NxK/2 packed as uint8
  // Convert to KXN uint8
  for (int64_t i = 0; i < gp_k; i += 2) {
    for (int64_t j = 0; j < gp_n; j++) {
      auto srcv = gp_wts[j * gp_k / 2 + i / 2];
      auto src0 = (srcv & 0xf) - 8;
      auto src1 = ((srcv & 0xf0) >> 4) - 8;
      gp_weights[i * gp_n + j] = static_cast<int8_t>(src0);
      gp_weights[(i + 1) * gp_n + j] = static_cast<int8_t>(src1);
    }
  }

  // Original Scales are in Nx(K/BlockSize) shape
  // Convert to (K/BLOCK_SIZE)xN shape
  for (int i = 0; i < gp_n; i++) {
    for (int j = 0; j < gp_kblks; j++) {
      gp_scales[j * gp_n + i] = gp_scl[i * gp_kblks + j];
    }
  }

  // fill this with zeros for Symmetric quantization
  if (!gp_zps_f.empty()) {
    int gp_kblks_pad = 2 * gzp_shape / gp_n;
    for (int i = 0; i < gp_n; i++) {
      for (int j = 0; j < gp_kblks_pad; j = j + 2) {
        // auto zpv = gp_zps[(i * (gp_kblks / 2)) + (j / 2)];
        auto zpv = gp_zps[((i * gp_kblks_pad) / 2 + j / 2)];
        gp_zpoints[j * gp_n + i] = (zpv & 0xf) - 8;
        gp_zpoints[(j + 1) * gp_n + i] = ((zpv & 0xf0) >> 4) - 8;
      }
    }
  }

  /////////////////////////// Up /////////////////////////////////
  size_t up_kblks = up_k / up_block_size;
  int64_t uzp_shape =
      (up_n * std::floor((float)((up_kblks + 1) * up_bits) / 8.0f));

  std::vector<float> up_bias(up_n, 0); // fill with zeros
  std::vector<float> up_scales(up_k * up_n / up_block_size);
  std::vector<int8_t> up_weights(up_k * up_n, 0);
  // fill this with zeros for Symmetric quantization
  // std::vector<int8_t> up_zpoints(up_k * up_n / up_block_size, 0);
  std::vector<int8_t> up_zpoints(uzp_shape * 2, 0);

  // Original weights are in NxK/2 packed as uint8
  // Convert to KXN uint8
  for (int64_t i = 0; i < up_k; i += 2) {
    for (int64_t j = 0; j < up_n; j++) {
      auto srcv = up_wts[j * up_k / 2 + i / 2];
      auto src0 = (srcv & 0xf) - 8;
      auto src1 = ((srcv & 0xf0) >> 4) - 8;
      up_weights[i * up_n + j] = static_cast<int8_t>(src0);
      up_weights[(i + 1) * up_n + j] = static_cast<int8_t>(src1);
    }
  }

  // Original Scales are in Nx(K/BlockSize) shape
  // Convert to (K/BLOCK_SIZE)xN shape
  for (int i = 0; i < up_n; i++) {
    for (int j = 0; j < up_kblks; j++) {
      up_scales[j * up_n + i] = up_scl[i * up_kblks + j];
    }
  }

  // fill this with zeros for Symmetric quantization
  if (!up_zps_f.empty()) {
    int up_kblks_pad = 2 * uzp_shape / up_n;
    for (int i = 0; i < up_n; i++) {
      for (int j = 0; j < up_kblks_pad; j = j + 2) {
        // auto zpv = up_zps[(i * (up_kblks / 2)) + (j / 2)];
        auto zpv = up_zps[((i * up_kblks_pad) / 2 + j / 2)];
        up_zpoints[j * up_n + i] = (zpv & 0xf) - 8;
        up_zpoints[(j + 1) * up_n + i] = ((zpv & 0xf0) >> 4) - 8;
      }
    }
  }

  /////////////////////////// Down /////////////////////////////////
  size_t dp_kblks = dp_k / dp_block_size;
  int64_t zp_shape =
      (dp_n * std::floor((float)((dp_kblks + 1) * dp_bits) / 8.0f));
  std::vector<float> dp_bias(dp_n, 0); // fill with zeros
  std::vector<float> dp_scales(dp_k * dp_n / dp_block_size);
  std::vector<int8_t> dp_weights(dp_k * dp_n, 0);
  // fill this with zeros for Symmetric quantization
  // std::vector<int8_t> dp_zpoints(dp_k * dp_n / dp_block_size, 0);
  std::vector<int8_t> dp_zpoints(zp_shape * 2, 0);

  // Original weights are in NxK/2 packed as uint8
  // Convert to KXN uint8
  for (int64_t i = 0; i < dp_k; i += 2) {
    for (int64_t j = 0; j < dp_n; j++) {
      auto srcv = dp_wts[j * dp_k / 2 + i / 2];
      auto src0 = (srcv & 0xf) - 8;
      auto src1 = ((srcv & 0xf0) >> 4) - 8;
      dp_weights[i * dp_n + j] = static_cast<int8_t>(src0);
      dp_weights[(i + 1) * dp_n + j] = static_cast<int8_t>(src1);
    }
  }

  // Original Scales are in Nx(K/BlockSize) shape
  // Convert to (K/BLOCK_SIZE)xN shape
  for (int i = 0; i < dp_n; i++) {
    for (int j = 0; j < dp_kblks; j++) {
      dp_scales[j * dp_n + i] = dp_scl[i * dp_kblks + j];
    }
  }

  // fill this with zeros for Symmetric quantization
  // Each row of zero points was padded to have an even length "dp_kblks_pad"
  if (!dp_zps_f.empty()) {
    int dp_kblks_pad = 2 * zp_shape / dp_n;
    for (int i = 0; i < dp_n; i++) {
      for (int j = 0; j < dp_kblks_pad; j = j + 2) {
        // auto zpv = dp_zps[(i * (dp_kblks / 2)) + (j / 2)];
        auto zpv = dp_zps[((i * dp_kblks_pad) / 2 + j / 2)];
        dp_zpoints[j * dp_n + i] = (zpv & 0xf) - 8;
        dp_zpoints[(j + 1) * dp_n + i] = ((zpv & 0xf0) >> 4) - 8;
      }
    }
  }

  // Create mladfmatmulbias operator handle
  std::string mladf_version_(MLADF_VERSION);

  std::map<std::string, std::any> attrs;
  attrs["op_version"] = mladf_version_;

  //////////////// Init Gate //////////////////
  // weights
  std::tuple<int, int> gp_wts_shape = {static_cast<int>(gp_k),
                                       static_cast<int>(gp_n)};
  std::vector<size_t> gp_wts_shape_dd = {static_cast<size_t>(gp_k),
                                         static_cast<size_t>(gp_n)};

  if (cnt_ == 0) {
    // Create qlinear-2 handle
    gate_proj_ = std::make_shared<
        ryzenai::mladfmatmulbias<uint16_t, int8_t, uint16_t, uint16_t>>(
        "bfloat16", "int4", "bfloat16", true, attrs);
  }

  auto gp_ptr =
      (ryzenai::mladfmatmulbias<uint16_t, int8_t, uint16_t, uint16_t>*)
          gate_proj_.get();

  Tensor gp_wts_tensor = {gp_weights.data(), gp_wts_shape_dd, "int4"};
  Tensor gp_scl_tensor = {
      gp_scales.data(), {(size_t)gp_block_size, 1}, "float"};
  Tensor gp_zps_tensor = {gp_zpoints.data(), gp_wts_shape_dd, "int4"};
  Tensor gp_bias_tensor = {gp_bias.data(), {(size_t)gp_block_size, 1}, "float"};

  std::vector<Tensor> gp_const_tensors = {gp_wts_tensor, gp_bias_tensor,
                                          gp_scl_tensor, gp_zps_tensor};

  std::map<std::string, std::any> gp_attrs;
  gp_attrs["default_shape"] = 1;
  gp_attrs["op_version"] = mladf_version_;
  gp_attrs["group_size"] = gp_block_size;
  gp_attrs["max_m"] = MAX_SEQ_LENGTH;
  gp_ptr->initialize_const_params(gp_const_tensors, gp_attrs);

  //////////////// Init Up //////////////////
  // weights
  std::tuple<int, int> up_wts_shape = {static_cast<int>(up_k),
                                       static_cast<int>(up_n)};
  std::vector<size_t> up_wts_shape_dd = {static_cast<size_t>(up_k),
                                         static_cast<size_t>(up_n)};

  if (cnt_ == 0) { // Create qlinear-2 handle
    up_proj_ = std::make_shared<
        ryzenai::mladfmatmulbias<uint16_t, int8_t, uint16_t, uint16_t>>(
        "bfloat16", "int4", "bfloat16", true, attrs);
  }

  auto up_ptr =
      (ryzenai::mladfmatmulbias<uint16_t, int8_t, uint16_t, uint16_t>*)
          up_proj_.get();

  Tensor up_wts_tensor = {up_weights.data(), up_wts_shape_dd, "int4"};
  Tensor up_scl_tensor = {
      up_scales.data(), {(size_t)up_block_size, 1}, "float"};
  Tensor up_zps_tensor = {up_zpoints.data(), up_wts_shape_dd, "int4"};
  Tensor up_bias_tensor = {up_bias.data(), {(size_t)up_block_size, 1}, "float"};

  std::vector<Tensor> up_const_tensors = {up_wts_tensor, up_bias_tensor,
                                          up_scl_tensor, up_zps_tensor};

  std::map<std::string, std::any> up_attrs;
  up_attrs["default_shape"] = 1;
  up_attrs["op_version"] = mladf_version_;
  up_attrs["group_size"] = up_block_size;
  up_attrs["max_m"] = MAX_SEQ_LENGTH;
  up_ptr->initialize_const_params(up_const_tensors, up_attrs);

  //////////////// Init Down //////////////////
  // weights
  std::tuple<int, int> dp_wts_shape = {static_cast<int>(dp_k),
                                       static_cast<int>(dp_n)};
  std::vector<size_t> dp_wts_shape_dd = {static_cast<size_t>(dp_k),
                                         static_cast<size_t>(dp_n)};

  if (cnt_ == 0) {
    // Create qlinear-2 handle
    down_proj_ = std::make_shared<
        ryzenai::mladfmatmulbias<uint16_t, int8_t, uint16_t, uint16_t>>(
        "bfloat16", "int4", "bfloat16", true, attrs);
  }

  auto dp_ptr =
      (ryzenai::mladfmatmulbias<uint16_t, int8_t, uint16_t, uint16_t>*)
          down_proj_.get();

  Tensor dp_wts_tensor = {dp_weights.data(), dp_wts_shape_dd, "int4"};
  Tensor dp_scl_tensor = {
      dp_scales.data(), {(size_t)dp_block_size, 1}, "float"};
  Tensor dp_zps_tensor = {dp_zpoints.data(), dp_wts_shape_dd, "int4"};
  Tensor dp_bias_tensor = {dp_bias.data(), {(size_t)dp_block_size, 1}, "float"};

  std::vector<Tensor> dp_const_tensors = {dp_wts_tensor, dp_bias_tensor,
                                          dp_scl_tensor, dp_zps_tensor};

  std::map<std::string, std::any> dp_attrs;
  dp_attrs["default_shape"] = 1;
  dp_attrs["op_version"] = mladf_version_;
  dp_attrs["group_size"] = dp_block_size;
  dp_attrs["max_m"] = MAX_SEQ_LENGTH;
  dp_ptr->initialize_const_params(dp_const_tensors, dp_attrs);

  if (cnt_ == 0) {
    std::vector<int> size_mul_silu_M{1, 128, 256, 512, 1024, 2048, 3072};
    std::vector<std::vector<int>> shape_list_mul_silu;
    for (auto M : size_mul_silu_M) {
      shape_list_mul_silu.push_back({M, (int)up_n});
    }
    std::map<std::string, std::any> attr_mul = {{"skip_create_input", 1},
                                                {"op_version", mladf_version_}};
    // ElwMul
    ewmul_ = std::make_shared<ryzenai::elw_mul<uint16_t, uint16_t, uint16_t>>(
        "bfloat16", true, attr_mul);
    std::map<std::string, std::any> attr_silu = {
        {"skip_create_input", 1},
        {"op_version", mladf_version_},
        {"shapes", shape_list_mul_silu}};
    // Silu
    silu_ = std::make_shared<ryzenai::silu<uint16_t, uint16_t>>(
        "bfloat16", true, attr_silu);
  }
}

MyCustomOp::~MyCustomOp() {}

void MyCustomOp::Compute(const OrtApi* api, OrtKernelContext* context) const {
  if (Ort::Global<void>::api_ == nullptr) {
    Ort::Global<void>::api_ = api;
  }

  Ort::KernelContext ctx(context);
  auto num_inputs = ctx.GetInputCount();
  auto num_outputs = ctx.GetOutputCount();

  __TIC__(MLP_PRE)

  // Extracting the input and output information
  auto input_tensor = ctx.GetInput(0); // Input activations
  // auto input_data = input_tensor.GetTensorData<float>();
  auto input_data = input_tensor.GetTensorData<uint16_t>(); // bfloat16 inputs
  auto input_shape = input_tensor.GetTensorTypeAndShapeInfo().GetShape();

  // std::cout << "run compute" << n_sizes_.size() << std::endl;
  std::vector<int64_t> out_shape;
  for (unsigned i = 0; i < (input_shape.size() - 1); i++)
    out_shape.push_back(input_shape[i]);
  out_shape.push_back(dp_n);

  std::string shape_in = std::to_string(input_shape[0]) + "_" +
                         std::to_string(input_shape[1]) + "_" +
                         std::to_string(input_shape[2]);

  std::string shape_in_1 = std::to_string(input_shape[0]) + "_" +
                           std::to_string(input_shape[1]) + "_" +
                           std::to_string(gp_n);

  // Create output tensor
  auto output_tensor = ctx.GetOutput(
      0, {out_shape.begin(), out_shape.end()}); // Output activation
  // auto outdata = output_tensor.GetTensorMutableData<float>();
  auto outdata =
      output_tensor.GetTensorMutableData<uint16_t>(); // bfloat16 outputs

  // Gate projection
  auto gp_ = (ryzenai::mladfmatmulbias<uint16_t, int8_t, uint16_t, uint16_t>*)
                 gate_proj_.get();
  // Up projection
  auto up_ = (ryzenai::mladfmatmulbias<uint16_t, int8_t, uint16_t, uint16_t>*)
                 up_proj_.get();
  // Down projection
  auto dp_ = (ryzenai::mladfmatmulbias<uint16_t, int8_t, uint16_t, uint16_t>*)
                 down_proj_.get();

  // ElwMul
  auto ewmul = (ryzenai::elw_mul<uint16_t, uint16_t, uint16_t>*)ewmul_.get();

  // Silu
  auto silu = (ryzenai::silu<uint16_t, uint16_t>*)silu_.get();

  // Ryzen-AI implementation
  int gp_M = input_shape[0] * input_shape[1];
  std::vector<size_t> gp_a_shape = {static_cast<size_t>(gp_M),
                                    static_cast<size_t>(input_shape[2])};

  std::vector<size_t> gp_c_shape = {static_cast<size_t>(gp_M),
                                    static_cast<size_t>(gp_n)};

  std::vector<size_t> gp_wts_shape = {static_cast<size_t>(gp_k),
                                      static_cast<size_t>(gp_n)};

  std::vector<size_t> up_wts_shape = {static_cast<size_t>(up_k),
                                      static_cast<size_t>(up_n)};

  std::vector<size_t> dp_wts_shape = {static_cast<size_t>(dp_k),
                                      static_cast<size_t>(dp_n)};

  std::vector<size_t> dp_a_shape = {static_cast<size_t>(gp_M),
                                    static_cast<size_t>(dp_k)};

  std::vector<size_t> dp_c_shape = {static_cast<size_t>(gp_M),
                                    static_cast<size_t>(dp_n)};

  // Input / Output XRT BOs
  auto gate_inputs = gp_->get_inputs(gp_M);
  auto gate_outputs = gp_->get_outputs(gp_M);
  auto up_outputs = up_->get_outputs(gp_M);
  auto dp_outputs = dp_->get_outputs(gp_M);
  // Set input/output/weights shape
  gp_->set_shape(gp_a_shape, gp_wts_shape, gp_block_size);
  up_->set_shape(gp_a_shape, up_wts_shape, up_block_size);
  dp_->set_shape(dp_a_shape, dp_wts_shape, dp_block_size);
  // Get constants
  auto gate_const = gp_->get_const();
  auto up_const = up_->get_const();
  auto down_const = dp_->get_const();

  bool wait = true; // by default, wait in execute call
  // do not wait in execute, i.e. async wait is enabled
  if (ENV_PARAM(USE_ASYNC_WAIT) == 1)
    wait = false;

  // Input BO
  uint16_t* in_map = gate_inputs[0].map<uint16_t*>();

  __TOC__(MLP_PRE)

  // Copy input data to BO
  __TIC__(MLP_INPUT_COPY)
  memcpy((void*)in_map, (void*)input_data, gp_M * gp_k * sizeof(uint16_t));
  __TOC__(MLP_INPUT_COPY)
  __TIC__(MLP_RUN)

  // Sync BO
  gate_inputs[0].sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // Exec
  // Gate proj
  std::vector<xrt::bo> gate_in = {gate_inputs[0], gate_const[cnt_]};

  TRY_EXECUTE_WITH_LOG(gp_->execute(gate_in, gate_outputs, wait), dry_run_,
                       ReportInventory::getInstance().addData,
                       "mladfmatmulbias_gp_" + std::to_string(input_shape[1]),
                       shape_in);

  // Up proj
  std::vector<xrt::bo> up_in = {gate_inputs[0], up_const[cnt_]};
  TRY_EXECUTE_WITH_LOG(up_->execute(up_in, up_outputs, wait), dry_run_,
                       ReportInventory::getInstance().addData,
                       "mladfmatmulbias_up_" + std::to_string(input_shape[1]),
                       shape_in);

  // Silu
  auto silu_output = silu->get_outputs();
  // TODO: Fixed M size for Silu for prefill phase
  std::vector<size_t> a_shape_silu = {static_cast<size_t>(gp_M),
                                      static_cast<size_t>(gp_n)};

  silu->set_kernel_shape(a_shape_silu);
  TRY_EXECUTE_WITH_LOG(silu->execute(gate_outputs, silu_output, wait), dry_run_,
                       ReportInventory::getInstance().addData,
                       "mladfmatmulbias_silu_" + std::to_string(input_shape[1]),
                       shape_in_1);

  // Emul
  ewmul->set_kernel_shape(a_shape_silu);
  std::vector<xrt::bo> ewmul_inputs = {silu_output[0], up_outputs[0]};
  auto ewmul_outputs = ewmul->get_outputs();
  TRY_EXECUTE_WITH_LOG(ewmul->execute(ewmul_inputs, ewmul_outputs, wait),
                       dry_run_, ReportInventory::getInstance().addData,
                       "mladfmatmulbias_emul_" + std::to_string(input_shape[1]),
                       shape_in_1);

  // Down Proj
  std::vector<xrt::bo> dp_inputs = {ewmul_outputs[0], down_const[cnt_]};
  TRY_EXECUTE_WITH_LOG(dp_->execute(dp_inputs, dp_outputs), dry_run_,
                       ReportInventory::getInstance().addData,
                       "mladfmatmulbias_dp_" + std::to_string(input_shape[1]),
                       shape_in_1);

  dp_outputs[0].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  __TOC__(MLP_RUN)

  __TIC__(MLP_OUTPUT_COPY)

  // Copy output data to host
  uint16_t* output_data_ = dp_outputs[0].map<uint16_t*>();
  memcpy(outdata, output_data_, gp_M * dp_n * sizeof(uint16_t));

  __TOC__(MLP_OUTPUT_COPY)
}
} // namespace vaip_mlp_custom_op
