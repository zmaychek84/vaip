/*
 *     The Xilinx Vitis AI Vaip in this distribution are provided under the
 * following free and permissive binary-only license, but are not provided in
 * source code form.  While the following free and permissive license is similar
 * to the BSD open source license, it is NOT the BSD open source license nor
 * other OSI-approved open source license.
 *
 *      Copyright (C) 2022 Xilinx, Inc. All rights reserved.
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

#if defined(_WIN32)
#  include <intrin.h>
#else
#  include "ryzenai/dynamic_dispatch/utils/instruction_cache.hpp"
#  include "ryzenai/dynamic_dispatch/utils/instruction_registry.hpp"
#  include <ryzenai/dynamic_dispatch/xrt_context/xrt_context.hpp>
#  include <x86intrin.h>
#endif
#include <mmintrin.h>
#include <xmmintrin.h>

#include <immintrin.h>

#include <cmath>
#include <iostream>
#include <memory>
#include <sstream>
#include <thread>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include "custom_op_ssmlp.hpp"
#include "reporter.hpp"
#include "vitis/ai/profiling.hpp"
#include <fstream>
#include <glog/logging.h>

DEF_ENV_PARAM(DRY_RUN, "0")
#ifdef _WIN32
DEF_ENV_PARAM(USE_ASYNC_WAIT, "1")
#else
DEF_ENV_PARAM(USE_ASYNC_WAIT, "0")
#endif
DEF_ENV_PARAM(DEBUG_SSMLP_CUSTOM_OP, "0")
DEF_ENV_PARAM(USE_AIE_SSMLP, "1")
DEF_ENV_PARAM_2(MLADF_VERSION, "v1", std::string)
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_SSMLP_CUSTOM_OP) >= n)

#define MAX_SEQ_LENGTH 3072

namespace ort_ssmlp_custom_op {

int cnt_ops = 0;
int MyCustomOpKernel::instances__ = 0;
std::shared_ptr<void> MyCustomOpKernel::ewmul_ = nullptr;
std::shared_ptr<void> MyCustomOpKernel::silu_ = nullptr;
std::shared_ptr<void> MyCustomOpKernel::gate_proj_ = nullptr;
std::shared_ptr<void> MyCustomOpKernel::up_proj_ = nullptr;
std::shared_ptr<void> MyCustomOpKernel::down_proj_ = nullptr;
float* MyCustomOpKernel::input_a = nullptr;
float* MyCustomOpKernel::input_b = nullptr;
float* MyCustomOpKernel::output_1 = nullptr;
float* MyCustomOpKernel::output_2 = nullptr;

// Custom Op Domain
std::once_flag MyCustomOpKernel::initFlag;
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
static float bfloat_to_float(uint16_t x) {
  float i = 0;
  uint8_t* src = (uint8_t*)&x;
  uint8_t* tmp = (uint8_t*)&i;
  // copy uint16_t to float (msb)
  std::memcpy(tmp + 2, src, sizeof(uint16_t));
  return i;
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
void bfloat16_to_float_avx512_unrolled(const uint16_t* s, float* d, int n) {
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

void bfloat16_to_float_avx512_unrolled(const uint16_t* s, float* d, int n) {
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

void MyCustomOpKernel::LazyInit() {
  dry_run_ = 0;
  if (ENV_PARAM(DRY_RUN) == 1)
    dry_run_ = 1;
  std::string mladf_version_("v1");

  if (mladf_version_ != "v1") {
    std::cerr << "Invalid version. Supported versions are v1" << std::endl;
  }
  std::map<std::string, std::any> attr = {{"op_version", mladf_version_}};
  // SSLRN1
  static ryzenai::rms_norm<uint16_t, uint16_t, uint16_t> rms_norm =
      ryzenai::rms_norm<uint16_t, uint16_t, uint16_t>("bfloat16", true, attr);
  static ryzenai::mladf_add<uint16_t, uint16_t, uint16_t> add =
      ryzenai::mladf_add<uint16_t, uint16_t, uint16_t>("bfloat16", true, attr);
  if (rms_norm_ == nullptr) {
    rms_norm_ = &rms_norm;
  }
  if (add_ == nullptr) {
    add_ = &add;
  }
  std::vector<Tensor> const_Tensor;
  rms_norm_->initialize_const_params(const_Tensor);
  // SSLRN2
  static ryzenai::rms_norm<uint16_t, uint16_t, uint16_t> rms_norm2 =
      ryzenai::rms_norm<uint16_t, uint16_t, uint16_t>("bfloat16", true, attr);
  static ryzenai::mladf_add<uint16_t, uint16_t, uint16_t> add2 =
      ryzenai::mladf_add<uint16_t, uint16_t, uint16_t>("bfloat16", true, attr);
  if (rms_norm2_ == nullptr) {
    rms_norm2_ = &rms_norm2;
  }

  if (add2_ == nullptr) {
    add2_ = &add2;
  }
  std::vector<Tensor> const_Tensor2;
  rms_norm2_->initialize_const_params(const_Tensor2);
}

MyCustomOpKernel::MyCustomOpKernel(const OrtKernelInfo* k_info) {
  std::string node_name;
  // Get constant info for the node
  Ort::ConstKernelInfo info{k_info};

  // Get Logger
  m_logger = info.GetLogger();

  // onnx built in sslrn op
  MY_LOG(2) << "initialization for onnx SkipSimplifiedLayerNormalization "
               "builtin op..."
            << std::endl;

  const char* add_type_constraint_names[2] = {"T"};
  ONNXTensorElementDataType add_type_constraint_values[1] = {
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT};
  // Create attributes
  epsilon_ = info.GetAttribute<float>("epsilon");
  float epsilon = epsilon_;
  auto attr_epsilon =
      Ort::OpAttr("epsilon", &epsilon, 1, OrtOpAttrType::ORT_OP_ATTR_FLOAT);
  Ort::OpAttr k_attrs[1] = {std::move(attr_epsilon)};
  // Create OP to call the appropriate built in operator
  // In this use case, we are using an ONNX Contrib OP from Microsoft
  op_k = Ort::Op::Create(info.Copy(), "SkipSimplifiedLayerNormalization",
                         "com.microsoft", 1, add_type_constraint_names,
                         add_type_constraint_values, 1, k_attrs, 1, 3,
                         4); // 1 attributes, 3 inputs, 2 outputs

  if (ENV_PARAM(USE_AIE_SSMLP) == 1) {
    // kernel objects for sslrn 1 and 2
    LazyInit();
  }

  int is_constant = 0;
  m_weights = info.GetTensorConstantInput(2, &is_constant);
  const float* wts_data = m_weights.GetTensorData<float>();

  auto dimensions_wts = m_weights.GetTensorTypeAndShapeInfo().GetShape();

  num_el = m_weights.GetTensorTypeAndShapeInfo().GetElementCount();

#ifdef _WIN32
  wts_ = (uint16_t*)_aligned_malloc(num_el * sizeof(uint16_t), 64);
#else
  wts_ = (uint16_t*)aligned_alloc(64, num_el * sizeof(uint16_t));
#endif
  // Convert floating point to bfloat16 using avx512
  float_to_bfloat16_avx512_unrolled(wts_data, wts_,
                                    num_el); // K
  is_constant = 0;
  m2_weights = info.GetTensorConstantInput(12, &is_constant);
  const float* wts2_data = m2_weights.GetTensorData<float>();

  auto dimensions_wts2 = m2_weights.GetTensorTypeAndShapeInfo().GetShape();

  num_el2 = m2_weights.GetTensorTypeAndShapeInfo().GetElementCount();

#ifdef _WIN32
  wts2_ = (uint16_t*)_aligned_malloc(num_el2 * sizeof(uint16_t), 64);
#else
  wts2_ = (uint16_t*)aligned_alloc(64, num_el2 * sizeof(uint16_t));
#endif
  // Convert floating point to bfloat16 using avx512
  float_to_bfloat16_avx512_unrolled(wts2_data, wts2_,
                                    num_el2); // K

  if (instances__ == 0) // allocate memory for SSLRN cpu ios only once
  {
    size_t num_elements = 1 * dimensions_wts[0]; // M*4096, M=1 for token phase
#ifdef _WIN32
    input_a = (float*)_aligned_malloc(num_elements * sizeof(float), 64);
    input_b = (float*)_aligned_malloc(num_elements * sizeof(float), 64);
    output_1 = (float*)_aligned_malloc(num_elements * sizeof(float), 64);
    output_2 = (float*)_aligned_malloc(num_elements * sizeof(float), 64);
#else
    input_a = (float*)aligned_alloc(64, num_elements * sizeof(float));
    input_b = (float*)aligned_alloc(64, num_elements * sizeof(float));
    output_1 = (float*)aligned_alloc(64, num_elements * sizeof(float));
    output_2 = (float*)aligned_alloc(64, num_elements * sizeof(float));
#endif
  }

  MY_LOG(2) << "initialization for SSLRN done." << std::endl;

  // MLP
  //  Extracting the attribute information
  //  Gate proj
  MY_LOG(2) << "initialization for MLP begin" << std::endl;
  gp_k = info.GetAttribute<int64_t>("gate_K");
  gp_n = info.GetAttribute<int64_t>("gate_N");
  gp_bits = info.GetAttribute<int64_t>("gate_bits");
  gp_block_size = info.GetAttribute<int64_t>("gate_block_size");
  // Up proj
  up_k = info.GetAttribute<int64_t>("up_K");
  up_n = info.GetAttribute<int64_t>("up_N");
  up_bits = info.GetAttribute<int64_t>("up_bits");
  up_block_size = info.GetAttribute<int64_t>("up_block_size");
  // Down proj
  dp_k = info.GetAttribute<int64_t>("down_K");
  dp_n = info.GetAttribute<int64_t>("down_N");
  dp_bits = info.GetAttribute<int64_t>("down_bits");
  dp_block_size = info.GetAttribute<int64_t>("down_block_size");

#ifdef _WIN32
  /////// gate proj //////
  is_constant = 0;
  Ort::ConstValue& gp_weights_tensor =
      info.GetTensorConstantInput(3, &is_constant);
  ;
  const int8_t* gp_wts = (gp_weights_tensor.GetTensorData<int8_t>());

  is_constant = 0;
  Ort::ConstValue& gp_scales_tensor =
      info.GetTensorConstantInput(4, &is_constant);
  const float* gp_scl = gp_scales_tensor.GetTensorData<float>();

  int is_gpz_constant = 0;
  Ort::ConstValue& gp_zeros = info.GetTensorConstantInput(5, &is_gpz_constant);
  const int8_t* gp_zps = gp_zeros.GetTensorData<int8_t>();

  /////// up proj //////
  is_constant = 0;
  Ort::ConstValue& up_weights_tensor =
      info.GetTensorConstantInput(6, &is_constant);
  const int8_t* up_wts = (up_weights_tensor.GetTensorData<int8_t>());
  is_constant = 0;
  Ort::ConstValue& up_scales_tensor =
      info.GetTensorConstantInput(7, &is_constant);
  const float* up_scl = up_scales_tensor.GetTensorData<float>();
  int is_upz_constant = 0;
  Ort::ConstValue& up_zeros = info.GetTensorConstantInput(8, &is_upz_constant);
  const int8_t* up_zps = up_zeros.GetTensorData<int8_t>();

  /////// down proj //////
  is_constant = 0;
  Ort::ConstValue& dp_weights_tensor =
      info.GetTensorConstantInput(9, &is_constant);
  const int8_t* dp_wts = (dp_weights_tensor.GetTensorData<int8_t>());
  is_constant = 0;
  Ort::ConstValue& dp_scales_tensor =
      info.GetTensorConstantInput(10, &is_constant);
  const float* dp_scl = dp_scales_tensor.GetTensorData<float>();
  int is_dpz_constant = 0;
  Ort::ConstValue& dp_zeros = info.GetTensorConstantInput(11, &is_dpz_constant);
  const int8_t* dp_zps = dp_zeros.GetTensorData<int8_t>();

  MY_LOG(2) << "Got attributes for MLP" << std::endl;
#else
  is_constant = 0;
  Ort::ConstValue temp_3 = info.GetTensorConstantInput(3, &is_constant);
  Ort::ConstValue& gp_weights_tensor = temp_3;
  const int8_t* gp_wts = (gp_weights_tensor.GetTensorData<int8_t>());

  is_constant = 0;
  Ort::ConstValue temp_4 = info.GetTensorConstantInput(4, &is_constant);

  Ort::ConstValue& gp_scales_tensor = temp_4;
  const float* gp_scl = gp_scales_tensor.GetTensorData<float>();

  int is_gpz_constant = 0;
  Ort::ConstValue temp_5 = info.GetTensorConstantInput(5, &is_gpz_constant);
  Ort::ConstValue& gp_zeros = temp_5;
  const int8_t* gp_zps = gp_zeros.GetTensorData<int8_t>();

  is_constant = 0;
  Ort::ConstValue temp_6 = info.GetTensorConstantInput(6, &is_constant);
  Ort::ConstValue& up_weights_tensor = temp_6;

  const int8_t* up_wts = (up_weights_tensor.GetTensorData<int8_t>());
  is_constant = 0;
  Ort::ConstValue temp_7 = info.GetTensorConstantInput(7, &is_constant);
  Ort::ConstValue& up_scales_tensor = temp_7;

  const float* up_scl = up_scales_tensor.GetTensorData<float>();
  int is_upz_constant = 0;
  Ort::ConstValue temp_8 = info.GetTensorConstantInput(8, &is_upz_constant);
  Ort::ConstValue& up_zeros = temp_8;
  const int8_t* up_zps = up_zeros.GetTensorData<int8_t>();

  is_constant = 0;
  Ort::ConstValue temp_9 = info.GetTensorConstantInput(9, &is_constant);
  Ort::ConstValue& dp_weights_tensor = temp_9;

  const int8_t* dp_wts = (dp_weights_tensor.GetTensorData<int8_t>());
  is_constant = 0;
  Ort::ConstValue temp_10 = info.GetTensorConstantInput(10, &is_constant);
  Ort::ConstValue& dp_scales_tensor = temp_10;

  const float* dp_scl = dp_scales_tensor.GetTensorData<float>();
  int is_dpz_constant = 0;
  Ort::ConstValue temp_11 = info.GetTensorConstantInput(11, &is_dpz_constant);
  Ort::ConstValue& dp_zeros = temp_11;
  const int8_t* dp_zps = dp_zeros.GetTensorData<int8_t>();
  MY_LOG(2) << "Got attributes for MLP" << std::endl;
#endif

  /////////////////////////// Gate /////////////////////////////////
  std::vector<float> gp_bias(gp_n, 0); // fill with zeros
  std::vector<float> gp_scales(gp_k * gp_n / gp_block_size);
  std::vector<int8_t> gp_weights(gp_k * gp_n, 0);
  // fill this with zeros for Symmetric quantization
  std::vector<int8_t> gp_zpoints(
      gp_zeros.GetTensorTypeAndShapeInfo().GetElementCount() * 2, 0);

  size_t gp_kblks = gp_k / gp_block_size;

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
  int64_t gzp_shape =
      (gp_n * std::floor((float)((gp_kblks + 1) * gp_bits) / 8.0f));
  // fill this with zeros for Symmetric quantization
  if (is_gpz_constant) {
    int gp_kblks_pad = 2 * gzp_shape / gp_n;
    for (int i = 0; i < gp_n; i++) {
      for (int j = 0; j < gp_kblks_pad; j = j + 2) {
        auto zpv = gp_zps[((i * gp_kblks_pad) / 2) + (j / 2)];
        gp_zpoints[j * gp_n + i] = (zpv & 0xf) - 8;
        gp_zpoints[(j + 1) * gp_n + i] = ((zpv & 0xf0) >> 4) - 8;
      }
    }
  }

  /////////////////////////// Up /////////////////////////////////
  std::vector<float> up_bias(up_n, 0); // fill with zeros
  std::vector<float> up_scales(up_k * up_n / up_block_size);
  std::vector<int8_t> up_weights(up_k * up_n, 0);
  // fill this with zeros for Symmetric quantization
  std::vector<int8_t> up_zpoints(
      up_zeros.GetTensorTypeAndShapeInfo().GetElementCount() * 2, 0);

  size_t up_kblks = up_k / up_block_size;

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
  int64_t uzp_shape =
      (up_n * std::floor((float)((up_kblks + 1) * up_bits) / 8.0f));
  // fill this with zeros for Symmetric quantization
  if (is_upz_constant) {
    int up_kblks_pad = 2 * uzp_shape / up_n;
    for (int i = 0; i < up_n; i++) {
      for (int j = 0; j < up_kblks_pad; j = j + 2) {
        auto zpv = up_zps[((i * up_kblks_pad) / 2) + (j / 2)];
        up_zpoints[j * up_n + i] = (zpv & 0xf) - 8;
        up_zpoints[(j + 1) * up_n + i] = ((zpv & 0xf0) >> 4) - 8;
      }
    }
  }

  /////////////////////////// Down /////////////////////////////////
  std::vector<float> dp_bias(dp_n, 0); // fill with zeros
  std::vector<float> dp_scales(dp_k * dp_n / dp_block_size);
  std::vector<int8_t> dp_weights(dp_k * dp_n, 0);
  // fill this with zeros for Symmetric quantization
  std::vector<int8_t> dp_zpoints(
      dp_zeros.GetTensorTypeAndShapeInfo().GetElementCount() * 2, 0);

  size_t dp_kblks = dp_k / dp_block_size;
  int64_t zp_shape =
      (dp_n * std::floor((float)((dp_kblks + 1) * dp_bits) / 8.0f));

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
  if (is_dpz_constant) {
    int dp_kblks_pad = 2 * zp_shape / dp_n;
    for (int i = 0; i < dp_n; i++) {
      for (int j = 0; j < dp_kblks_pad; j = j + 2) {
        // auto zpv = dp_zps[((i * dp_kblks) / 2) + (j / 2)];
        auto zpv = dp_zps[((i * dp_kblks_pad) / 2 + j / 2)];
        dp_zpoints[j * dp_n + i] = (zpv & 0xf) - 8;
        dp_zpoints[(j + 1) * dp_n + i] = ((zpv & 0xf0) >> 4) - 8;
      }
    }
  }

  std::string mladf_version_("v1");

  std::map<std::string, std::any> attrs;
  attrs["op_version"] = mladf_version_;

  //////////////// Init Gate //////////////////
  // weights
  std::tuple<int, int> gp_wts_shape = {static_cast<int>(gp_k),
                                       static_cast<int>(gp_n)};
  std::vector<size_t> gp_wts_shape_dd = {static_cast<size_t>(gp_k),
                                         static_cast<size_t>(gp_n)};

  if (instances__ == 0) {
    // Create qlinear-2 handle
    gate_proj_ = std::make_shared<
        ryzenai::mladfmatmulbias<uint16_t, int8_t, uint16_t, uint16_t>>(
        "bfloat16", "int4", "bfloat16", true, attrs);
  }

  MY_LOG(2) << "- Init gate projection MatMul: created handle";
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

  if (instances__ == 0) {
    // Create qlinear-2 handle
    up_proj_ = std::make_shared<
        ryzenai::mladfmatmulbias<uint16_t, int8_t, uint16_t, uint16_t>>(
        "bfloat16", "int4", "bfloat16", true, attrs);
  }

  MY_LOG(2) << "- Init up projection MatMul: created handle";
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

  if (instances__ == 0) {
    // Create qlinear-2 handle
    down_proj_ = std::make_shared<
        ryzenai::mladfmatmulbias<uint16_t, int8_t, uint16_t, uint16_t>>(
        "bfloat16", "int4", "bfloat16", true, attrs);
  }

  MY_LOG(2) << "- Init down projection MatMul: created handle";
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

  if (instances__ == 0) {
    std::vector<int> size_mul_silu_M{1, 128, 256, 512, 1024, 2048, 3072};
    std::vector<std::vector<int>> shape_list_mul_silu;
    for (auto M : size_mul_silu_M) {
      shape_list_mul_silu.push_back({M, (int)up_n});
    }
    std::map<std::string, std::any> attr_mul = {
        {"skip_create_input", 1},
        {"op_version", mladf_version_},
        {"shapes", shape_list_mul_silu}};
    // ElwMul
    ewmul_ = std::make_shared<ryzenai::elw_mul<uint16_t, uint16_t, uint16_t>>(
        "bfloat16", true, attr_mul);

    // Silu
    silu_ = std::make_shared<ryzenai::silu<uint16_t, uint16_t>>("bfloat16",
                                                                true, attr_mul);
  }

  cnt_ = instances__++;
  MY_LOG(2) << "SSMLP- Init MLP done";
}

MyCustomOpKernel::~MyCustomOpKernel() {
#ifdef _WIN32
  if (wts_) {
    _aligned_free(wts_);
    wts_ = nullptr;
  }
  if (wts2_) {
    _aligned_free(wts2_);
    wts2_ = nullptr;
  }
  if (input_a) {
    _aligned_free(input_a);
    input_a = nullptr;
  }
  if (input_b) {
    _aligned_free(input_b);
    input_b = nullptr;
  }
  if (output_1) {
    _aligned_free(output_1);
    output_1 = nullptr;
  }
  if (output_2) {
    _aligned_free(output_2);
    output_2 = nullptr;
  }
#else
  if (wts_)
    free(wts_);
  wts_ = nullptr;
  if (wts2_)
    free(wts2_);
  wts2_ = nullptr;
  if (input_a)
    free(input_a);
  input_a = nullptr;
  if (input_b)
    free(input_b);
  input_b = nullptr;

  if (output_1)
    free(output_1);
  output_1 = nullptr;
  if (output_2)
    free(output_2);
  output_2 = nullptr;
  ewmul_.reset();
  silu_.reset();
  gate_proj_.reset();
  up_proj_.reset();
  down_proj_.reset();
  ryzenai::dynamic_dispatch::xrt_context::destroy_ctx_map();

#endif
}

void MyCustomOpKernel::Compute(OrtKernelContext* context) {
  // to dump to file
  MY_LOG(2) << "\n\n- AMD SSLRN compute start ...\n";

  Ort::KernelContext ctx(context);
  auto num_inputs = ctx.GetInputCount();
  auto num_outputs = ctx.GetOutputCount();
  MY_LOG(2) << "num_inputs " << num_inputs << " "
            << "num_outputs " << num_outputs << " ";

  auto input = ctx.GetInput(0); // Input
  auto skip = ctx.GetInput(1);  // skip

  auto dimensions_input = input.GetTensorTypeAndShapeInfo().GetShape();
  auto dimensions_skip = skip.GetTensorTypeAndShapeInfo().GetShape();

  size_t B = dimensions_input[0]; // Batch
  size_t M = dimensions_input[1]; // Seq len
  size_t K = dimensions_input[2]; // Hidden size = Num_heads * Head_size

  auto in_data = input.GetTensorData<uint16_t>();
  auto skip_data = skip.GetTensorData<uint16_t>();
  std::vector<size_t> a_shape = {M, K};

  // sslrn1 aie kernel bos
  std::vector<xrt::bo> rms_norm1_outputs_;
  std::vector<xrt::bo> add1_outputs_;

  bool supported_shapes = false;
  if ((std::find(supported_lengths.begin(), supported_lengths.end(), M) !=
       supported_lengths.end()) &&
      ENV_PARAM(USE_AIE_SSMLP) == 1)
    supported_shapes = true;

  // unsupported shapes cpu float pointers
  float* un_input_a = nullptr;
  float* un_input_b = nullptr;
  float* un_output_1 = nullptr;
  float* un_output_2 = nullptr;
  // for cpu
  sslrn_cpu_out = false;
#ifdef _WIN32
  uint16_t* sslrn_out_data_token1 =
      (uint16_t*)_aligned_malloc(M * K * sizeof(uint16_t), 64);
#else
  uint16_t* sslrn_out_data_token1 =
      (uint16_t*)aligned_alloc(64, M * K * sizeof(uint16_t));
#endif

  bool wait = true; // by default, wait in execute call
  // do not wait in execute, i.e. async wait is enabled
  if (ENV_PARAM(USE_ASYNC_WAIT) == 1)
    wait = false;

  if ((std::find(supported_lengths.begin(), supported_lengths.end(), M) !=
       supported_lengths.end()) &&
      ENV_PARAM(USE_AIE_SSMLP) == 1) {

    auto add_inputs = add_->get_inputs();
    add1_outputs_ = add_->get_outputs();
    auto rms_input_0 = rms_norm_->get_inputs()[0];
    auto rms_norm_wt_inputs = rms_norm_->get_inputs()[1];
    rms_norm1_outputs_ = rms_norm_->get_outputs();

    add_->set_kernel_shape(a_shape);
    rms_norm_->set_kernel_shape(a_shape);

    const auto add_operand_size_in_bytes =
        a_shape[0] * a_shape[1] * sizeof(uint16_t);

    // Input BO
    uint16_t* add_input_0_map = add_inputs[0].map<uint16_t*>();
    memcpy((void*)add_input_0_map, (void*)in_data, add_operand_size_in_bytes);
    add_inputs[0].sync(XCL_BO_SYNC_BO_TO_DEVICE);

    uint16_t* add_input_1_map = add_inputs[1].map<uint16_t*>();
    memcpy((void*)add_input_1_map, (void*)skip_data, add_operand_size_in_bytes);
    add_inputs[1].sync(XCL_BO_SYNC_BO_TO_DEVICE);

    TRY_EXECUTE_WITH_LOG(add_->execute(add_inputs, add1_outputs_, wait),
                         dry_run_, ReportInventory::getInstance().addData,
                         "mladf_add_" + std::to_string(M),
                         std::to_string(B) + "_" + std::to_string(M) + "_" +
                             std::to_string(K));

    uint16_t* rms_wt_map = rms_norm_wt_inputs.map<uint16_t*>();
    memcpy((void*)rms_wt_map, (void*)wts_, num_el * sizeof(uint16_t));
    rms_norm_wt_inputs.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    std::vector<xrt::bo> rms_in = {add1_outputs_[0], rms_norm_wt_inputs};

    // Execute RMS Norm

    TRY_EXECUTE_WITH_LOG(
        rms_norm_->execute(rms_in, rms_norm1_outputs_, wait), dry_run_,
        ReportInventory::getInstance().addData, "rms_norm_" + std::to_string(M),
        std::to_string(B) + "_" + std::to_string(M) + "_" + std::to_string(K));

    MY_LOG(2) << "- SSLRN 1 AIE done ...\n";

  } else {
    MY_LOG(2) << "- AMD SSLRN CPU ...\n";
    sslrn_cpu_out =
        true; // set flag to notify MLP and next SSLRN to memcopy and sync
              // outputs of SSLRN 1 with input of MLP and SSLRN2

    size_t num_elements = B * M * K;
    // Define input shape
    std::vector<int64_t> input_shape = {(int64_t)B, (int64_t)M, (int64_t)K};

    if (M != 1) // M is an usupported shape
    {
#ifdef _WIN32
      un_input_a = (float*)_aligned_malloc(num_elements * sizeof(float), 64);
      un_input_b = (float*)_aligned_malloc(num_elements * sizeof(float), 64);
      un_output_1 = (float*)_aligned_malloc(num_elements * sizeof(float), 64);
      un_output_2 = (float*)_aligned_malloc(num_elements * sizeof(float), 64);
#else
      un_input_a = (float*)aligned_alloc(64, num_elements * sizeof(float));
      un_input_b = (float*)aligned_alloc(64, num_elements * sizeof(float));
      un_output_1 = (float*)aligned_alloc(64, num_elements * sizeof(float));
      un_output_2 = (float*)aligned_alloc(64, num_elements * sizeof(float));
#endif

      bfloat16_to_float_avx512_unrolled(in_data, un_input_a,
                                        num_elements); // M x K
      bfloat16_to_float_avx512_unrolled(skip_data, un_input_b, num_elements);
    } else {
      if (input_a == nullptr) {
#ifdef _WIN32
        input_a = (float*)_aligned_malloc(num_elements * sizeof(float), 64);
#else
        input_a = (float*)aligned_alloc(64, num_elements * sizeof(float));
#endif
      }
      if (input_b == nullptr) {
#ifdef _WIN32
        input_b = (float*)_aligned_malloc(num_elements * sizeof(float), 64);
#else
        input_b = (float*)aligned_alloc(64, num_elements * sizeof(float));
#endif
      }
      if (output_1 == nullptr) {
#ifdef _WIN32
        output_1 = (float*)_aligned_malloc(num_elements * sizeof(float), 64);
#else
        output_1 = (float*)aligned_alloc(64, num_elements * sizeof(float));
#endif
      }
      if (output_2 == nullptr) {
#ifdef _WIN32
        output_2 = (float*)_aligned_malloc(num_elements * sizeof(float), 64);
#else
        output_2 = (float*)aligned_alloc(64, num_elements * sizeof(float));
#endif
      }

      bfloat16_to_float_avx512_unrolled(in_data, input_a,
                                        num_elements); // M x K
      bfloat16_to_float_avx512_unrolled(skip_data, input_b, num_elements);
    }

    // Create memory info
    Ort::MemoryInfo memory_info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // Create input tensor

    if (M != 1) {
      Ort::Value input_tensor_a = Ort::Value::CreateTensor<float>(
          memory_info, un_input_a, num_elements, input_shape.data(),
          input_shape.size());
      Ort::Value input_tensor_b = Ort::Value::CreateTensor<float>(
          memory_info, un_input_b, num_elements, input_shape.data(),
          input_shape.size());
      Ort::Value output_tensor_1 = Ort::Value::CreateTensor<float>(
          memory_info, un_output_1, num_elements, input_shape.data(),
          input_shape.size());
      Ort::Value output_tensor_2 = Ort::Value::CreateTensor<float>(
          memory_info, un_output_2, num_elements, input_shape.data(),
          input_shape.size());
      const OrtValue* inputs[3] = {input_tensor_a, input_tensor_b, m_weights};

      OrtValue* outputs[4] = {output_tensor_1, nullptr, nullptr,
                              output_tensor_2};

      op_k.Invoke(context, inputs, 3, outputs, 4);
    } else {
      Ort::Value input_tensor_a = Ort::Value::CreateTensor<float>(
          memory_info, input_a, num_elements, input_shape.data(),
          input_shape.size());
      Ort::Value input_tensor_b = Ort::Value::CreateTensor<float>(
          memory_info, input_b, num_elements, input_shape.data(),
          input_shape.size());
      Ort::Value output_tensor_1 = Ort::Value::CreateTensor<float>(
          memory_info, output_1, num_elements, input_shape.data(),
          input_shape.size());
      Ort::Value output_tensor_2 = Ort::Value::CreateTensor<float>(
          memory_info, output_2, num_elements, input_shape.data(),
          input_shape.size());
      const OrtValue* inputs[3] = {input_tensor_a, input_tensor_b, m_weights};

      OrtValue* outputs[4] = {output_tensor_1, nullptr, nullptr,
                              output_tensor_2};

      op_k.Invoke(context, inputs, 3, outputs, 4);
    }

    if (M != 1) {
      float_to_bfloat16_avx512_unrolled(un_output_1, sslrn_out_data_token1,
                                        M * K); // M x K
    } else
      float_to_bfloat16_avx512_unrolled(output_1, sslrn_out_data_token1,
                                        M * K); // M x K
  }

  MY_LOG(2) << "- AMD SSLRN1 compute done ...\n";
  ///////////////////////////////////////////////////////////
  MY_LOG(2) << "- MLP compute start ...";

  auto input_shape = dimensions_input;

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

  auto gate_inputs = gp_->get_inputs(gp_M);
  auto gate_outputs = gp_->get_outputs(gp_M);
  auto up_outputs = up_->get_outputs(gp_M);
  auto dp_outputs = dp_->get_outputs(gp_M);

  gp_->set_shape(gp_a_shape, gp_wts_shape, gp_block_size);
  up_->set_shape(gp_a_shape, up_wts_shape, up_block_size);
  dp_->set_shape(dp_a_shape, dp_wts_shape, dp_block_size);

  auto gate_const = gp_->get_const();
  auto up_const = up_->get_const();
  auto down_const = dp_->get_const();

  if (gp_M > 1) {
    MY_LOG(2) << "gp_M>1...  PROMPT PHASE, cnt=" << cnt_ << std::endl;

    std::vector<xrt::bo> gate_in;
    std::vector<xrt::bo> up_in;

    if (sslrn_cpu_out) { // M>1 but unsupported shapes for sslrn
      // Input BO
      uint16_t* in_map = gate_inputs[0].map<uint16_t*>();
      memcpy((void*)in_map, (void*)sslrn_out_data_token1,
             gp_M * input_shape[2] * sizeof(uint16_t));
      gate_inputs[0].sync(XCL_BO_SYNC_BO_TO_DEVICE);
      gate_in = {gate_inputs[0], gate_const[cnt_]};
      up_in = {gate_inputs[0], up_const[cnt_]};
    } else {
      gate_in = {rms_norm1_outputs_[0], gate_const[cnt_]};
      up_in = {rms_norm1_outputs_[0], up_const[cnt_]};
    }

    TRY_EXECUTE_WITH_LOG(gp_->execute(gate_in, gate_outputs, wait), dry_run_,
                         ReportInventory::getInstance().addData,
                         "mladfmatmulbias_gp_" + std::to_string(gp_M),
                         std::to_string(input_shape[0]) + "_" +
                             std::to_string(gp_M) + "_" +
                             std::to_string(input_shape[2]));

    MY_LOG(2) << "exec gp...";
    // Up proj

    TRY_EXECUTE_WITH_LOG(up_->execute(up_in, up_outputs, wait), dry_run_,
                         ReportInventory::getInstance().addData,
                         "mladfmatmulbias_up_" + std::to_string(gp_M),
                         std::to_string(input_shape[0]) + "_" +
                             std::to_string(gp_M) + "_" +
                             std::to_string(input_shape[2]));

    // Silu
    MY_LOG(2) << "exec up...";
    auto silu_output = silu->get_outputs();
    // TODO: Fixed M size for Silu for prefill phase
    std::vector<size_t> a_shape_silu = {static_cast<size_t>(gp_M),
                                        static_cast<size_t>(gp_n)};

    silu->set_kernel_shape(a_shape_silu);

    TRY_EXECUTE_WITH_LOG(silu->execute(gate_outputs, silu_output, wait),
                         dry_run_, ReportInventory::getInstance().addData,
                         "mladfmatmulbias_silu_" + std::to_string(gp_M),
                         std::to_string(input_shape[0]) + "_" +
                             std::to_string(gp_M) + "_" + std::to_string(gp_n));

    MY_LOG(2) << "execute silu...";
    ewmul->set_kernel_shape(a_shape_silu);
    std::vector<xrt::bo> ewmul_inputs = {silu_output[0], up_outputs[0]};
    auto ewmul_outputs = ewmul->get_outputs();

    TRY_EXECUTE_WITH_LOG(ewmul->execute(ewmul_inputs, ewmul_outputs, wait),
                         dry_run_, ReportInventory::getInstance().addData,
                         "mladfmatmulbias_emul_" + std::to_string(gp_M),
                         std::to_string(input_shape[0]) + "_" +
                             std::to_string(gp_M) + "_" + std::to_string(gp_n));

    MY_LOG(2) << "exec mul...";
    std::vector<xrt::bo> dp_inputs = {ewmul_outputs[0], down_const[cnt_]};

    bool dp_wait = true; // by default, dp execute will wait
    if (supported_shapes == true &&
        ENV_PARAM(USE_ASYNC_WAIT) ==
            1) // async wait is enabled, and M is a supported shape, so dp
               // output will go to SSLRN NPU => dp execute will not wait
      dp_wait = false;

    TRY_EXECUTE_WITH_LOG(dp_->execute(dp_inputs, dp_outputs, dp_wait), dry_run_,
                         ReportInventory::getInstance().addData,
                         "mladfmatmulbias_dp_" + std::to_string(gp_M),
                         std::to_string(input_shape[0]) + "_" +
                             std::to_string(gp_M) + "_" + std::to_string(gp_n));

  } else {
    MY_LOG(2) << "gp_M==1..."; // token phase
    std::vector<xrt::bo> gate_in;
    std::vector<xrt::bo> up_in;

    if (sslrn_cpu_out) {
      // Input BO
      uint16_t* in_map = gate_inputs[0].map<uint16_t*>();
      memcpy((void*)in_map, (void*)sslrn_out_data_token1,
             gp_k * sizeof(uint16_t));
      gate_inputs[0].sync(XCL_BO_SYNC_BO_TO_DEVICE);
      gate_in = {gate_inputs[0], gate_const[cnt_]};
      up_in = {gate_inputs[0], up_const[cnt_]};
    } else {
      gate_in = {rms_norm1_outputs_[0], gate_const[cnt_]};
      up_in = {rms_norm1_outputs_[0], up_const[cnt_]};
    }

    // Exec
    // Gate proj

    TRY_EXECUTE_WITH_LOG(gp_->execute(gate_in, gate_outputs, wait), dry_run_,
                         ReportInventory::getInstance().addData,
                         "mladfmatmulbias_gp_" + std::to_string(gp_M),
                         std::to_string(input_shape[0]) + "_" +
                             std::to_string(gp_M) + "_" +
                             std::to_string(input_shape[2]));

    // Up proj

    TRY_EXECUTE_WITH_LOG(up_->execute(up_in, up_outputs, wait), dry_run_,
                         ReportInventory::getInstance().addData,
                         "mladfmatmulbias_up_" + std::to_string(gp_M),
                         std::to_string(input_shape[0]) + "_" +
                             std::to_string(gp_M) + "_" +
                             std::to_string(input_shape[2]));

    // Silu
    auto silu_output = silu->get_outputs();
    std::vector<size_t> a_shape_silu = {static_cast<size_t>(gp_M),
                                        static_cast<size_t>(gp_n)};
    silu->set_kernel_shape(a_shape_silu);

    TRY_EXECUTE_WITH_LOG(silu->execute(gate_outputs, silu_output, wait),
                         dry_run_, ReportInventory::getInstance().addData,
                         "mladfmatmulbias_silu_" + std::to_string(gp_M),
                         std::to_string(input_shape[0]) + "_" +
                             std::to_string(gp_M) + "_" + std::to_string(gp_n));

    ewmul->set_kernel_shape(a_shape_silu);
    std::vector<xrt::bo> ewmul_inputs = {silu_output[0], up_outputs[0]};
    auto ewmul_outputs = ewmul->get_outputs();

    TRY_EXECUTE_WITH_LOG(ewmul->execute(ewmul_inputs, ewmul_outputs, wait),
                         dry_run_, ReportInventory::getInstance().addData,
                         "mladfmatmulbias_emul_" + std::to_string(gp_M),
                         std::to_string(input_shape[0]) + "_" +
                             std::to_string(gp_M) + "_" + std::to_string(gp_n));

    std::vector<xrt::bo> dp_inputs = {ewmul_outputs[0], down_const[cnt_]};

    TRY_EXECUTE_WITH_LOG(dp_->execute(dp_inputs, dp_outputs), dry_run_,
                         ReportInventory::getInstance().addData,
                         "mladfmatmulbias_dp_" + std::to_string(gp_M),
                         std::to_string(input_shape[0]) + "_" +
                             std::to_string(gp_M) + "_" + std::to_string(gp_n));
  }

  MY_LOG(2) << "- MLP compute done ...";

  if ((std::find(supported_lengths.begin(), supported_lengths.end(), M) !=
       supported_lengths.end()) &&
      ENV_PARAM(USE_AIE_SSMLP) == 1) {

    MY_LOG(2) << "- AMD SSLRN 2 AIE...";
    auto add2_inputs = add2_->get_inputs();
    auto add2_outputs = add2_->get_outputs();
    auto rms_norm2_wt_inputs = rms_norm2_->get_inputs()[1];
    auto rms_norm2_outputs = rms_norm2_->get_outputs();

    auto output = ctx.GetOutput(0, dimensions_input); // Output activation
    auto ssmlp_out_data = output.GetTensorMutableData<uint16_t>();

    add2_->set_kernel_shape(a_shape);
    rms_norm2_->set_kernel_shape(a_shape);
    if (num_outputs == 2) {
      // Input BO
      std::vector<xrt::bo> add_in = {add1_outputs_[0], dp_outputs[0]};
      TRY_EXECUTE_WITH_LOG(add_->execute(add_in, add2_outputs), dry_run_,
                           ReportInventory::getInstance().addData,
                           "mladf_add_" + std::to_string(M),
                           std::to_string(input_shape[0]) + "_" +
                               std::to_string(M) + "_" +
                               std::to_string(input_shape[2]));

      auto skip_input_bias_add_output =
          ctx.GetOutput(1, dimensions_input); // Output activation

      auto skip_out_data =
          skip_input_bias_add_output.GetTensorMutableData<uint16_t>();

      uint16_t* rms_wt2_map = rms_norm2_wt_inputs.map<uint16_t*>();
      memcpy((void*)rms_wt2_map, (void*)wts2_, num_el2 * sizeof(uint16_t));
      rms_norm2_wt_inputs.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      std::vector<xrt::bo> rms_in = {add2_outputs[0], rms_norm2_wt_inputs};

      TRY_EXECUTE_WITH_LOG(rms_norm_->execute(rms_in, rms_norm2_outputs),
                           dry_run_, ReportInventory::getInstance().addData,
                           "rms_norm_" + std::to_string(M),
                           std::to_string(input_shape[0]) + "_" +
                               std::to_string(M) + "_" +
                               std::to_string(input_shape[2]));

      add2_outputs[0].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
      uint16_t* output_data_1 = add2_outputs[0].map<uint16_t*>();
      memcpy(skip_out_data, output_data_1, B * M * K * sizeof(uint16_t));

    } else {
      std::vector<xrt::bo> rms_in = {add1_outputs_[0], rms_norm2_wt_inputs};

      // Execute RMS Norm
      TRY_EXECUTE_WITH_LOG(rms_norm_->execute(rms_in, rms_norm2_outputs),
                           dry_run_, ReportInventory::getInstance().addData,
                           "rms_norm_" + std::to_string(M),
                           std::to_string(input_shape[0]) + "_" +
                               std::to_string(M) + "_" +
                               std::to_string(input_shape[2]));
    }
    // copy and sync rms_norm_outputs
    rms_norm2_outputs[0].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    uint16_t* output_data_ = rms_norm2_outputs[0].map<uint16_t*>();
    memcpy(ssmlp_out_data, output_data_, B * M * K * sizeof(uint16_t));

  } else {
    MY_LOG(2) << "- AMD SSLRN 2 CPU ...\n";
    size_t num_elements = B * M * K;

    // syncing DP OUTPUT from MLP
    dp_outputs[0].sync(XCL_BO_SYNC_BO_FROM_DEVICE);          // dp_output
    uint16_t* dp_output_bo = dp_outputs[0].map<uint16_t*>(); // dp_output

    if (M != 1) // M is an usupported shape
    {
      if (un_input_a == nullptr)
#ifdef _WIN32
        un_input_a = (float*)_aligned_malloc(num_elements * sizeof(float), 64);
#else
        un_input_a = (float*)aligned_alloc(64, num_elements * sizeof(float));
#endif

      if (un_input_b == nullptr)
#ifdef _WIN32
        un_input_b = (float*)_aligned_malloc(num_elements * sizeof(float), 64);
#else
        un_input_b = (float*)aligned_alloc(64, num_elements * sizeof(float));
#endif
      if (un_output_1 == nullptr)
#ifdef _WIN32
        un_output_1 = (float*)_aligned_malloc(num_elements * sizeof(float), 64);
#else
        un_output_1 = (float*)aligned_alloc(64, num_elements * sizeof(float));
#endif

      if (un_output_2 == nullptr)
#ifdef _WIN32
        un_output_2 = (float*)_aligned_malloc(num_elements * sizeof(float), 64);
#else
        un_output_2 = (float*)aligned_alloc(64, num_elements * sizeof(float));
#endif
      bfloat16_to_float_avx512_unrolled(dp_output_bo, un_input_b, num_elements);
    } else {
#ifdef _WIN32
      if (input_a == nullptr)
        input_a = (float*)_aligned_malloc(num_elements * sizeof(float), 64);
      if (input_b == nullptr)
        input_b = (float*)_aligned_malloc(num_elements * sizeof(float), 64);
      if (output_1 == nullptr)
        output_1 = (float*)_aligned_malloc(num_elements * sizeof(float), 64);
      if (output_2 == nullptr) {
        output_2 = (float*)_aligned_malloc(num_elements * sizeof(float), 64);
      }

#else
      if (input_a == nullptr)
        input_a = (float*)aligned_alloc(64, num_elements * sizeof(float));
      if (input_b == nullptr)
        input_b = (float*)aligned_alloc(64, num_elements * sizeof(float));

      if (output_1 == nullptr)
        output_1 = (float*)aligned_alloc(64, num_elements * sizeof(float));

      if (output_2 == nullptr) {
        output_2 = (float*)aligned_alloc(64, num_elements * sizeof(float));
      }
      un_output_2 = (float*)aligned_alloc(64, num_elements * sizeof(float));
#endif
      bfloat16_to_float_avx512_unrolled(dp_output_bo, input_b, num_elements);
    }
    // Define input shape
    std::vector<int64_t> input_shape = {
        (int64_t)B, (int64_t)M,
        (int64_t)K}; // Replace with your actual input shape

    // Create memory info
    Ort::MemoryInfo memory_info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // Create input tensor

    if (M != 1) {
      Ort::Value input_tensor_a = Ort::Value::CreateTensor<float>(
          memory_info, un_output_2, num_elements, input_shape.data(),
          input_shape.size());
      Ort::Value input_tensor_b = Ort::Value::CreateTensor<float>(
          memory_info, un_input_b, num_elements, input_shape.data(),
          input_shape.size());
      Ort::Value output_tensor_1 = Ort::Value::CreateTensor<float>(
          memory_info, un_output_1, num_elements, input_shape.data(),
          input_shape.size());
      Ort::Value output_tensor_2 = Ort::Value::CreateTensor<float>(
          memory_info, un_input_a, num_elements, input_shape.data(),
          input_shape.size());
      const OrtValue* inputs[3] = {input_tensor_a, input_tensor_b, m2_weights};

      OrtValue* outputs[4] = {output_tensor_1, nullptr, nullptr,
                              output_tensor_2};

      op_k.Invoke(context, inputs, 3, outputs, 4);
    } else {
      Ort::Value input_tensor_a = Ort::Value::CreateTensor<float>(
          memory_info, output_2, num_elements, input_shape.data(),
          input_shape.size());
      Ort::Value input_tensor_b = Ort::Value::CreateTensor<float>(
          memory_info, input_b, num_elements, input_shape.data(),
          input_shape.size());
      Ort::Value output_tensor_1 = Ort::Value::CreateTensor<float>(
          memory_info, output_1, num_elements, input_shape.data(),
          input_shape.size());
      Ort::Value output_tensor_2 = Ort::Value::CreateTensor<float>(
          memory_info, input_a, num_elements, input_shape.data(),
          input_shape.size());
      const OrtValue* inputs[3] = {input_tensor_a, input_tensor_b, m2_weights};

      OrtValue* outputs[4] = {output_tensor_1, nullptr, nullptr,
                              output_tensor_2};

      op_k.Invoke(context, inputs, 3, outputs, 4);
    }

    auto output_token_tensor =
        ctx.GetOutput(0, dimensions_input); // Output activation
    auto sslrn_out_data_token =
        output_token_tensor.GetTensorMutableData<uint16_t>();

    if (M != 1) {
      float_to_bfloat16_avx512_unrolled(un_output_1, sslrn_out_data_token,
                                        M * K); // M x K
    } else {
      float_to_bfloat16_avx512_unrolled(output_1, sslrn_out_data_token,
                                        M * K); // M x K
    }

    if (num_outputs == 2) {
      auto skip_input_bias_add_output_token =
          ctx.GetOutput(1, dimensions_input); // Output activation
      auto skip_out_data_token =
          skip_input_bias_add_output_token.GetTensorMutableData<uint16_t>();

      if (M != 1) {
        float_to_bfloat16_avx512_unrolled(un_input_a, skip_out_data_token,
                                          M * K);
      } // M x K
      else {
        float_to_bfloat16_avx512_unrolled(input_a, skip_out_data_token,
                                          M * K); // M x K
      }
    }

    MY_LOG(2) << "- AMD SSLRN2 CPU done ...\n";
  }
#ifdef _WIN32
  if (sslrn_out_data_token1)
    _aligned_free(sslrn_out_data_token1);
  if (supported_shapes == false && M != 1) {
    if (un_input_a)
      _aligned_free(un_input_a);
    if (un_input_b)
      _aligned_free(un_input_b);
    if (un_output_1)
      _aligned_free(un_output_1);
    if (un_output_2)
      _aligned_free(un_output_2);
  }
#else
  if (sslrn_out_data_token1)
    free(sslrn_out_data_token1);
  if (supported_shapes == false && M != 1) {
    if (un_input_a)
      free(un_input_a);
    if (un_input_b)
      free(un_input_b);
    if (un_output_1)
      free(un_output_1);
    if (un_output_2)
      free(un_output_2);
  }
#endif

  MY_LOG(2) << "- AMD SSMLP compute done ...\n";

} // end of compute

} // namespace ort_ssmlp_custom_op
