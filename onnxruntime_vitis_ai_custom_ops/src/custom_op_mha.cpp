/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */
#include "custom_op_mha.hpp"

#if defined(_WIN32)
#  include <intrin.h>
#else
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

#include <glog/logging.h>
#include <sstream>

#include "reporter.hpp"
#include "vitis/ai/profiling.hpp"

DEF_ENV_PARAM(DEBUG_MHA_CUSTOM_OP, "0")
DEF_ENV_PARAM(USE_AIE_MHA, "1")
DEF_ENV_PARAM(MHA_PARALLEL_BATCH, "1")
DEF_ENV_PARAM_2(MLADF_VERSION, "v1", std::string)
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_MHA_CUSTOM_OP) >= n)
DEF_ENV_PARAM(DRY_RUN, "0")

namespace ort_mha_custom_op {
// Custom Op Domain

std::once_flag MyCustomOpKernel::initFlag;

void* MHA_Allocator::get_buffer(size_t sz, BufferInfo& buffer) {
  if (buffer.first && buffer.second >= sz) {
    // size already satisfied
    return buffer.first;
  } else if (!buffer.first && buffer.second == 0) {
    // first alloc
    MY_LOG(2) << "mha initial memory allocatation, size " << sz;
    buffer.first = (void*)allocator_.Alloc(sz);
    buffer.second = sz;
  } else {
    // reallocation
    size_t new_sz = sz * growth_factor_;
    if (buffer.first) {
      // allocator_.Free(buffer.first);
      free_list_.push_back(buffer);
    }
    MY_LOG(2) << "mha reallocating memory, new size " << new_sz
              << " original size " << buffer.second;
    buffer.first = (void*)allocator_.Alloc(new_sz);
    buffer.second = new_sz;
  }
  return buffer.first;
}

static float bfloat16_to_float_single(uint16_t v) {
  union {
    uint32_t i;
    float f;
  } u;
  u.i = (uint32_t(v)) << 16;
  return u.f;
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

static std::string shape2str(const std::vector<int64_t>& v) {
  std::stringstream ss("");
  for (size_t i = 0; i < v.size() - 1; i++)
    ss << v[i] << "x";
  ss << v[v.size() - 1];
  return ss.str();
}

// transpose [0, 1, 2, 3] to [0, 2, 1, 3]
static void transpose0213_old(uint16_t* input_data, int D0, int D1, int D2,
                              int D3) {
  int tensor_size = D0 * D1 * D2 * D3;
  uint16_t* temp_data = new uint16_t[tensor_size];
  for (int d0 = 0; d0 < D0; ++d0) {
    for (int d1 = 0; d1 < D1; ++d1) {
      for (int d2 = 0; d2 < D2; ++d2) {
        for (int d3 = 0; d3 < D3; ++d3) {
          int inputIndex = d0 * D1 * D2 * D3 + d1 * D2 * D3 + d2 * D3 + d3;
          int outputIndex = d0 * D2 * D1 * D3 + d2 * D1 * D3 + d1 * D3 + d3;
          temp_data[outputIndex] = input_data[inputIndex];
        }
      }
    }
  }
  std::memcpy(input_data, temp_data, tensor_size * sizeof(uint16_t));
  delete[] temp_data;
}

// transpose [0, 1, 2, 3] to [0, 2, 1, 3]
void MyCustomOpKernel::transpose0213(uint16_t* output_data,
                                     uint16_t* input_data, int D0, int D1,
                                     int D2, int D3,
                                     OrtKernelContext* context) {
  int tensor_size = D0 * D1 * D2 * D3;
  int64_t input_shape[4] = {D0, D1, D2, D3};
  int64_t output_shape[4] = {D0, D2, D1, D3};
  Ort::MemoryInfo info =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  auto bf16_enum =
      ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
  OrtValue* input = nullptr;
  api_->CreateTensorWithDataAsOrtValue(info, input_data,
                                       tensor_size * sizeof(uint16_t),
                                       input_shape, 4, bf16_enum, &input);
  OrtValue* output = nullptr;
  api_->CreateTensorWithDataAsOrtValue(info, output_data,
                                       tensor_size * sizeof(uint16_t),
                                       output_shape, 4, bf16_enum, &output);
  const OrtValue* inputs[1] = {input};
  OrtValue* outputs[1] = {output};

  transpose0213_built_in.Invoke(context, inputs, 1, outputs, 1);
}

// slice [1, 1, S1, S2] from [1, N, S1, S2]
template <typename T>
static void slice_tensor_loop(const T* input_tensor, int B, int N, int S1,
                              int S2, T* output_tensor) {
  int slice_size = S1 * S2;
  for (int b = 0; b < B; b++) {
    std::memcpy(output_tensor + b * slice_size,
                input_tensor + b * N * slice_size, slice_size * sizeof(T));
  }
}

inline bool check_prefill(int past_seq_len) { return (past_seq_len == 0); }

// free buffer if data_ptr is not null
template <typename T>
void free_buffer(Ort::AllocatorWithDefaultOptions& allocator, T* data_ptr) {
  if (data_ptr != nullptr) {
    allocator.Free(data_ptr);
  }
}

// pad qkv to qkv_padded with 0
void pad_qkv(uint16_t* t, uint16_t* t_padded, int64_t seqlen,
             int64_t seqlen_padded, int64_t num_heads, int64_t head_size) {
  std::memset(t_padded, 0,
              num_heads * seqlen_padded * head_size * sizeof(uint16_t));
  for (int64_t n = 0; n < num_heads; n++) {
    std::memcpy(t_padded + n * seqlen_padded * head_size,
                t + n * seqlen * head_size,
                seqlen * head_size * sizeof(uint16_t));
  }
}

// pad s to s_padded with neg_inf_ui16
void pad_rpb(uint16_t* t, uint16_t* t_padded, int64_t seqlen,
             int64_t seqlen_padded) {
  const uint16_t neg_inf_ui16 = float_to_bfloat16(-3.389e38f);
  for (int64_t s = 0; s < seqlen; s++) {
    std::memcpy(t_padded + s * seqlen_padded, t + s * seqlen,
                seqlen * sizeof(uint16_t));
    // pad s to s_padded with neg_inf_ui16
    for (int64_t s1 = seqlen; s1 < seqlen_padded; s1++) {
      t_padded[s * seqlen_padded + s1] = neg_inf_ui16;
    }
  }
  for (int64_t s = seqlen; s < seqlen_padded; s++) {
    for (int64_t s1 = 0; s1 < seqlen_padded; s1++) {
      t_padded[s * seqlen_padded + s1] = neg_inf_ui16;
    }
  }
}

// slice the actual output from padded output
void slice_output(uint16_t* output_padded, uint16_t* output, int64_t seqlen,
                  int64_t seqlen_padded, int64_t num_heads, int64_t head_size,
                  std::string version = "v0") {
  if (version == "v0") {
    for (int64_t n = 0; n < num_heads; n++) {
      std::memcpy(output + n * seqlen * head_size,
                  output_padded + n * seqlen_padded * head_size,
                  seqlen * head_size * sizeof(uint16_t));
    }
  } else {
    std::memcpy(output, output_padded,
                seqlen * num_heads * head_size * sizeof(uint16_t));
  }
}

void MyCustomOpKernel::set_params() {
  std::vector<size_t> a_shape_1 = {32, 2048, 128};
  bmm1_->set_params("BMM", a_shape_1);
  std::vector<size_t> a_shape_2 = {32, 2048, 2048};
  bmm2_->set_params("BMM", a_shape_2);
}

void MyCustomOpKernel::LazyInit() {
  dry_run_ = 0;
  if (ENV_PARAM(DRY_RUN) == 1)
    dry_run_ = 1;
  mladf_version_ = ENV_PARAM(MLADF_VERSION);
  std::map<std::string, std::any> attr = {{"op_version", mladf_version_}};
  MY_LOG(2) << "MLADF_VERSION: " << mladf_version_ << std::endl;
  static ryzenai::bmm<uint16_t, uint16_t, uint16_t> bmm1 =
      ryzenai::bmm<uint16_t, uint16_t, uint16_t>("bfloat16", "bfloat16",
                                                 "bfloat16", true, true, attr);
  std::map<std::string, std::any> attr_softmax = {
      {"skip_create_input", 1},
      {"skip_create_output", 1},
      {"op_version", mladf_version_}};
  static ryzenai::masked_softmax<uint16_t, uint16_t, uint16_t> softmax =
      ryzenai::masked_softmax<uint16_t, uint16_t, uint16_t>("bfloat16", true,
                                                            attr_softmax);
  static ryzenai::bmm<uint16_t, uint16_t, uint16_t> bmm2 =
      ryzenai::bmm<uint16_t, uint16_t, uint16_t>("bfloat16", "bfloat16",
                                                 "bfloat16", true, false, attr);
  if (bmm1_ == nullptr) {
    bmm1_ = &bmm1;
  }

  if (bmm2_ == nullptr) {
    bmm2_ = &bmm2;
  }
  if (softmax_ == nullptr) {
    softmax_ = &softmax;
  }
}

MyCustomOpKernel::MyCustomOpKernel(const OrtKernelInfo* k_info,
                                   const OrtApi& api) {
  api_ = &api;
  std::string node_name;
  // Get constant info for the node
  Ort::ConstKernelInfo info{k_info};

  // Get Logger
  m_logger = info.GetLogger();

  // Get attrs
  num_heads_ = info.GetAttribute<int64_t>("num_heads");
  //  Get inputs attrs
  m_node_name = info.GetNodeName();

  // onnx built in mha op
  MY_LOG(2) << "initialization for onnx mha builtin op..." << std::endl;

  const char* mha_type_constraint_names[1] = {"T"};
  ONNXTensorElementDataType mha_type_constraint_values[1] = {
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT};

  int64_t val_num_heads = num_heads_;
  auto attr_num_heads = Ort::OpAttr("num_heads", &val_num_heads, 1,
                                    OrtOpAttrType::ORT_OP_ATTR_INT);
  Ort::OpAttr mha_attrs[1] = {std::move(attr_num_heads)};
  mha_built_in =
      Ort::Op::Create(info.Copy(), "MultiHeadAttention", "com.microsoft", 1,
                      mha_type_constraint_names, mha_type_constraint_values, 1,
                      mha_attrs, 1, 8, 3); // 1 attributes, 8 inputs, 3 output
  MY_LOG(2) << "initialization for onnx mha builtin op done..." << std::endl;

  // onnx built in transpose op
  MY_LOG(2) << "initialization for onnx transpose builtin op..." << std::endl;

  const char* transpose_type_constraint_names[1] = {"T"};
  ONNXTensorElementDataType transpose_type_constraint_values[1] = {
      ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16};
  std::vector<int64_t> perm_vec{0, 2, 1, 3};
  auto perm = Ort::OpAttr("perm", perm_vec.data(), perm_vec.size(),
                          OrtOpAttrType::ORT_OP_ATTR_INTS);
  Ort::OpAttr transpose_attrs[1] = {std::move(perm)};
  transpose0213_built_in = Ort::Op::Create(
      info.Copy(), "Transpose", "ai.onnx", 21, transpose_type_constraint_names,
      transpose_type_constraint_values, 1, transpose_attrs, 1, 1,
      1); // 1 attributes, 1 input, 1 output
  MY_LOG(2) << "initialization for onnx transpose builtin op done..."
            << std::endl;
  if (ENV_PARAM(USE_AIE_MHA) == 1) {
    // aie mha op from DD
    MY_LOG(2) << "initialization for mha aie custom-op..." << std::endl;

    LazyInit();
    std::call_once(initFlag, [this]() { this->set_params(); });
    bmm1_->debug(false);
    bmm2_->debug(false);
    softmax_->debug(false);
    bmm1_inputs = bmm1_->get_inputs();
    bmm1_outputs = bmm1_->get_outputs();
    bmm2_inputs = bmm2_->get_inputs();
    bmm2_outputs = bmm2_->get_outputs();
    softmax_mask = softmax_->get_inputs()[1];

    MY_LOG(2) << "initialization for mha aie custom-op done." << std::endl;
  }
}

void MyCustomOpKernel::aie_execute(OrtTensor& query_states,
                                   OrtTensor& key_states,
                                   OrtTensor& value_states,
                                   OrtTensor& attention_mask,
                                   OrtTensor& output) {
  // Code taken from:
  // https://gitenterprise.xilinx.com/VitisAI/transformers/blob/main/ops/torch_cpp/src/mha_npu_torch.cpp#L41

  // Get Shapes
  int B = query_states.shape[0];   // Batch
  int N = query_states.shape[1];   // Number of heads
  int S_q = query_states.shape[2]; // Sequence length of query
  int H = query_states.shape[3];   // Head_size
  int S_k = key_states.shape[2];   // Sequence length of key

  // Get data pointers
  auto xCasted = static_cast<uint16_t*>(query_states.data);
  auto yCasted = static_cast<uint16_t*>(key_states.data);
  auto mCasted = static_cast<uint16_t*>(attention_mask.data);
  auto y2Casted = static_cast<uint16_t*>(value_states.data);

  std::vector<size_t> bmm1_shape{(size_t)B * N, (size_t)S_q, (size_t)H};
  std::vector<size_t> softmax_shape{(size_t)N, (size_t)S_q, (size_t)S_k};
  std::vector<size_t> bmm2_shape{(size_t)B * N, (size_t)S_q, (size_t)S_k};
  bmm1_->set_execute_kernel_shape(bmm1_shape);
  bmm2_->set_execute_kernel_shape(bmm2_shape);
  softmax_->set_params("softmax", softmax_shape);

  // Get XRT Buffers
  uint16_t* a_bo_map = bmm1_inputs[0].map<uint16_t*>();
  memcpy((void*)a_bo_map, (void*)xCasted, B * N * S_q * H * sizeof(uint16_t));

  uint16_t* b_bo_map = bmm1_inputs[1].map<uint16_t*>();
  memcpy((void*)b_bo_map, (void*)yCasted, B * N * S_k * H * sizeof(uint16_t));

  // Sync data
  bmm1_inputs[0].sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bmm1_inputs[1].sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bmm1_outputs[0].sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // Execute QKT MatMul
  TRY_EXECUTE_WITH_LOG(bmm1_->execute(bmm1_inputs, bmm1_outputs), dry_run_,
                       ReportInventory::getInstance().addData,
                       "bmm_" + std::to_string(S_q),
                       std::to_string(B * N) + "_" + std::to_string(S_q) + "_" +
                           std::to_string(H));

  // Get SM buffers
  uint16_t* mask_bo_map = softmax_mask.map<uint16_t*>();
  memcpy((void*)mask_bo_map, (void*)mCasted, S_q * S_k * sizeof(uint16_t));

  // Sync
  softmax_mask.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bmm2_inputs[1].sync(XCL_BO_SYNC_BO_TO_DEVICE);

  std::vector<xrt::bo> inputs = {bmm1_outputs[0], softmax_mask};
  std::vector<xrt::bo> outputs = {bmm2_inputs[0]};

  // Execute Softmax
  TRY_EXECUTE_WITH_LOG(softmax_->execute(inputs, outputs), dry_run_,
                       ReportInventory::getInstance().addData,
                       "masked_softmax_" + std::to_string(S_q),
                       std::to_string(N) + "_" + std::to_string(S_q) + "_" +
                           std::to_string(S_k));

  // Get SMV MatMul buffers
  uint16_t* value_bo_map = bmm2_inputs[1].map<uint16_t*>();
  memcpy((void*)value_bo_map, (void*)y2Casted,
         B * N * S_k * H * sizeof(uint16_t));

  // Sync
  bmm2_inputs[1].sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bmm2_outputs[0].sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // Execute SMV MatMul
  TRY_EXECUTE_WITH_LOG(bmm2_->execute(bmm2_inputs, bmm2_outputs), dry_run_,
                       ReportInventory::getInstance().addData,
                       "bmm_" + std::to_string(S_q),
                       std::to_string(B * N) + "_" + std::to_string(S_q) + "_" +
                           std::to_string(S_k));

  // Sync output
  bmm2_outputs[0].sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  // Copy output from XRT BO to OrtTensor
  uint16_t* out = bmm2_outputs[0].map<uint16_t*>();
  uint64_t tensor_size = output.size;
  MY_LOG(2) << "output size from aie: " << tensor_size;
  memcpy(output.data, out, tensor_size * sizeof(uint16_t));
  MY_LOG(2) << "output copy done.";
}

static void GetInputTensorData(DataPtrWrapper& data_ptr, int data_type,
                               Ort::ConstValue& input) {
  if (data_type == 16) {
    data_ptr.T = (void*)input.GetTensorData<uint16_t>();
    data_ptr.dtag = "bf16";
  } else if (data_type == 1) {
    data_ptr.T = (void*)input.GetTensorData<float>();
    data_ptr.dtag = "float32";
  } else {
    throw std::runtime_error("Not supported data type, tag:" +
                             std::string(std::to_string(data_type)));
  }
}

static void GetOutputTensorMutableData(DataPtrWrapper& data_ptr, int data_type,
                                       Ort::UnownedValue& output) {
  if (data_type == 16) {
    data_ptr.T = (void*)output.GetTensorMutableData<uint16_t>();
    data_ptr.dtag = "bf16";
  } else if (data_type == 1) {
    data_ptr.T = (void*)output.GetTensorMutableData<float>();
    data_ptr.dtag = "float32";
  } else {
    throw std::runtime_error("Not supported data type, tag:" +
                             std::string(std::to_string(data_type)));
  }
}

bool isBf16Model(const DataPtrWrapper& q, const DataPtrWrapper& k,
                 DataPtrWrapper& v, DataPtrWrapper& out,
                 DataPtrWrapper& present_key, DataPtrWrapper& present_value) {
  return q.is_bf16() && k.is_bf16() && v.is_bf16() && out.is_bf16() &&
         present_key.is_bf16() && present_value.is_bf16();
}

struct KVCacheData {
  /// past/present k
  const uint16_t* past_k_bf16 = {};
  const float* present_k_fp32 = {};
  uint16_t* present_k_bf16 = {};
  /// past/present v
  const uint16_t* past_v_bf16 = {};
  const float* present_v_fp32 = {};
  uint16_t* present_v_bf16 = {};
  /// length
  int past_SxH = 0;
  int present_SxH = 0;
  int head_size = 0;
};

void KV_cache_copy(void* raw_data, size_t n) {
  auto data = reinterpret_cast<KVCacheData*>(raw_data);
  auto past_offset = n * data->past_SxH;
  auto present_offset = n * data->present_SxH;
  auto copy_size = data->past_SxH;
  auto new_token_offset = data->past_SxH;
  std::memcpy(data->present_k_bf16 + present_offset,
              data->past_k_bf16 + past_offset, copy_size * sizeof(uint16_t));
  vec_float32_to_bf16(data->present_k_bf16 + present_offset + new_token_offset,
                      data->present_k_fp32 + present_offset + new_token_offset,
                      data->head_size);
  std::memcpy(data->present_v_bf16 + present_offset,
              data->past_v_bf16 + past_offset, copy_size * sizeof(uint16_t));
  vec_float32_to_bf16(data->present_v_bf16 + present_offset + new_token_offset,
                      data->present_v_fp32 + present_offset + new_token_offset,
                      data->head_size);
}

/// get the best num_batch for ctx.ParallelFor
/// based on the TPS on Birman+
int get_best_parallel_batch(int S) {
  assert(S >= 0 && S <= 2048);
  int64_t best_batch = 1;
  if ((S - 128) <= 0)
    best_batch = 1;
  else if ((S - 256) <= 0)
    best_batch = 2;
  else if ((S - 512) <= 0)
    best_batch = 4;
  else if ((S - 1024) <= 0)
    best_batch = 8;
  else
    best_batch = 16;
  return best_batch;
}

void MyCustomOpKernel::Compute(OrtKernelContext* context) {
  MY_LOG(2) << "- AMD MHA compute start ...\n";
  __TIC__(Compute)
  Ort::KernelContext ctx(context);
  auto num_inputs = ctx.GetInputCount();
  auto num_outputs = ctx.GetOutputCount();

  MY_LOG(2) << "num_inputs " << num_inputs << " "
            << "num_outputs " << num_outputs << " \n";

  // Prepare input/output tensors
  // Extracting the input and output information
  MY_LOG(2) << "Getting inputs...\n";
  auto query = ctx.GetInput(0);            // Query
  auto key = ctx.GetInput(1);              // Key
  auto value = ctx.GetInput(2);            // Value
  auto bias = ctx.GetInput(3);             // bias
  auto key_padding_mask = ctx.GetInput(4); // key_padding_mask
  auto relative_position_bias = ctx.GetInput(
      5); // relative_position_bias (extra_add_qk in onnx cpu kernel)
  auto past_key = ctx.GetInput(6);   // past_key
  auto past_value = ctx.GetInput(7); // past_value
  MY_LOG(2) << "Getting inputs shape and data...\n";

  Ort::AllocatorWithDefaultOptions allocator;

  // Query data / shape
  auto query_data_type = query.GetTensorTypeAndShapeInfo().GetElementType();
  auto query_shape = query.GetTensorTypeAndShapeInfo().GetShape();
  auto query_size = query.GetTensorTypeAndShapeInfo().GetElementCount();
  DataPtrWrapper query_data;
  GetInputTensorData(query_data, query_data_type, query);
  MY_LOG(2) << "Query data: " << query_data.toString();

  // Key data / shape
  auto key_data_type = key.GetTensorTypeAndShapeInfo().GetElementType();
  auto key_shape = key.GetTensorTypeAndShapeInfo().GetShape();
  auto key_size = key.GetTensorTypeAndShapeInfo().GetElementCount();
  DataPtrWrapper key_data;
  GetInputTensorData(key_data, key_data_type, key);
  MY_LOG(2) << "Key data: " << key_data.toString();

  // Value data / shape
  auto value_data_type = value.GetTensorTypeAndShapeInfo().GetElementType();
  auto value_shape = value.GetTensorTypeAndShapeInfo().GetShape();
  auto value_size = value.GetTensorTypeAndShapeInfo().GetElementCount();
  DataPtrWrapper value_data;
  GetInputTensorData(value_data, value_data_type, value);
  MY_LOG(2) << "Value data: " << value_data.toString();

  // relative_position_bias / shape
  auto relative_position_bias_type =
      relative_position_bias.GetTensorTypeAndShapeInfo().GetElementType();
  auto relative_position_bias_shape =
      relative_position_bias.GetTensorTypeAndShapeInfo().GetShape();
  auto relative_position_bias_size =
      relative_position_bias.GetTensorTypeAndShapeInfo().GetElementCount();
  DataPtrWrapper relative_position_bias_data;
  GetInputTensorData(relative_position_bias_data, relative_position_bias_type,
                     relative_position_bias);
  MY_LOG(2) << "rpb data: " << relative_position_bias_data.toString();

  // Past Key data / shape
  auto past_key_shape = past_key.GetTensorTypeAndShapeInfo().GetShape();
  auto past_key_size = past_key.GetTensorTypeAndShapeInfo().GetElementCount();
  auto past_key_type = past_key.GetTensorTypeAndShapeInfo().GetElementType();
  DataPtrWrapper past_key_data;
  GetInputTensorData(past_key_data, past_key_type, past_key);

  // Past Value data / shape
  auto past_value_shape = past_value.GetTensorTypeAndShapeInfo().GetShape();
  auto past_value_size =
      past_value.GetTensorTypeAndShapeInfo().GetElementCount();
  auto past_value_type =
      past_value.GetTensorTypeAndShapeInfo().GetElementType();
  DataPtrWrapper past_value_data;
  GetInputTensorData(past_value_data, past_value_type, past_value);

  __TIC__(ALLOC_OUTPUT)

  MY_LOG(2) << "Getting outputs...\n";
  // Allocate output (primary output shape is same as query shape)
  auto output = ctx.GetOutput(0, query_shape);
  auto output_data_type = output.GetTensorTypeAndShapeInfo().GetElementType();
  auto output_size = output.GetTensorTypeAndShapeInfo().GetElementCount();
  DataPtrWrapper output_data;
  GetOutputTensorMutableData(output_data, output_data_type, output);
  MY_LOG(2) << "output data: " << output_data.toString();

  MY_LOG(2) << "calculate shape for present k/v...\n";
  // calculate shape for present k/v
  int batch_size = static_cast<int>(query_shape[0]);
  int past_sequence_length = static_cast<int>(past_key_shape[2]);
  int kv_sequence_length = static_cast<int>(key_shape[1]);
  int total_sequence_length = past_sequence_length + kv_sequence_length;
  int hidden_size = static_cast<int>(query_shape[2]);
  int v_hidden_size = static_cast<int>(value_shape[2]);
  int num_heads =
      32; // TODO(haozhu): hardcode, should get num_heads from node attrs
  int head_size = static_cast<int>(hidden_size) / num_heads;
  int v_head_size = static_cast<int>(v_hidden_size) / num_heads;
  std::vector<int64_t> present_k_shape(
      {static_cast<int64_t>(batch_size), static_cast<int64_t>(num_heads_),
       static_cast<int64_t>(total_sequence_length),
       static_cast<int64_t>(head_size)});
  std::vector<int64_t> present_v_shape(
      {static_cast<int64_t>(batch_size), static_cast<int64_t>(num_heads_),
       static_cast<int64_t>(total_sequence_length),
       static_cast<int64_t>(v_head_size)});
  MY_LOG(2) << "Getting present_key and present_value...\n";
  auto present_key = ctx.GetOutput(1, present_k_shape);
  auto present_key_size =
      present_key.GetTensorTypeAndShapeInfo().GetElementCount();
  auto present_key_data_type =
      present_key.GetTensorTypeAndShapeInfo().GetElementType();
  DataPtrWrapper present_key_data;
  GetOutputTensorMutableData(present_key_data, present_key_data_type,
                             present_key);
  MY_LOG(2) << "present_key data: " << present_key_data.toString();

  auto present_value = ctx.GetOutput(2, present_v_shape);
  auto present_value_size =
      present_value.GetTensorTypeAndShapeInfo().GetElementCount();
  auto present_value_data_type =
      present_value.GetTensorTypeAndShapeInfo().GetElementType();
  DataPtrWrapper present_value_data;
  GetOutputTensorMutableData(present_value_data, present_value_data_type,
                             present_value);
  MY_LOG(2) << "present_value data: " << present_value_data.toString();

  __TOC__(ALLOC_OUTPUT)

  // print shapes
  MY_LOG(2) << "q shape: " << shape2str(query_shape) << std::endl;
  MY_LOG(2) << "k shape: " << shape2str(key_shape) << std::endl;
  MY_LOG(2) << "v shape: " << shape2str(value_shape) << std::endl;
  MY_LOG(2) << "relative_position_bias shape: "
            << shape2str(relative_position_bias_shape) << std::endl;
  MY_LOG(2) << "past k shape: " << shape2str(past_key_shape) << std::endl;
  MY_LOG(2) << "past v shape: " << shape2str(past_value_shape) << std::endl;
  MY_LOG(2) << "present k shape: " << shape2str(present_k_shape) << std::endl;
  MY_LOG(2) << "present v shape: " << shape2str(present_v_shape) << std::endl;

  int B = query_shape[0];     // batch
  int S = query_shape[1];     // sequence length
  int N = num_heads;          // num_heads
  int H = query_shape[2] / N; // head size

  bool is_prefill = check_prefill(past_sequence_length);
  MY_LOG(2) << "is_prefill: " << is_prefill << std::endl;

  auto& mha_allocator = MHA_Allocator::get_instance();
  // Note(ltp): Using aie kernel when:
  // - prefill phase
  // - USE_AIE_MHA = 1
  // - S <= aie max S length
  if (is_prefill && ENV_PARAM(USE_AIE_MHA) == 1 &&
      S <= mha_aie_kernel_info_.max_seq_length()) {
    MY_LOG(2) << "running AIE kernel" << std::endl;
    if (isBf16Model(query_data, key_data, value_data, output_data,
                    present_key_data, present_value_data)) {
      __TIC__(TransposeInput)
      uint16_t* bf16_query_data_transposed =
          mha_allocator.get_aie_q_t(query_size * sizeof(uint16_t));
      transpose0213(bf16_query_data_transposed, query_data.cast<uint16_t>(), B,
                    S, N, H, context);
      // reuse the present_k/present_v buffer
      transpose0213(present_key_data.cast<uint16_t>(),
                    key_data.cast<uint16_t>(), B, S, N, H, context);
      transpose0213(present_value_data.cast<uint16_t>(),
                    value_data.cast<uint16_t>(), B, S, N, H, context);
      __TOC__(TransposeInput)

      std::vector<int64_t> query_shape_transposed(
          {static_cast<int64_t>(B), static_cast<int64_t>(N),
           static_cast<int64_t>(S), static_cast<int64_t>(H)});
      std::vector<int64_t> key_shape_transposed(
          {static_cast<int64_t>(B), static_cast<int64_t>(N),
           static_cast<int64_t>(S), static_cast<int64_t>(H)});
      std::vector<int64_t> value_shape_transposed(
          {static_cast<int64_t>(B), static_cast<int64_t>(N),
           static_cast<int64_t>(S), static_cast<int64_t>(H)});
      OrtTensor qTensor = {query_shape_transposed, query_size,
                           (void*)bf16_query_data_transposed};
      OrtTensor kTensor = {key_shape_transposed, key_size,
                           (void*)present_key_data.cast<uint16_t>()};
      OrtTensor vTensor = {value_shape_transposed, value_size,
                           (void*)present_value_data.cast<uint16_t>()};

      // Note(ltp): since masked_softmax only supports (1, 1, S, S) as second
      // input, thus we need to slice the position_bias data.
      // Todo(ltp): check all the data in N dimension is equal in
      // relative_position_bias_data.
      size_t rpb_sliced_size = B * 1 * S * S;
      __TIC__(RPBSlicedFloat32ToBf16)
      uint16_t* bf16_rpb_sliced_data =
          mha_allocator.get_aie_rpb(rpb_sliced_size * sizeof(uint16_t));
      vec_float32_to_bf16(bf16_rpb_sliced_data,
                          relative_position_bias_data.cast<float>(),
                          rpb_sliced_size);
      __TOC__(RPBSlicedFloat32ToBf16)

      OrtTensor RPBSlicedTensor = {
          {B, 1, S, S}, rpb_sliced_size, (void*)bf16_rpb_sliced_data};
      __TIC__(bf16KernelOutputAlloc)
      uint16_t* output_ptr = nullptr;
      uint16_t* bf16_kernel_output_data = nullptr;
      if (mladf_version_ == "v0") {
        bf16_kernel_output_data =
            mha_allocator.get_aie_output(output_size * sizeof(uint16_t));
        output_ptr = bf16_kernel_output_data;
      } else if (mladf_version_ == "v1") {
        output_ptr = output_data.cast<uint16_t>();
      } else {
        throw std::runtime_error("MLADF_VERSION mismatch, should be v0 or v1.");
      }
      OrtTensor outTensor = {query_shape, output_size, (void*)output_ptr};
      __TOC__(bf16KernelOutputAlloc)
      uint16_t* q_padded_bf16 = nullptr;
      uint16_t* k_padded_bf16 = nullptr;
      uint16_t* v_padded_bf16 = nullptr;
      uint16_t* rpb_padded_bf16 = nullptr;
      uint16_t* output_padded_bf16 = nullptr;

      // if S in {256, 512, 1024, 2048}, no need to pad,
      // any other S in between need pad.
      if (mha_aie_kernel_info_.is_seq_aie_supported(S)) {
        MY_LOG(2) << "original shape, no need to pad." << std::endl;
        __TIC__(AIECompute)
        const_cast<MyCustomOpKernel*>(this)->aie_execute(
            qTensor, kTensor, vTensor, RPBSlicedTensor, outTensor);
        __TOC__(AIECompute)
      } else {
        MY_LOG(2) << "not original shape, need to pad." << std::endl;
        __TIC__(AllocAndPadTensor)
        int64_t S_padded = mha_aie_kernel_info_.try_pad_seq(S);
        MY_LOG(2) << "padding " << S << " -> " << S_padded << std::endl;
        /// q, k, v and output shapes are all the same {B, N, S_padded, H}
        std::vector<int64_t> io_padded_shape{B, N, S_padded, H};
        std::vector<int64_t> rpb_padded_shape{B, 1, S_padded, S_padded};
        size_t io_padded_size = B * N * S_padded * H;
        size_t rpb_padded_size = B * S_padded * S_padded;
        q_padded_bf16 =
            (uint16_t*)allocator.Alloc(io_padded_size * sizeof(uint16_t));
        k_padded_bf16 =
            (uint16_t*)allocator.Alloc(io_padded_size * sizeof(uint16_t));
        v_padded_bf16 =
            (uint16_t*)allocator.Alloc(io_padded_size * sizeof(uint16_t));
        rpb_padded_bf16 =
            (uint16_t*)allocator.Alloc(rpb_padded_size * sizeof(uint16_t));
        output_padded_bf16 =
            (uint16_t*)allocator.Alloc(io_padded_size * sizeof(uint16_t));
        MY_LOG(2) << "padding qkv" << std::endl;
        pad_qkv(bf16_query_data_transposed, q_padded_bf16, S, S_padded, N, H);
        pad_qkv(present_key_data.cast<uint16_t>(), k_padded_bf16, S, S_padded,
                N, H);
        pad_qkv(present_value_data.cast<uint16_t>(), v_padded_bf16, S, S_padded,
                N, H);
        MY_LOG(2) << "padding rpb" << std::endl;
        pad_rpb(bf16_rpb_sliced_data, rpb_padded_bf16, S, S_padded);
        OrtTensor q_padded_tensor = {io_padded_shape, io_padded_size,
                                     (void*)q_padded_bf16};
        OrtTensor k_padded_tensor = {io_padded_shape, io_padded_size,
                                     (void*)k_padded_bf16};
        OrtTensor v_padded_tensor = {io_padded_shape, io_padded_size,
                                     (void*)v_padded_bf16};
        OrtTensor rpb_padded_tensor = {rpb_padded_shape, rpb_padded_size,
                                       (void*)rpb_padded_bf16};
        OrtTensor output_padded_tensor = {io_padded_shape, io_padded_size,
                                          (void*)output_padded_bf16};
        __TOC__(AllocAndPadTensor)

        __TIC__(AIEComputePad)
        MY_LOG(2) << "aie execute pad" << std::endl;
        const_cast<MyCustomOpKernel*>(this)->aie_execute(
            q_padded_tensor, k_padded_tensor, v_padded_tensor,
            rpb_padded_tensor, output_padded_tensor);
        __TOC__(AIEComputePad)
        MY_LOG(2) << "slice padded output" << std::endl;
        slice_output(output_padded_bf16, output_ptr, S, S_padded, N, H,
                     mladf_version_);
      }
      __TIC__(TransposeOut)
      if (mladf_version_ == "v0") {
        transpose0213(output_data.cast<uint16_t>(), output_ptr, B, N, S, H,
                      context);
      }
      __TOC__(TransposeOut)

      __TIC__(memoryFree)
      free_buffer<uint16_t>(allocator, q_padded_bf16);
      free_buffer<uint16_t>(allocator, k_padded_bf16);
      free_buffer<uint16_t>(allocator, v_padded_bf16);
      free_buffer<uint16_t>(allocator, rpb_padded_bf16);
      free_buffer<uint16_t>(allocator, output_padded_bf16);
      __TOC__(memoryFree)

    } else {
      throw std::runtime_error(
          "Not supported now, only support QKV with bfloat16 as inputs.");
    }

  } else {
    MY_LOG(2) << "running ORT kernel" << std::endl;
    if (isBf16Model(query_data, key_data, value_data, output_data,
                    present_key_data, present_value_data)) {
      __TIC__(ORTCompute)
      __TIC__(ORTKernelAllocInput)
      // inputs: q, k, v, past_key and past_value, relative_postion_bias
      float* float_query_data_conveter =
          mha_allocator.get_q(query_size * sizeof(float));
      float* float_key_data_conveter =
          mha_allocator.get_k(key_size * sizeof(float));
      float* float_value_data_conveter =
          mha_allocator.get_v(value_size * sizeof(float));
      float* float_pask_k_data_conveter =
          mha_allocator.get_past_k(past_key_size * sizeof(float));
      float* float_past_v_data_conveter =
          mha_allocator.get_past_v(past_value_size * sizeof(float));
      __TOC__(ORTKernelAllocInput)

      __TIC__(ORTKernelAllocOutput)
      // outputs
      float* float_output_data_conveter =
          mha_allocator.get_output(output_size * sizeof(float));
      float* float_present_key_data_converter =
          mha_allocator.get_present_k(present_key_size * sizeof(float));
      float* float_present_value_data_converter =
          mha_allocator.get_present_v(present_value_size * sizeof(float));
      __TOC__(ORTKernelAllocOutput)

      Ort::MemoryInfo info =
          Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
      __TIC__(ORTKernelInputBf16ToFloat32)
      vec_bf16_to_float(float_query_data_conveter, query_data.cast<uint16_t>(),
                        query_size);
      vec_bf16_to_float(float_key_data_conveter, key_data.cast<uint16_t>(),
                        key_size);
      vec_bf16_to_float(float_value_data_conveter, value_data.cast<uint16_t>(),
                        value_size);
      vec_bf16_to_float(float_pask_k_data_conveter,
                        past_key_data.cast<uint16_t>(), past_key_size);
      vec_bf16_to_float(float_past_v_data_conveter,
                        past_value_data.cast<uint16_t>(), past_value_size);
      __TOC__(ORTKernelInputBf16ToFloat32)

      // Create single Ort tensors
      OrtValue* fp_q = nullptr;
      OrtValue* fp_k = nullptr;
      OrtValue* fp_v = nullptr;
      OrtValue* fp_p_k = nullptr;
      OrtValue* fp_p_v = nullptr;
      __TIC__(ORTKernelCreateInputOrtTensorValue)
      auto t = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
      api_->CreateTensorWithDataAsOrtValue(
          info, float_query_data_conveter, query_size * sizeof(float),
          query_shape.data(), query_shape.size(), t, &fp_q);
      api_->CreateTensorWithDataAsOrtValue(
          info, float_key_data_conveter, key_size * sizeof(float),
          key_shape.data(), key_shape.size(), t, &fp_k);
      api_->CreateTensorWithDataAsOrtValue(
          info, float_value_data_conveter, value_size * sizeof(float),
          value_shape.data(), value_shape.size(), t, &fp_v);
      api_->CreateTensorWithDataAsOrtValue(
          info, float_pask_k_data_conveter, past_key_size * sizeof(float),
          past_key_shape.data(), past_key_shape.size(), t, &fp_p_k);
      api_->CreateTensorWithDataAsOrtValue(
          info, float_past_v_data_conveter, past_value_size * sizeof(float),
          past_value_shape.data(), past_value_shape.size(), t, &fp_p_v);
      __TOC__(ORTKernelCreateInputOrtTensorValue)

      const OrtValue* inputs[8] = {fp_q,    fp_k,    fp_v,
                                   nullptr, nullptr, relative_position_bias,
                                   fp_p_k,  fp_p_v};

      OrtValue* fp_out = nullptr;
      OrtValue* fp_present_key = nullptr;
      OrtValue* fp_present_value = nullptr;
      __TIC__(ORTKernelCreateOutputOrtTensorValue)
      api_->CreateTensorWithDataAsOrtValue(
          info, float_output_data_conveter, output_size * sizeof(float),
          query_shape.data(), query_shape.size(), t, &fp_out);
      api_->CreateTensorWithDataAsOrtValue(
          info, float_present_key_data_converter,
          present_key_size * sizeof(float), present_k_shape.data(),
          present_k_shape.size(), t, &fp_present_key);
      api_->CreateTensorWithDataAsOrtValue(
          info, float_present_value_data_converter,
          present_value_size * sizeof(float), present_v_shape.data(),
          present_v_shape.size(), t, &fp_present_value);
      __TOC__(ORTKernelCreateOutputOrtTensorValue)

      OrtValue* outputs[3] = {fp_out, fp_present_key, fp_present_value};

      __TIC__(ORTBuiltInKernelCompute)
      mha_built_in.Invoke(context, inputs, 8, outputs, 3);
      __TOC__(ORTBuiltInKernelCompute)

      __TIC__(ORTOutputFloat32ToBf16)
      /// convert float32 output to bfloat16
      vec_float32_to_bf16(output_data.cast<uint16_t>(),
                          float_output_data_conveter, output_size);
      if (ENV_PARAM(USE_AIE_MHA) == 1) {
        int past_seq_len = past_key_shape[2];
        int present_seq_len = present_k_shape[2];
        int num_batch = get_best_parallel_batch(S);
        KVCacheData kv_cache = {past_key_data.cast<uint16_t>(),
                                float_present_key_data_converter,
                                present_key_data.cast<uint16_t>(),
                                past_value_data.cast<uint16_t>(),
                                float_present_value_data_converter,
                                present_value_data.cast<uint16_t>(),
                                past_seq_len * head_size,
                                present_seq_len * head_size,
                                head_size};
        ctx.ParallelFor(KV_cache_copy, static_cast<size_t>(num_heads_),
                        num_batch, &kv_cache);
      } else {
        vec_float32_to_bf16(present_key_data.cast<uint16_t>(),
                            float_present_key_data_converter, present_key_size);
        vec_float32_to_bf16(present_value_data.cast<uint16_t>(),
                            float_present_value_data_converter,
                            present_value_size);
      }
      __TOC__(ORTOutputFloat32ToBf16)

      __TIC__(ORTMemoryFree)
      __TIC__(ORTMemoryFreeInput)
      // place holder
      __TOC__(ORTMemoryFreeInput)
      __TOC__(ORTMemoryFree)

      __TOC__(ORTCompute)

    } else {
      throw std::runtime_error(
          "Not supported now, only support QKV with bfloat16 as inputs.");
    }
  }

  __TOC__(Compute)
  MY_LOG(2) << "- AMD MHA compute done ...\n";
}

} // namespace ort_mha_custom_op
