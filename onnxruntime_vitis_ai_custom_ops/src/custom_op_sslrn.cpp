/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */
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

#include "custom_op_sslrn.hpp"
#include "reporter.hpp"
#include "vitis/ai/profiling.hpp"
#include <glog/logging.h>
#include <sstream>
int cnt_ops = 0;

DEF_ENV_PARAM(DRY_RUN, "0")
DEF_ENV_PARAM(DEBUG_SSLRN_CUSTOM_OP, "0")
DEF_ENV_PARAM(USE_AIE_SSLRN, "1")
DEF_ENV_PARAM_2(MLADF_VERSION, "v1", std::string)
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_SSLRN_CUSTOM_OP) >= n)

namespace ort_sslrn_custom_op {
// Custom Op Domain

std::once_flag MyCustomOpKernel1::initFlag;

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

// static void vec_bf16_to_float(float* dest, const uint16_t* src, size_t size)
// {
//   assert(src != nullptr);
//   assert(dest != nullptr);
//   assert(size > 0);
//   for (auto i = 0u; i < size; i++) {
//     dest[i] = bfloat_to_float(src[i]);
//   }
// }

static void vec_bf16_to_float(float* dest, const uint16_t* src, size_t size) {
  assert(src != nullptr);
  assert(dest != nullptr);
  assert(size > 0);

  bfloat16_to_float_avx512_unrolled(src, dest, size);
}

void MyCustomOpKernel1::LazyInit() {
  dry_run_ = 0;
  if (ENV_PARAM(DRY_RUN) == 1)
    dry_run_ = 1;
  std::string mladf_version_ = ENV_PARAM(MLADF_VERSION);

  if (mladf_version_ != "v0" && mladf_version_ != "v1") {
    std::cerr << "Invalid version. Supported versions are v0 and v1"
              << std::endl;
  }

  std::map<std::string, std::any> attr = {{"op_version", mladf_version_}};
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
}

MyCustomOpKernel1::MyCustomOpKernel1(const OrtKernelInfo* k_info) {
  std::string node_name;
  // Get constant info for the node
  Ort::ConstKernelInfo info{k_info};

  // Get Logger
  m_logger = info.GetLogger();

  //  Get inputs attrs
  m_node_name = info.GetNodeName();

  // onnx built in mha op
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
                         4); // 1 attributes, 5 inputs, 4 outputs
  if (ENV_PARAM(USE_AIE_SSLRN) == 1) {
    LazyInit();
  }

  // Get weights
  int is_constant = 0;
  m_weights = info.GetTensorConstantInput(2, &is_constant);
  const float* wts_data = m_weights.GetTensorData<float>();

  auto dimensions_wts = m_weights.GetTensorTypeAndShapeInfo().GetShape();

  size_t num_el = std::accumulate(dimensions_wts.begin(), dimensions_wts.end(),
                                  (size_t)1, std::multiplies<int64_t>());

#ifdef _WIN32
  wts_ = (uint16_t*)_aligned_malloc(num_el * sizeof(uint16_t), 64);
#else
  wts_ = (uint16_t*)aligned_alloc(64, num_el * sizeof(uint16_t));
#endif
  // Convert floating point to bfloat16 using avx512
  float_to_bfloat16_avx512_unrolled(wts_data, wts_,
                                    num_el); // K

  MY_LOG(2) << "initialization for SSLRN aie custom-op done." << std::endl;
}

MyCustomOpKernel1::~MyCustomOpKernel1() {
  if (wts_)
#ifdef _WIN32
    _aligned_free(wts_);
#else
    free(wts_);
#endif
}

void MyCustomOpKernel1::Compute(OrtKernelContext* context) {
  MY_LOG(2) << "- AMD SSLRN compute start ...\n";
  Ort::KernelContext ctx(context);

  auto num_outputs = ctx.GetOutputCount();

  auto input = ctx.GetInput(0); // Input
  auto skip = ctx.GetInput(1);  // skip
  auto gamma = ctx.GetInput(2); // gamma

  auto dimensions_input = input.GetTensorTypeAndShapeInfo().GetShape();
  auto dimensions_skip = skip.GetTensorTypeAndShapeInfo().GetShape();
  auto dimensions_gamma = gamma.GetTensorTypeAndShapeInfo().GetShape();

  size_t B = dimensions_input[0]; // Batch
  size_t M = dimensions_input[1]; // Seq len
  size_t K = dimensions_input[2]; // Hidden size = Num_heads * Head_size

  auto in_data = input.GetTensorData<uint16_t>();
  auto wts_data = gamma.GetTensorData<float>();
  auto skip_data = skip.GetTensorData<uint16_t>();
  std::vector<size_t> a_shape = {M, K};
  std::vector<size_t> wts_shape = {K};
  std::vector<size_t> aie_out_shape = a_shape;
  std::string dtype = "bfloat16";

  if ((std::find(supported_lengths.begin(), supported_lengths.end(), M) !=
       supported_lengths.end()) &&
      ENV_PARAM(USE_AIE_SSLRN) == 1) {
    auto output = ctx.GetOutput(0, dimensions_input); // Output activation

    auto sslrn_out_data = output.GetTensorMutableData<uint16_t>();
    if (num_outputs == 4) {

      auto skip_input_bias_add_output =
          ctx.GetOutput(3, dimensions_input); // Output activation

      auto skip_out_data =
          skip_input_bias_add_output.GetTensorMutableData<uint16_t>();

      std::vector<Tensor> add_input_Tensors;
      add_input_Tensors = {{(uint16_t*)in_data, a_shape, "bfloat16"},
                           {(uint16_t*)skip_data, a_shape, "bfloat16"}};

      std::vector<Tensor> add_output_Tensors;
      add_output_Tensors = {{(uint16_t*)skip_out_data, a_shape, "bfloat16"}};

      TRY_EXECUTE_WITH_LOG(add_->execute(add_input_Tensors, add_output_Tensors),
                           dry_run_, ReportInventory::getInstance().addData,
                           "mladf_add_" + std::to_string(M),
                           std::to_string(B) + "_" + std::to_string(M) + "_" +
                               std::to_string(K));

      std::vector<Tensor> rms_norm_input_Tensors;
      rms_norm_input_Tensors = {{(uint16_t*)skip_out_data, a_shape, "bfloat16"},
                                {(uint16_t*)wts_, wts_shape, "bfloat16"}};

      std::vector<Tensor> rms_norm_output_Tensors;
      rms_norm_output_Tensors = {
          {(uint16_t*)sslrn_out_data, a_shape, "bfloat16"}};

      // Execute RMS Norm
      TRY_EXECUTE_WITH_LOG(
          rms_norm_->execute(rms_norm_input_Tensors, rms_norm_output_Tensors),
          dry_run_, ReportInventory::getInstance().addData,
          "rms_norm_" + std::to_string(M),
          std::to_string(B) + "_" + std::to_string(M) + "_" +
              std::to_string(K));
    } else {

      std::vector<uint16_t> skip_input_bias_add_output(B * M * K);

      auto skip_out_data = skip_input_bias_add_output.data();

      std::vector<Tensor> add_input_Tensors;
      add_input_Tensors = {{(uint16_t*)in_data, a_shape, "bfloat16"},
                           {(uint16_t*)skip_data, a_shape, "bfloat16"}};

      std::vector<Tensor> add_output_Tensors;
      add_output_Tensors = {{(uint16_t*)skip_out_data, a_shape, "bfloat16"}};

      TRY_EXECUTE_WITH_LOG(add_->execute(add_input_Tensors, add_output_Tensors),
                           dry_run_, ReportInventory::getInstance().addData,
                           "mladf_add_" + std::to_string(M),
                           std::to_string(B) + "_" + std::to_string(M) + "_" +
                               std::to_string(K));

      std::vector<Tensor> rms_norm_input_Tensors;
      rms_norm_input_Tensors = {{(uint16_t*)skip_out_data, a_shape, "bfloat16"},
                                {(uint16_t*)wts_, wts_shape, "bfloat16"}};

      std::vector<Tensor> rms_norm_output_Tensors;
      rms_norm_output_Tensors = {
          {(uint16_t*)sslrn_out_data, a_shape, "bfloat16"}};

      // Execute RMS Norm
      TRY_EXECUTE_WITH_LOG(
          rms_norm_->execute(rms_norm_input_Tensors, rms_norm_output_Tensors),
          dry_run_, ReportInventory::getInstance().addData,
          "rms_norm_" + std::to_string(M),
          std::to_string(B) + "_" + std::to_string(M) + "_" +
              std::to_string(K));
    }

    MY_LOG(2) << "- AMD SSLRN Prefile phase ...\n";

  } else {

    // Get Output
    auto dimensions_out = dimensions_input;
    auto output = ctx.GetOutput(0, dimensions_out); // Output activation

    MY_LOG(2) << "- AMD SSLRN Token phase ...\n";

    // std::vector<float> input_a(B * M * K,
    //                            0.0); // Replace with actual input size and
    //                            data
    // std::vector<float> input_b(B * M * K, 0.0);
    // std::vector<float> output_1(B * M * K, 0.0);
    // std::vector<float> output_2(B * M * K, 0.0);
    size_t num_elements = B * M * K;
#ifdef _WIN32
    float* input_a = (float*)_aligned_malloc(num_elements * sizeof(float), 64);
    float* input_b = (float*)_aligned_malloc(num_elements * sizeof(float), 64);
    float* output_1 = (float*)_aligned_malloc(num_elements * sizeof(float), 64);
    float* output_2 = (float*)_aligned_malloc(num_elements * sizeof(float), 64);
#else
    float* input_a = (float*)aligned_alloc(64, num_elements * sizeof(float));
    float* input_b = (float*)aligned_alloc(64, num_elements * sizeof(float));
    float* output_1 = (float*)aligned_alloc(64, num_elements * sizeof(float));
    float* output_2 = (float*)aligned_alloc(64, num_elements * sizeof(float));
#endif

    // std::generate(input_tensor_values.begin(), input_tensor_values.end(),
    // std::rand);
    // vec_bf16_to_float(input_a, in_data, num_elements); // M x K
    // vec_bf16_to_float(input_b, skip_data,num_elements);

    bfloat16_to_float_avx512_unrolled(in_data, input_a, num_elements); // M x K

    bfloat16_to_float_avx512_unrolled(skip_data, input_b, num_elements);

    // Define input shape
    std::vector<int64_t> input_shape = {
        (int64_t)B, (int64_t)M,
        (int64_t)K}; // Replace with your actual input shape

    // Create memory info
    Ort::MemoryInfo memory_info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // Create input tensor
    Ort::Value input_tensor_a =
        Ort::Value::CreateTensor<float>(memory_info, input_a, num_elements,
                                        input_shape.data(), input_shape.size());
    Ort::Value input_tensor_b =
        Ort::Value::CreateTensor<float>(memory_info, input_b, num_elements,
                                        input_shape.data(), input_shape.size());
    Ort::Value output_tensor_1 =
        Ort::Value::CreateTensor<float>(memory_info, output_1, num_elements,
                                        input_shape.data(), input_shape.size());
    Ort::Value output_tensor_2 =
        Ort::Value::CreateTensor<float>(memory_info, output_2, num_elements,
                                        input_shape.data(), input_shape.size());
    const OrtValue* inputs[3] = {input_tensor_a, input_tensor_b, gamma};

    OrtValue* outputs[4] = {output_tensor_1, nullptr, nullptr, output_tensor_2};
    op_k.Invoke(context, inputs, 3, outputs, 4);

    auto output_token_tensor =
        ctx.GetOutput(0, dimensions_input); // Output activation
    auto sslrn_out_data_token =
        output_token_tensor.GetTensorMutableData<uint16_t>();

    float_to_bfloat16_avx512_unrolled(output_1, sslrn_out_data_token,
                                      M * K); // M x K
    if (num_outputs == 4) {
      auto skip_input_bias_add_output_token =
          ctx.GetOutput(3, dimensions_input); // Output activation
      auto skip_out_data_token =
          skip_input_bias_add_output_token.GetTensorMutableData<uint16_t>();

      float_to_bfloat16_avx512_unrolled(output_2, skip_out_data_token,
                                        M * K); // M x K
    }
#ifdef _WIN32
    _aligned_free(input_a);
    _aligned_free(input_b);
    _aligned_free(output_1);
    _aligned_free(output_2);
#else
    free(input_a);
    free(input_b);
    free(output_1);
    free(output_2);
#endif
  }

  MY_LOG(2) << "- AMD SSLRN compute done ...\n";
}

} // namespace ort_sslrn_custom_op