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

#include <glog/logging.h>
#include <sstream>

#include "custom_op_rope.hpp"
#include "reporter.hpp"
#include "vitis/ai/profiling.hpp"

DEF_ENV_PARAM(DEBUG_ROPE_CUSTOM_OP, "0")
DEF_ENV_PARAM(USE_AIE_ROPE, "1")
DEF_ENV_PARAM_2(MLADF_VERSION, "v1", std::string)
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_ROPE_CUSTOM_OP) >= n)
DEF_ENV_PARAM(DRY_RUN, "0")
namespace ort_rope_custom_op {
// Custom Op Domain

std::once_flag MyCustomOpKernel::initFlag;

inline void read_bin_file(std::string filename, char* data) {
  std::ifstream file(filename, std::ios::binary);

  // Check if the file is opened successfully
  if (!file.is_open()) {
    std::cerr << "Error opening file." << std::endl;
    // return 1;
  }

  // Get the file size
  file.seekg(0, std::ios::end);
  std::streampos fileSize = file.tellg();
  file.seekg(0, std::ios::beg);
  file.read(data, fileSize);
}

static float bfloat16_to_float_single(uint16_t v) {
  union {
    uint32_t i;
    float f;
  } u;
  u.i = (uint32_t(v)) << 16;
  return u.f;
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

void MyCustomOpKernel::LazyInit() {
  dry_run_ = 0;
  if (ENV_PARAM(DRY_RUN) == 1)
    dry_run_ = 1;
  std::string mladf_version_ = ENV_PARAM(MLADF_VERSION);
  std::string transpose_type = "all";
  std::map<std::string, std::any> attr = {{"op_version", mladf_version_},
                                          {"transpose", transpose_type}};

  if (mladf_version_ != "v0" && mladf_version_ != "v1") {
    std::cerr << "Invalid version. Supported versions are v0 and v1"
              << std::endl;
  }

  // std::cout << "ROPE MLADF_VERSION: " << mladf_version_ << std::endl;

  static ryzenai::mha_rope<uint16_t, uint16_t, uint16_t> mha_rope =
      ryzenai::mha_rope<uint16_t, uint16_t, uint16_t>("bfloat16", true, attr);

  if (rope_ == nullptr) {
    rope_ = &mha_rope;
  }
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

MyCustomOpKernel::MyCustomOpKernel(const OrtKernelInfo* k_info,
                                   const OrtApi& api) {
  api_ = &api;
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

  const char* add_type_constraint_names[2] = {"T", "M"};
  ONNXTensorElementDataType add_type_constraint_values[2] = {
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64};

  int64_t interleaved = 0;
  auto attr_interleaved = Ort::OpAttr("interleaved", &interleaved, 1,
                                      OrtOpAttrType::ORT_OP_ATTR_INT);
  Ort::OpAttr k_attrs[1] = {std::move(attr_interleaved)};

  // Create OP to call the appropriate built in operator
  // In this use case, we are using an ONNX Contrib OP from Microsoft

  op_k = Ort::Op::Create(info.Copy(), "RotaryEmbedding", "com.microsoft", 1,
                         add_type_constraint_names, add_type_constraint_values,
                         2, k_attrs, 1, 4,
                         1); // 1 attributes, 5 inputs, 4 outputs
  if (ENV_PARAM(USE_AIE_ROPE) == 1) {
    LazyInit();

    // Get weights
    int is_constant = 0;
    const_cos_ = info.GetTensorConstantInput(2, &is_constant); // 4096x64
    const float* wts_cos = const_cos_.GetTensorData<float>();
    auto cos_shape = const_cos_.GetTensorTypeAndShapeInfo().GetShape();
    int shape_cs_0 = cos_shape[0];
    int shape_cs_1 = cos_shape[1];

    is_constant = 0;
    const_sin_ = info.GetTensorConstantInput(3, &is_constant);
    const float* wts_sin = const_sin_.GetTensorData<float>(); // 4096x64

    std::vector<float> cos_embed(shape_cs_0 * shape_cs_1 * 2);
    std::vector<float> sin_embed(shape_cs_0 * shape_cs_1 * 2);

    int newK = 2 * shape_cs_1;
    // duplicate second dimention for cos and sin cache
    for (int i = 0; i < shape_cs_0; ++i) {
      for (int j = 0; j < shape_cs_1; ++j) {

        float cos_val = wts_cos[i * shape_cs_1 + j];
        float sin_val = wts_sin[i * shape_cs_1 + j];

        // Duplicate the values for second half
        cos_embed[i * newK + j] = cos_val;
        cos_embed[i * newK + j + shape_cs_1] = cos_val;

        sin_embed[i * newK + j] = sin_val;
        sin_embed[i * newK + j + shape_cs_1] = sin_val;
      }
    }

    max_seq_length = shape_cs_0;
    cs_1 = shape_cs_1;
#ifdef _WIN32
    trig_max_len = (uint16_t*)_aligned_malloc(
        2 * max_seq_length * (2 * shape_cs_1) * sizeof(uint16_t), 64);
#else
    trig_max_len = (uint16_t*)aligned_alloc(
        64, 2 * max_seq_length * (2 * shape_cs_1) * sizeof(uint16_t));
#endif

    size_t offset = 0;
    auto cos_element_num = max_seq_length * shape_cs_1;

    float_to_bfloat16_avx512_unrolled(cos_embed.data() + offset, trig_max_len,
                                      cos_element_num * 2);
    float_to_bfloat16_avx512_unrolled(sin_embed.data() + offset,
                                      trig_max_len + 2 * cos_element_num,
                                      cos_element_num * 2);

    MY_LOG(2) << "initialization for ROPE aie custom-op done." << std::endl;
  }
  MY_LOG(2) << "initialization for onnx transpose ROPE builtin op..."
            << std::endl;

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
}

MyCustomOpKernel::~MyCustomOpKernel() {
  if (trig_max_len)
#ifdef _WIN32
    _aligned_free(trig_max_len);
#else
    free(trig_max_len);
#endif
}

void MyCustomOpKernel::Compute(OrtKernelContext* context) {
  MY_LOG(2) << "- AMD ROPE compute start ...\n";
  Ort::KernelContext ctx(context);

  // Extracting the input and output information
  auto input_tensor_count = ctx.GetInputCount();

  auto input_tensor = ctx.GetInput(0);
  auto posid_tensor = ctx.GetInput(1);

  // std::cout<<"####### rope custom op Compute 1"<<std::endl;
  // get tensor data ptrs
  auto input_data = input_tensor.GetTensorData<uint16_t>();
  auto posid_data = posid_tensor.GetTensorData<int64_t>();

  // std::cout<<"####### rope custom op Compute 2"<<std::endl;
  // get tensor shapes
  auto input_shape = input_tensor.GetTensorTypeAndShapeInfo().GetShape();
  auto posid_shape = posid_tensor.GetTensorTypeAndShapeInfo().GetShape();

  size_t B = input_shape[2] / 128;
  size_t M = input_shape[1];
  size_t K = input_shape[2] / 32;

  auto in_element_num = B * M * K;
  auto cos_element_num = M * K;

  int batch = input_shape[0]; // batch
  int S = input_shape[1];     // sequence length
  int N = 32;                 // num_heads
  int H = input_shape[2] / N; // head size

  // get output tensors
  std::vector<int64_t> out_shape;
  for (unsigned i = 0; i < input_shape.size(); i++)
    out_shape.push_back(input_shape[i]);

  auto output_tensor = ctx.GetOutput(0, {out_shape.begin(), out_shape.end()});
  auto out = output_tensor.GetTensorMutableData<uint16_t>();

  if (M > 1) {
    prefil_m = M;
    token_counter = 0;
  }

  // Disable M=1 support due to performance degradation
  if (((M == 2048) || (M == 1024) || (M == 512) || (M == 256) || (M == 128)) &&
      ENV_PARAM(USE_AIE_ROPE) == 1) {

    std::vector<size_t> in_shape{B, M, K};
    rope_->set_params("rope", in_shape);

    std::vector<xrt::bo> rope_inbos;
    std::vector<xrt::bo> rope_outbos;
    rope_inbos = rope_->get_inputs();
    rope_outbos = rope_->get_outputs();

    uint16_t* a_bo_map = rope_inbos[0].map<uint16_t*>();
    memcpy((void*)a_bo_map, (uint16_t*)input_data,
           B * M * K * sizeof(uint16_t));
    uint16_t* b_bo_map = rope_inbos[1].map<uint16_t*>();

    auto trig_max_len_offset = 2 * max_seq_length * cs_1;
    auto b_bo_map_offset = M * K;

    // End position Offset for sin and cos when M = 1
    auto pos_offset = 0;
    if (M == 1)
      pos_offset = (prefil_m + token_counter) * K;

    memcpy((void*)b_bo_map, (void*)(trig_max_len + pos_offset),
           M * K * sizeof(uint16_t));
    memcpy((void*)(b_bo_map + b_bo_map_offset),
           (void*)(trig_max_len + pos_offset + trig_max_len_offset),
           M * K * sizeof(uint16_t));

    uint16_t* rope_out = rope_outbos[0].map<uint16_t*>();

    rope_inbos[0].sync(XCL_BO_SYNC_BO_TO_DEVICE);
    rope_inbos[1].sync(XCL_BO_SYNC_BO_TO_DEVICE);

    TRY_EXECUTE_WITH_LOG(
        rope_->execute(rope_inbos, rope_outbos), dry_run_,
        ReportInventory::getInstance().addData, "mha_rope_" + std::to_string(M),
        std::to_string(B) + "_" + std::to_string(M) + "_" + std::to_string(K));

    rope_outbos[0].sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    memcpy(out, rope_out, N * S * H * sizeof(uint16_t));

    if (M == 1)
      token_counter += 1;

    MY_LOG(2) << "- AMD Prefill ROPE compute done\n";
  } else {
    size_t batch = input_shape[0];
    size_t seq_lentgh = input_shape[1];
    size_t K_new = input_shape[2];

    auto cos_cache = ctx.GetInput(2);
    auto sin_cache = ctx.GetInput(3);

    size_t num_elements = batch * seq_lentgh * K_new;

#ifdef _WIN32
    float* input_fl = (float*)_aligned_malloc(num_elements * sizeof(float), 64);
    float* output_fl =
        (float*)_aligned_malloc(num_elements * sizeof(float), 64);
#else
    float* input_fl = (float*)aligned_alloc(64, num_elements * sizeof(float));
    float* output_fl = (float*)aligned_alloc(64, num_elements * sizeof(float));
#endif
    std::vector<int64_t> input_shape = {
        (int64_t)batch, (int64_t)seq_lentgh,
        (int64_t)K_new}; // Replace with your actual input shape

    bfloat16_to_float_avx512_unrolled(input_data, input_fl,
                                      batch * seq_lentgh * K_new);
    // Create memory info
    Ort::MemoryInfo memory_info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // Create input tensor
    Ort::Value input_tensor_a =
        Ort::Value::CreateTensor<float>(memory_info, input_fl, num_elements,
                                        input_shape.data(), input_shape.size());
    // Create output tensor
    Ort::Value output_tensor =
        Ort::Value::CreateTensor<float>(memory_info, output_fl, num_elements,
                                        input_shape.data(), input_shape.size());

    const OrtValue* inputs[4] = {input_tensor_a, posid_tensor, cos_cache,
                                 sin_cache};
    OrtValue* outputs[1] = {output_tensor};
    op_k.Invoke(context, inputs, 4, outputs, 1);
    float_to_bfloat16_avx512_unrolled(output_fl, out,
                                      batch * seq_lentgh * K_new); // M x K
#ifdef _WIN32
    _aligned_free(input_fl);
    _aligned_free(output_fl);
#else
    free(input_fl);
    free(output_fl);
#endif
  }
  MY_LOG(2) << "- AMD ROPE compute done ...\n";
}

} // namespace ort_rope_custom_op