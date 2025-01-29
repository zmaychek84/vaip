/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */
#include "custom_op_prefill_gqa.hpp"

#include <cmath>
#include <thread>
#include <utility>

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include <fstream>
#include <glog/logging.h>

#include "./reporter.hpp"
#include "vitis/ai/profiling.hpp"

DEF_ENV_PARAM(DEBUG_MHA_CUSTOM_OP, "0")
DEF_ENV_PARAM(USE_AIE_GQA, "1")
DEF_ENV_PARAM(USE_AIE_RoPE, "0")
DEF_ENV_PARAM(MHA_PARALLEL_BATCH, "1")
DEF_ENV_PARAM(DRY_RUN, "0")
DEF_ENV_PARAM_2(MLADF_VERSION, "v1", std::string)
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_MHA_CUSTOM_OP) >= n)

namespace ort_prefill_gqa_custom_op {
// Custom Op Domain
bool PrefillGQACustomOpKernel::is_const_cache = true;
int PrefillGQACustomOpKernel::instances__ = 0;
int PrefillGQACustomOpKernel::compute_instances__ = 0;
std::shared_ptr<uint16_t> PrefillGQACustomOpKernel::cos_sin_cache__ = nullptr;
int PrefillGQACustomOpKernel::fused_env_param = 1;
std::once_flag PrefillGQACustomOpKernel::initFlag;

uint16_t* PrefillAttenMaskProvider::get_atten_mask(int32_t S, int32_t S_pad,
                                                   int32_t past_S,
                                                   int32_t kv_size) {
  // if S is in aie support list, try to get LUT first.
  auto is_supported =
      aie_kernel_info_->is_seq_aie_supported(past_S + kv_size); // TODO
  if (is_supported && attn_mask_lut_->hasLut(past_S + kv_size)) {
    assert(attn_mask_lut_ != nullptr);
    MY_LOG(2) << "Use atten_mask LUT for current S: " << past_S + kv_size;
    return attn_mask_lut_->getLut(past_S + kv_size);
  } else {
    MY_LOG(2) << "Construct atten_mask on the fly for current S: " << S << " "
              << S_pad << " " << past_S << " " << kv_size;
    // otherwise construct the LUT on the fly.
    // Todo(ltp): consider case when b != 1;
    auto size = 1 * 1 * S_pad * (past_S + kv_size);
    auto bf16_attention_mask =
        (uint16_t*)allocator_.Alloc(size * sizeof(uint16_t));
    fill_attn_mask_3d(bf16_attention_mask, S, S_pad, past_S, kv_size);
    free_list_.push_back(bf16_attention_mask);

    return bf16_attention_mask;
  }
}

void* PrefillGQA_Allocator::get_buffer(size_t sz, BufferInfo& buffer) {
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

void PrefillGQACustomOpKernel::get_rope_cache(Ort::ConstValue& cos_tensor,
                                              Ort::ConstValue& sin_tensor) {
  const float* cos_data = cos_tensor.GetTensorData<float>();
  auto cos_shape = cos_tensor.GetTensorTypeAndShapeInfo().GetShape();
  int shape_cs_0 = cos_shape[0];
  int shape_cs_1 = cos_shape[1];

  const float* sin_data = sin_tensor.GetTensorData<float>();

  std::vector<float> cos_embed(shape_cs_0 * shape_cs_1 * 2);
  std::vector<float> sin_embed(shape_cs_0 * shape_cs_1 * 2);
  int newK = 2 * shape_cs_1;
  // duplicate second dimention for cos and sin cache
  for (int i = 0; i < shape_cs_0; ++i) {
    for (int j = 0; j < shape_cs_1; ++j) {

      float cos_val = cos_data[i * shape_cs_1 + j];
      float sin_val = sin_data[i * shape_cs_1 + j];

      cos_embed[i * newK + j] = cos_val;
      cos_embed[i * newK + j + shape_cs_1] = cos_val;
      sin_embed[i * newK + j] = sin_val;
      sin_embed[i * newK + j + shape_cs_1] = sin_val;
    }
  }
  max_seq_length = shape_cs_0;
  cs_1 = shape_cs_1;

#ifdef _WIN32
  cos_sin_cache__ = std::shared_ptr<uint16_t>(
      (uint16_t*)_aligned_malloc(
          2 * max_seq_length * (2 * shape_cs_1) * sizeof(uint16_t), 64),
      [](void* ptr) { _aligned_free(ptr); });
#else
  cos_sin_cache__ = std::shared_ptr<uint16_t>(
      (uint16_t*)aligned_alloc(64, 2 * max_seq_length * (2 * shape_cs_1) *
                                       sizeof(uint16_t)),
      [](void* ptr) { free(ptr); });
#endif
  size_t offset = 0;
  auto cos_element_num = max_seq_length * shape_cs_1;

  float_to_bfloat16_avx512_unrolled(cos_embed.data() + offset,
                                    cos_sin_cache__.get(), cos_element_num * 2);
  float_to_bfloat16_avx512_unrolled(sin_embed.data() + offset,
                                    cos_sin_cache__.get() + 2 * cos_element_num,
                                    cos_element_num * 2);
}

// Transpose [0, 1, 2, 3] to [0, 2, 1, 3] using ORT builtin kernel
void PrefillGQACustomOpKernel::transpose0213(uint16_t* output_data,
                                             uint16_t* input_data, int D0,
                                             int D1, int D2, int D3,
                                             OrtKernelContext* context) {
  for (int i0 = 0; i0 < D0; i0++) {
    size_t i0_idx = i0 * D1 * D2 * D3;
    for (int i1 = 0; i1 < D1; i1++) {
      for (int i2 = 0; i2 < D2; i2++) {
        memcpy((void*)(output_data + i0_idx + i2 * D1 * D3 + i1 * D3),
               (void*)(input_data + i0_idx + i1 * D2 * D3 + i2 * D3),
               D3 * sizeof(uint16_t));
      }
    }
  }
}

inline bool check_prefill(int seq_len) { return (seq_len != 1); }

// Try pad qkv from [B, N, S, H] to [B, N, S_pad, H] with 0
void try_pad_qkv(uint16_t* dst, uint16_t* src, int64_t N, int64_t S,
                 int64_t S_pad, int64_t H) {
  if (S != S_pad) {
    std::memset(dst, 0, N * S_pad * H * sizeof(uint16_t));
    for (int64_t n = 0; n < N; n++) {
      std::memcpy(dst + n * S_pad * H, src + n * S * H,
                  S * H * sizeof(uint16_t));
    }
  } else {
    std::memcpy(dst, src, N * S * H * sizeof(uint16_t));
  }
}

/// @brief built in rope
void PrefillGQACustomOpKernel::RoPE(float* output_data, float* input_data,
                                    int64_t* pos_ids_data,
                                    const OrtValue* cos_cache,
                                    const OrtValue* sin_cache, int B, int N,
                                    int S, int H, OrtKernelContext* context) {
  int tensor_size = B * N * S * H;
  std::vector<int64_t> tensor_shape{B, N, S, H};
  std::vector<int64_t> pos_ids_shape{B, S};
  Ort::MemoryInfo info =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
      info, input_data, tensor_size, tensor_shape.data(), tensor_shape.size());
  Ort::Value pos_ids = Ort::Value::CreateTensor<int64_t>(
      info, pos_ids_data, B * S, pos_ids_shape.data(), pos_ids_shape.size());
  Ort::Value output_tensor = Ort::Value::CreateTensor<float>(
      info, output_data, tensor_size, tensor_shape.data(), tensor_shape.size());
  const OrtValue* inputs[4] = {input_tensor, pos_ids, cos_cache, sin_cache};
  MY_LOG(2) << "RoPE input init done." << std::endl;
  OrtValue* outputs[1] = {output_tensor};
  try {
    if (N == num_heads_) {
      MY_LOG(2) << "Invoke rope_built_in_q." << std::endl;
      rope_built_in_q.Invoke(context, inputs, 4, outputs, 1);
    } else {
      MY_LOG(2) << "Invoke rope_built_in_k." << std::endl;
      rope_built_in_k.Invoke(context, inputs, 4, outputs, 1);
    }
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
  }

  MY_LOG(2) << "RoPE done." << std::endl;
}

///
/// @brief concat past and current, then pad (B = 1)
///
/// @param dst      : shape [B, N_kv, T_pad, H]
/// @param past     : shape [B, N_kv, past_S, H]
/// @param current  : shape [B, N_kv, chunk_sz, H]
/// @param share_buffer: past_present_share_buffer flag
///
/// @note
/// 1. concat pask_v with current_v to total_v:
///       [B, N_kv, past_S, H] + [B, N_kv,chunk_sz, H] -> [B, N_kv, T, H], where
///       T = past_S + chunk_sz
/// 2. pad 3rd dim of total_v to multiples of 128:
///       [B, N_kv, T, H] to [B, N_kv, T_pad, H], where T_pad is multiples of
///       128
///
///
void pad_concat_kv(uint16_t* dst, uint16_t* past, uint16_t* current, int N_kv,
                   int past_S, int T_pad, int H, bool share_buffer,
                   int T_buffer, int chunk_sz = 1) {
  // std::memset(dst, 0, N_kv * (past_S+ chunk_sz) * H * sizeof(uint16_t));
  /// if past_presenst_share_buffer is true, S stride will be 4096
  int past_s_stride = share_buffer ? T_buffer : past_S;
  for (int n = 0; n < N_kv; n++) {
    std::memcpy(dst + n * T_pad * H, past + n * past_s_stride * H,
                past_S * H * sizeof(uint16_t));
    // pad v chunk and concat new v with past v
    std::memcpy(dst + n * T_pad * H + past_S * H, current + n * chunk_sz * H,
                chunk_sz * H * sizeof(uint16_t));
  }
}

void save_present_kv(uint16_t* dst, uint16_t* past, uint16_t* current, int N_kv,
                     int past_S, int H, int T_buffer, int chunk_sz = 1) {
  for (int n = 0; n < N_kv; n++) {
    std::memcpy(dst + n * T_buffer * H, past + n * T_buffer * H,
                past_S * H * sizeof(uint16_t));
    std::memcpy(dst + n * T_buffer * H + past_S * H, current + n * chunk_sz * H,
                chunk_sz * H * sizeof(uint16_t));
  }
}

/// For ChatGLM3-6b, updated M = 3072 /////////
void PrefillGQACustomOpKernel::set_params() {
  std::vector<size_t> a_shape_1 = {32, MAX_SEQ_LENGTH, 128};
  std::vector<size_t> w_shape_1 = {32, 128, MAX_SEQ_LENGTH};
  bmm1_->set_params("BMM", a_shape_1, w_shape_1);
  std::vector<size_t> a_shape_2 = {32, MAX_SEQ_LENGTH, MAX_SEQ_LENGTH};
  std::vector<size_t> w_shape_2 = {32, MAX_SEQ_LENGTH, 128};
  // bmm2_->set_params("BMM", a_shape_2, w_shape_2);
  std::map<std::string, std::any> attr = {{"op_version", mladf_version_},
                                          {"skip_create_input_a", 1}};
  bmm2_->set_params("BMM", a_shape_2, w_shape_2, attr);
}

void PrefillGQACustomOpKernel::LazyInit() {
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
      {"op_version", mladf_version_},
      {"headsize", headsize_}};
  static ryzenai::masked_softmax<uint16_t, uint16_t, uint16_t> softmax =
      ryzenai::masked_softmax<uint16_t, uint16_t, uint16_t>("bfloat16", true,
                                                            attr_softmax);
  static ryzenai::bmm<uint16_t, uint16_t, uint16_t> bmm2 =
      ryzenai::bmm<uint16_t, uint16_t, uint16_t>("bfloat16", "bfloat16",
                                                 "bfloat16", true, false, attr);
  std::string transpose_type = "input";
  std::map<std::string, std::any> rope_attr = {{"op_version", mladf_version_},
                                               {"transpose", transpose_type}};
  if (rotary_interleaved_) {
    std::string modelname = "CHATGLM";
    rope_attr["model_name"] = modelname;
  }
  static ryzenai::mha_rope<uint16_t, uint16_t, uint16_t> mha_rope_q =
      ryzenai::mha_rope<uint16_t, uint16_t, uint16_t>("bfloat16", true,
                                                      rope_attr);
  static ryzenai::mha_rope<uint16_t, uint16_t, uint16_t> mha_rope_k =
      ryzenai::mha_rope<uint16_t, uint16_t, uint16_t>("bfloat16", true,
                                                      rope_attr);
  if (rope_q_ == nullptr) {
    rope_q_ = &mha_rope_q;
  }

  if (rope_k_ == nullptr) {
    rope_k_ = &mha_rope_k;
  }

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

PrefillGQACustomOpKernel::PrefillGQACustomOpKernel(const OrtKernelInfo* k_info,
                                                   const OrtApi& api) {
  // Get constant info for the node
  Ort::ConstKernelInfo info{k_info};

  // Get Logger
  m_logger = info.GetLogger();

  int is_cos_const = 0;
  Ort::ConstValue const_cos = info.GetTensorConstantInput(7, &is_cos_const);
  int is_sin_const = 0;
  Ort::ConstValue const_sin = info.GetTensorConstantInput(8, &is_sin_const);
  MY_LOG(2) << "is_cos_const: " << is_cos_const << std::endl;
  MY_LOG(2) << "is_sin_const: " << is_sin_const << std::endl;

  InitializeParams(api, info, const_cos, const_sin, is_cos_const, is_sin_const);
}

void PrefillGQACustomOpKernel::InitializeParams(
    const OrtApi& api, Ort::ConstKernelInfo& info, Ort::ConstValue const_cos,
    Ort::ConstValue const_sin, int is_cos_const, int is_sin_const) {
  api_ = &api;

  do_rotary_ = info.GetAttribute<int64_t>("do_rotary");
  kv_num_heads_ = info.GetAttribute<int64_t>("kv_num_heads");
  num_heads_ = info.GetAttribute<int64_t>("num_heads");
  rotary_interleaved_ = info.GetAttribute<int64_t>("rotary_interleaved");
  scale_ = info.GetAttribute<float>("scale");
  //  Get inputs attrs
  m_node_name = info.GetNodeName();

  const_cos_ = const_cos;
  const_sin_ = const_sin;

  /// check if cos/sin cache is const tensor
  if (is_cos_const == 0 || is_sin_const == 0) {
    is_const_cache = false;
  }
  /// if cos/sin cache is const tensor, then set rotary_embedding_dim to
  /// cos_shape[1] * 2, we need to pass this param to ort RoPE kernel.
  /// if not, set it to defalut value 0.
  /// Also set headsize for softmax dd operator.
  int64_t rotary_embedding_dim = 0;
  if (is_const_cache) {
    auto cos_shape = const_cos_.GetTensorTypeAndShapeInfo().GetShape();
    rotary_embedding_dim = cos_shape[1] * 2;
    headsize_ = 2 * cos_shape[1];
    if (rotary_interleaved_)
      headsize_ *= 2;
  } else
    headsize_ = 96; // hardcode for phi3.5

  /// ort built in RotaryEmbedding kernel
  MY_LOG(2) << "initialization for onnx rope builtin op..." << std::endl;
  const char* rope_type_constraint_names[2] = {"T", "M"};
  ONNXTensorElementDataType rope_type_constraint_values[2] = {
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64};
  // scale
  int64_t val_scale_rope_q = scale_;
  auto attr_scale_rope_q = Ort::OpAttr("scale", &val_scale_rope_q, 1,
                                       OrtOpAttrType::ORT_OP_ATTR_FLOAT);
  // interleaved
  int64_t val_interleaved_q = rotary_interleaved_;
  MY_LOG(2) << "val_interleaved_q: " << val_interleaved_q << std::endl;
  auto attr_interleaved_q = Ort::OpAttr("interleaved", &val_interleaved_q, 1,
                                        OrtOpAttrType::ORT_OP_ATTR_INT);
  // num_heads for q
  int64_t val_num_heads_q = num_heads_;
  auto attr_num_heads_q = Ort::OpAttr("num_heads", &val_num_heads_q, 1,
                                      OrtOpAttrType::ORT_OP_ATTR_INT);
  int64_t val_rotary_embedding_dim_q = rotary_embedding_dim;
  auto attr_rotary_embedding_dim_q =
      Ort::OpAttr("rotary_embedding_dim", &val_rotary_embedding_dim_q, 1,
                  OrtOpAttrType::ORT_OP_ATTR_INT);
  Ort::OpAttr rope_attrs_q[4] = {
      std::move(attr_scale_rope_q), std::move(attr_interleaved_q),
      std::move(attr_num_heads_q), std::move(attr_rotary_embedding_dim_q)};
  rope_built_in_q =
      Ort::Op::Create(info.Copy(), "RotaryEmbedding", "com.microsoft", 1,
                      rope_type_constraint_names, rope_type_constraint_values,
                      2, rope_attrs_q, 4, 4,
                      1); // 4 attributes, 4 inputs, 1 outputs
  // scale
  int64_t val_scale_rope_k = scale_;
  auto attr_scale_rope_k = Ort::OpAttr("scale", &val_scale_rope_k, 1,
                                       OrtOpAttrType::ORT_OP_ATTR_FLOAT);
  // interleaved
  int64_t val_interleaved_k = rotary_interleaved_;
  MY_LOG(2) << "val_interleaved_k: " << val_interleaved_k << std::endl;
  auto attr_interleaved_k = Ort::OpAttr("interleaved", &val_interleaved_k, 1,
                                        OrtOpAttrType::ORT_OP_ATTR_INT);
  // num_heads for k
  int64_t val_num_heads_k = kv_num_heads_;
  auto attr_num_heads_k = Ort::OpAttr("num_heads", &val_num_heads_k, 1,
                                      OrtOpAttrType::ORT_OP_ATTR_INT);
  int64_t val_rotary_embedding_dim_k = rotary_embedding_dim;
  auto attr_rotary_embedding_dim_k =
      Ort::OpAttr("rotary_embedding_dim", &val_rotary_embedding_dim_k, 1,
                  OrtOpAttrType::ORT_OP_ATTR_INT);
  Ort::OpAttr rope_attrs_k[4] = {
      std::move(attr_scale_rope_k), std::move(attr_interleaved_k),
      std::move(attr_num_heads_k), std::move(attr_rotary_embedding_dim_k)};
  rope_built_in_k =
      Ort::Op::Create(info.Copy(), "RotaryEmbedding", "com.microsoft", 1,
                      rope_type_constraint_names, rope_type_constraint_values,
                      2, rope_attrs_k, 4, 4,
                      1); // 4 attributes, 4 inputs, 1 outputs
  MY_LOG(2) << "initialization for onnx rope builtin op done..." << std::endl;

  // aie mha op from DD
  MY_LOG(2) << "initialization for mha aie custom-op..." << std::endl;

  if (ENV_PARAM(USE_AIE_GQA) == 1 && fused_env_param) {
    LazyInit();
    std::call_once(initFlag, [this]() { this->set_params(); });
    bmm1_->debug(false);
    bmm2_->debug(false);
    softmax_->debug(false);
    rope_q_inputs = rope_q_->get_inputs();
    rope_q_outputs = rope_q_->get_outputs();
    rope_k_inputs = rope_k_->get_inputs();
    rope_k_outputs = rope_k_->get_outputs();
    bmm1_inputs = bmm1_->get_inputs();
    bmm1_outputs = bmm1_->get_outputs();
    bmm2_inputs = bmm2_->get_inputs();
    // skip creating input BO for first input of bmm2
    // reuse output BO of bmm1
    // this works due to softmax being able to operate in place
    // otherwise output of softmax would need to go to bmm2_in[0]
    // bmm1_in[0] ---> bmm1 -> bmm1_out[0] -> softmax -> bmm1_out[0] --> bmm2 ->
    // bmm2_out[0] bmm1_in[1] --/          mask -------/ bmm2_in[1]----/
    bmm2_inputs[0] = bmm1_outputs[0];
    bmm2_outputs = bmm2_->get_outputs();
    softmax_mask = softmax_->get_inputs()[1];

    // initialize the atten_provider
    atten_mask_provider_ =
        std::make_unique<PrefillAttenMaskProvider>(&mha_aie_kernel_info_);

    MY_LOG(2) << "initialization for mha aie custom-op done." << std::endl;

    if (ENV_PARAM(USE_AIE_RoPE) == 1 && is_const_cache) {
      MY_LOG(2) << "Getting cos/sin cache as constant input.\n";
      get_rope_cache(const_cos_, const_sin_);
    }
  }
}

PrefillGQACustomOpKernel::~PrefillGQACustomOpKernel() {
#ifdef __linux__
  ryzenai::dynamic_dispatch::xrt_context::destroy_ctx_map();
#endif
}

void PrefillGQACustomOpKernel::aie_execute_mha(
    uint16_t* output, const int N_q, const int N_kv, const int seq_len,
    const int seq_len_pad, const int S, const int S_pad, const int H) {
  MY_LOG(2) << "Running AIE MHA alone.";
  /// set kernel shapes
  std::vector<size_t> bmm1_shape_a{(size_t)N_q, (size_t)seq_len_pad, (size_t)H};
  std::vector<size_t> bmm1_shape_w{(size_t)N_kv, (size_t)H, (size_t)S_pad};
  std::vector<size_t> softmax_shape{(size_t)N_q, (size_t)seq_len_pad,
                                    (size_t)S_pad};
  std::vector<size_t> bmm2_shape_a{(size_t)N_q, (size_t)seq_len_pad,
                                   (size_t)S_pad};
  std::vector<size_t> bmm2_shape_w{(size_t)N_kv, (size_t)S_pad, (size_t)H};
  bmm1_->set_execute_kernel_shape(bmm1_shape_a, bmm1_shape_w);
  bmm2_->set_execute_kernel_shape(bmm2_shape_a, bmm2_shape_w);
  softmax_->set_params("softmax", softmax_shape);

  MY_LOG(2) << "AIE excute params: \n"
            << "  N_q: " << N_q << " N_kv: " << N_kv << "  S: " << S
            << " H: " << H << " S_pad: " << S_pad << " seq_len:" << seq_len
            << " seq_len_pad:" << seq_len_pad;

  // Sync data
  __TIC__(SyncBOs)
  bmm1_inputs[0].sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bmm1_inputs[1].sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bmm1_outputs[0].sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bmm2_outputs[0].sync(XCL_BO_SYNC_BO_TO_DEVICE);
  __TOC__(SyncBOs)

  __TIC__(RunBMM1)
#ifdef _WIN32
  // Execute QKT MatMul
  TRY_EXECUTE_WITH_LOG(bmm1_->execute(bmm1_inputs, bmm1_outputs, false),
                       dry_run_, ReportInventory::getInstance().addData,
                       "bmm_" + std::to_string(seq_len),
                       std::to_string(N_q) + "_" + std::to_string(S_pad) + "_" +
                           std::to_string(H));
#else
  TRY_EXECUTE_WITH_LOG(bmm1_->execute(bmm1_inputs, bmm1_outputs, true),
                       dry_run_, ReportInventory::getInstance().addData,
                       "bmm_" + std::to_string(seq_len),
                       std::to_string(N_q) + "_" + std::to_string(S_pad) + "_" +
                           std::to_string(H));
#endif
  __TOC__(RunBMM1)

  std::vector<xrt::bo> inputs = {bmm1_outputs[0], softmax_mask};
  std::vector<xrt::bo> outputs = {bmm2_inputs[0]};

  __TIC__(RunSFTMax)
// Execute Softmax
#ifdef _WIN32
  TRY_EXECUTE_WITH_LOG(softmax_->execute(inputs, outputs, false), dry_run_,
                       ReportInventory::getInstance().addData,
                       "masked_softmax_" + std::to_string(S_pad),
                       std::to_string(N_q) + "_" + std::to_string(S_pad) + "_" +
                           std::to_string(S_pad));
#else
  TRY_EXECUTE_WITH_LOG(softmax_->execute(inputs, outputs, true), dry_run_,
                       ReportInventory::getInstance().addData,
                       "masked_softmax_" + std::to_string(S_pad),
                       std::to_string(N_q) + "_" + std::to_string(S_pad) + "_" +
                           std::to_string(S_pad));
#endif
  __TOC__(RunSFTMax)

  __TIC__(RunBMM2)
  // Execute SMV MatMul
  TRY_EXECUTE_WITH_LOG(bmm2_->execute(bmm2_inputs, bmm2_outputs, true),
                       dry_run_, ReportInventory::getInstance().addData,
                       "bmm_" + std::to_string(seq_len),
                       std::to_string(N_q) + "_" + std::to_string(S_pad) + "_" +
                           std::to_string(S_pad));
  // Sync output
  bmm2_outputs[0].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  __TOC__(RunBMM2)

  __TIC__(CpyMHAOut)
  // Copy output from XRT BO to OrtTensor
  if (output != nullptr) {
    uint16_t* output_bo = bmm2_outputs[0].map<uint16_t*>();
    memcpy(output, output_bo, N_q * seq_len * H * sizeof(uint16_t));
  }
  __TOC__(CpyMHAOut)
}

void PrefillGQACustomOpKernel::aie_execute_rope_mha(
    uint16_t* output, const int N_q, const int N_kv, const int seq_len,
    const int seq_len_pad, const int S, const int S_pad, const int H,
    bool past_present_share_buffer, int T_buffer, DataPtrWrapper past_k_data,
    const int T, std::shared_future<void> cache_v,
    std::function<void(uint16_t*)> save_k_func) {
  MY_LOG(2) << "Running AIE RoPE MHA with BO sharing.";
  /// set kernel shapes
  std::vector<size_t> rope_q_shape{(size_t)N_q, (size_t)seq_len, (size_t)H};
  std::vector<size_t> rope_k_shape{(size_t)N_kv, (size_t)seq_len, (size_t)H};

  std::vector<size_t> bmm1_shape_a{(size_t)N_q, (size_t)seq_len, (size_t)H};
  std::vector<size_t> bmm1_shape_w{(size_t)N_kv, (size_t)H, (size_t)S_pad};
  std::vector<size_t> softmax_shape{(size_t)N_q, (size_t)seq_len,
                                    (size_t)S_pad};
  std::vector<size_t> bmm2_shape_a{(size_t)N_q, (size_t)seq_len, (size_t)S_pad};
  std::vector<size_t> bmm2_shape_w{(size_t)N_kv, (size_t)S_pad, (size_t)H};

  rope_q_->set_params("rope", rope_q_shape);
  rope_k_->set_params("rope", rope_k_shape);
  bmm1_->set_execute_kernel_shape(bmm1_shape_a, bmm1_shape_w);
  bmm2_->set_execute_kernel_shape(bmm2_shape_a, bmm2_shape_w);
  softmax_->set_params("softmax", softmax_shape);

  uint16_t* rope_q_cs_bo = rope_q_inputs[1].map<uint16_t*>();
  uint16_t* rope_k_cs_bo = rope_k_inputs[1].map<uint16_t*>();
  auto trig_max_len_offset = 2 * max_seq_length * cs_1;
  auto cs_bo_offset = seq_len * H;
  size_t pos_offset = (S_pad - seq_len) * H;
  MY_LOG(2) << "aie rope offset: " << pos_offset << std::endl;
  MY_LOG(2) << "aie rope cs_bo_offset: " << cs_bo_offset << std::endl;
  MY_LOG(2) << "aie rope tig mas len: " << max_seq_length << " " << cs_1
            << std::endl;

  __TIC__(CpySinCos)
  memcpy((void*)rope_q_cs_bo, (void*)((cos_sin_cache__.get() + pos_offset)),
         cs_bo_offset * sizeof(uint16_t));
  memcpy((void*)(rope_q_cs_bo + cs_bo_offset),
         (void*)(cos_sin_cache__.get() + trig_max_len_offset + pos_offset),
         cs_bo_offset * sizeof(uint16_t));

  memcpy((void*)rope_k_cs_bo, (void*)(cos_sin_cache__.get() + pos_offset),
         cs_bo_offset * sizeof(uint16_t));
  memcpy((void*)(rope_k_cs_bo + cs_bo_offset),
         (void*)(cos_sin_cache__.get() + trig_max_len_offset + pos_offset),
         cs_bo_offset * sizeof(uint16_t));
  __TOC__(CpySinCos)

  __TIC__(SyncBos)
  rope_q_inputs[0].sync(XCL_BO_SYNC_BO_TO_DEVICE);
  rope_q_inputs[1].sync(XCL_BO_SYNC_BO_TO_DEVICE);
  rope_k_inputs[0].sync(XCL_BO_SYNC_BO_TO_DEVICE);
  rope_k_inputs[1].sync(XCL_BO_SYNC_BO_TO_DEVICE);
  __TOC__(SyncBos)

#ifdef _WIN32
  __TIC__(RunRopeQ)
  TRY_EXECUTE_WITH_LOG(rope_q_->execute(rope_q_inputs, rope_q_outputs, false),
                       dry_run_, ReportInventory::getInstance().addData,
                       "mha_rope_" + std::to_string(seq_len),
                       std::to_string(N_q) + "_" + std::to_string(seq_len) +
                           "_" + std::to_string(H));
  __TOC__(RunRopeQ)
  __TIC__(RunRopeK)
  TRY_EXECUTE_WITH_LOG(rope_k_->execute(rope_k_inputs, rope_k_outputs, true),
                       dry_run_, ReportInventory::getInstance().addData,
                       "mha_rope_" + std::to_string(S_pad),
                       std::to_string(N_q) + "_" + std::to_string(S_pad) + "_" +
                           std::to_string(H));
  __TOC__(RunRopeK)
#else
  TRY_EXECUTE_WITH_LOG(rope_q_->execute(rope_q_inputs, rope_q_outputs, true),
                       dry_run_, ReportInventory::getInstance().addData,
                       "mha_rope_" + std::to_string(seq_len),
                       std::to_string(N_q) + "_" + std::to_string(seq_len) +
                           "_" + std::to_string(H));

  TRY_EXECUTE_WITH_LOG(rope_k_->execute(rope_k_inputs, rope_k_outputs, true),
                       dry_run_, ReportInventory::getInstance().addData,
                       "mha_rope_" + std::to_string(S_pad),
                       std::to_string(N_q) + "_" + std::to_string(S_pad) + "_" +
                           std::to_string(H));
#endif

  std::vector<xrt::bo> bmm1_shared_inputs = {rope_q_outputs[0],
                                             // rope_k_outputs[0]};
                                             bmm1_inputs[1]};
  cache_v.wait();
  __TIC__(PadConcatK)
  rope_k_outputs[0].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  std::shared_future<void> rst_save_k = std::async(
      std::launch::async, save_k_func, rope_k_outputs[0].map<uint16_t*>());
  pad_concat_kv(bmm1_inputs[1].map<uint16_t*>(), past_k_data.cast<uint16_t>(),
                rope_k_outputs[0].map<uint16_t*>(), N_kv, S - seq_len, S_pad, H,
                true, T_buffer, seq_len);
  bmm1_inputs[1].sync(XCL_BO_SYNC_BO_TO_DEVICE);
  __TOC__(PadConcatK)
#ifdef _WIN32
  // Execute QKT MatMul
  __TIC__(RunBMM1)
  TRY_EXECUTE_WITH_LOG(bmm1_->execute(bmm1_shared_inputs, bmm1_outputs, false),
                       dry_run_, ReportInventory::getInstance().addData,
                       "bmm_" + std::to_string(seq_len),
                       std::to_string(N_kv) + "_" + std::to_string(S_pad) +
                           "_" + std::to_string(H));
  __TOC__(RunBMM1)
#else
  TRY_EXECUTE_WITH_LOG(bmm1_->execute(bmm1_shared_inputs, bmm1_outputs, true),
                       dry_run_, ReportInventory::getInstance().addData,
                       "bmm_" + std::to_string(seq_len),
                       std::to_string(N_kv) + "_" + std::to_string(S_pad) +
                           "_" + std::to_string(H));
#endif

  std::vector<xrt::bo> sm_inputs = {bmm1_outputs[0], softmax_mask};
  std::vector<xrt::bo> sm_outputs = {bmm2_inputs[0]};

#ifdef _WIN32
  // Execute Softmax
  __TIC__(RunSftmax)
  TRY_EXECUTE_WITH_LOG(softmax_->execute(sm_inputs, sm_outputs, false),
                       dry_run_, ReportInventory::getInstance().addData,
                       "masked_softmax_" + std::to_string(S_pad),
                       std::to_string(N_q) + "_" + std::to_string(S_pad) + "_" +
                           std::to_string(S_pad));
  __TOC__(RunSftmax)
#else
  TRY_EXECUTE_WITH_LOG(softmax_->execute(sm_inputs, sm_outputs, true), dry_run_,
                       ReportInventory::getInstance().addData,
                       "masked_softmax_" + std::to_string(S_pad),
                       std::to_string(N_q) + "_" + std::to_string(S_pad) + "_" +
                           std::to_string(S_pad));
#endif
  // Execute SMV MatMul
  __TIC__(RunBMM2)
  TRY_EXECUTE_WITH_LOG(bmm2_->execute(bmm2_inputs, bmm2_outputs, true),
                       dry_run_, ReportInventory::getInstance().addData,
                       "bmm_" + std::to_string(seq_len),
                       std::to_string(N_q) + "_" + std::to_string(S_pad) + "_" +
                           std::to_string(S_pad));
  // Sync output

  __TOC__(RunBMM2)

  __TIC__(CpyBmmOut)
  bmm2_outputs[0].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  // Copy output from XRT BO to OrtTensor
  if (output != nullptr) {
    uint16_t* output_bo = bmm2_outputs[0].map<uint16_t*>();
    memcpy(output, output_bo, seq_len * N_q * H * sizeof(uint16_t));
  }
  __TOC__(CpyBmmOut)
  rst_save_k.wait();
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

bool isBf16Model(const DataPtrWrapper& packed_qkv, DataPtrWrapper& out,
                 DataPtrWrapper& present_key, DataPtrWrapper& present_value) {
  return packed_qkv.is_bf16() && out.is_bf16() && present_key.is_bf16() &&
         present_value.is_bf16();
}

struct SplitQKVData {
  /// qkv
  const uint16_t* packed_qkv = {};
  uint16_t* q = {};
  uint16_t* k = {};
  uint16_t* v = {};
  /// params
  const int N_q = 0;
  const int N_kv = 0;
  const int S = 0;
  const int H = 0;
};

void Split_QKV(void* raw_data, size_t n) {
  auto data = reinterpret_cast<SplitQKVData*>(raw_data);
  int N_q = data->N_q;
  int N_kv = data->N_kv;
  int S = data->S;
  int H = data->H;
  auto q_src_offset = n * (N_q + 2 * N_kv) * H;
  auto k_src_offset = q_src_offset + N_q * H;
  auto v_src_offset = k_src_offset + N_kv * H;
  auto q_dst_offset = n * N_q * H;
  auto k_dst_offset = n * N_kv * H;
  auto v_dst_offset = n * N_kv * H;
  std::memcpy(data->q + q_dst_offset, data->packed_qkv + q_src_offset,
              N_q * H * sizeof(uint16_t));
  std::memcpy(data->k + k_dst_offset, data->packed_qkv + k_src_offset,
              N_kv * H * sizeof(uint16_t));
  std::memcpy(data->v + v_dst_offset, data->packed_qkv + v_src_offset,
              N_kv * H * sizeof(uint16_t));
}

void PrefillGQACustomOpKernel::Compute(OrtKernelContext* context) {
  MY_LOG(2) << "- AMD GQA compute start ...\n";
  Ort::KernelContext ctx(context);
  MY_LOG(2) << "num_inputs " << ctx.GetInputCount() << " "
            << "num_outputs " << ctx.GetOutputCount() << " \n";

  // Prepare input/output tensors
  // Extracting the input and output information
  MY_LOG(2) << "Getting inputs...\n";
  Ort::ConstValue packed_qkv = ctx.GetInput(0);
  Ort::ConstValue key = ctx.GetInput(1);
  Ort::ConstValue value = ctx.GetInput(2);
  Ort::ConstValue past_k = ctx.GetInput(3);
  Ort::ConstValue past_v = ctx.GetInput(4);
  Ort::ConstValue seqlens_k = ctx.GetInput(5);
  Ort::ConstValue total_seqlen = ctx.GetInput(6);
  Ort::ConstValue cos_cache = ctx.GetInput(7);
  Ort::ConstValue sin_cache = ctx.GetInput(8);

  std::vector<Ort::ConstValue> inputs = {packed_qkv,   key,       value,
                                         past_k,       past_v,    seqlens_k,
                                         total_seqlen, cos_cache, sin_cache};
  auto qkv_data_type = packed_qkv.GetTensorTypeAndShapeInfo().GetElementType();
  auto qkv_shape = packed_qkv.GetTensorTypeAndShapeInfo().GetShape();
  auto past_k_shape = past_k.GetTensorTypeAndShapeInfo().GetShape();

  int batch_size = qkv_shape[0];
  int hidden_size = qkv_shape[2];
  int head_size = hidden_size / (num_heads_ + 2 * kv_num_heads_);

  int past_sequence_length = static_cast<int>(past_k_shape[2]);
  int total_seq_len = total_seqlen.GetTensorData<int32_t>()[0];
  int total_sequence_length = total_seq_len > past_sequence_length
                                  ? total_seq_len
                                  : past_sequence_length;

  std::vector<int64_t> present_k_shape(
      {static_cast<int64_t>(batch_size), static_cast<int64_t>(kv_num_heads_),
       static_cast<int64_t>(total_sequence_length),
       static_cast<int64_t>(head_size)});
  std::vector<int64_t> present_v_shape(
      {static_cast<int64_t>(batch_size), static_cast<int64_t>(kv_num_heads_),
       static_cast<int64_t>(total_sequence_length),
       static_cast<int64_t>(head_size)});

  Ort::UnownedValue present_k = ctx.GetOutput(1, present_k_shape);
  Ort::UnownedValue present_v = ctx.GetOutput(2, present_v_shape);

  std::vector<Ort::UnownedValue> gqa_outs = {present_k, present_v};
  uint16_t* temp_out = nullptr;
  std::vector<xrt::bo> temp_outputs =
      Execute(inputs, temp_out, gqa_outs, context, false);
}

std::vector<xrt::bo>
PrefillGQACustomOpKernel::Execute(std::vector<Ort::ConstValue>& inputs,
                                  uint16_t*& mm_inp,
                                  std::vector<Ort::UnownedValue>& gqa_out,
                                  OrtKernelContext* context, bool bo_sharing) {
  __TIC__(Compute)
  Ort::KernelContext ctx(context);
  auto packed_qkv = inputs[0];
  auto key = inputs[1];
  auto value = inputs[2];
  auto past_k = inputs[3];
  auto past_v = inputs[4];
  auto seqlens_k = inputs[5];
  auto total_seqlen = inputs[6];
  auto cos_cache = inputs[7];
  auto sin_cache = inputs[8];
  MY_LOG(2) << "Getting inputs shape and data...\n";
  bool is_packed_qkv = key == nullptr ? true : false;
  Ort::AllocatorWithDefaultOptions allocator;
  auto& PrefillGQA_Allocator = PrefillGQA_Allocator::get_instance();

  // when RoPE cache is dynamic, we get the cache from input when the first op
  // instance call Compute().
  if (ENV_PARAM(USE_AIE_RoPE) && !is_const_cache && compute_instances__ == 0) {
    MY_LOG(2) << "Getting cos/sin cache dynamicly.\n";
    compute_instances__++;
    get_rope_cache(cos_cache, sin_cache);
  }

  // Query data / shape
  auto qkv_data_type = packed_qkv.GetTensorTypeAndShapeInfo().GetElementType();
  auto qkv_shape = packed_qkv.GetTensorTypeAndShapeInfo().GetShape();
  auto qkv_size = packed_qkv.GetTensorTypeAndShapeInfo().GetElementCount();
  DataPtrWrapper qkv_data;
  GetInputTensorData(qkv_data, qkv_data_type, packed_qkv);
  int batch_size = qkv_shape[0];
  int seq_len = qkv_shape[1];
  int hidden_size = qkv_shape[2];
  int head_size = is_packed_qkv
                      ? (hidden_size / (num_heads_ + 2 * kv_num_heads_))
                      : (hidden_size / num_heads_);

  MY_LOG(2) << "packed qkv shape: " << shape2str(qkv_shape) << std::endl;
  MY_LOG(2) << "packed qkv size: " << qkv_size << std::endl;
  MY_LOG(2) << "Packed qkv data: " << qkv_data.toString();
  DataPtrWrapper k_data;
  DataPtrWrapper v_data;
  uint16_t* q_data_ptr = nullptr;
  uint16_t* k_data_ptr = nullptr;
  uint16_t* v_data_ptr = nullptr;
  size_t q_size = batch_size * num_heads_ * seq_len * head_size;
  size_t kv_size = batch_size * kv_num_heads_ * seq_len * head_size;
  MY_LOG(2) << "q_size: " << q_size;
  MY_LOG(2) << "kv_size: " << kv_size;
  if (is_packed_qkv) {
    __TIC__(SplitQKV)
    q_data_ptr = PrefillGQA_Allocator.get_buffer_generic<uint16_t>(
        q_size * sizeof(uint16_t), PrefillGQA_Allocator::BufferType::AIE_Q);
    k_data_ptr = PrefillGQA_Allocator.get_buffer_generic<uint16_t>(
        kv_size * sizeof(uint16_t), PrefillGQA_Allocator::BufferType::AIE_K);
    v_data_ptr = PrefillGQA_Allocator.get_buffer_generic<uint16_t>(
        kv_size * sizeof(uint16_t), PrefillGQA_Allocator::BufferType::AIE_V);
    SplitQKVData split_qkv_data = {qkv_data.cast<uint16_t>(),
                                   q_data_ptr,
                                   k_data_ptr,
                                   v_data_ptr,
                                   static_cast<int>(num_heads_),
                                   static_cast<int>(kv_num_heads_),
                                   seq_len,
                                   head_size};

    MY_LOG(2) << "ParallelFor Split_QKV.";
    ctx.ParallelFor(Split_QKV, static_cast<size_t>(seq_len), 0,
                    &split_qkv_data);
    __TOC__(SplitQKV)
  } else {
    auto k_data_type = key.GetTensorTypeAndShapeInfo().GetElementType();
    auto k_shape = key.GetTensorTypeAndShapeInfo().GetShape();
    GetInputTensorData(k_data, k_data_type, key);
    auto v_data_type = value.GetTensorTypeAndShapeInfo().GetElementType();
    auto v_shape = value.GetTensorTypeAndShapeInfo().GetShape();
    GetInputTensorData(v_data, v_data_type, value);
    q_data_ptr = qkv_data.cast<uint16_t>();
    k_data_ptr = k_data.cast<uint16_t>();
    v_data_ptr = v_data.cast<uint16_t>();
  }

  // Past Key data / shape
  auto past_k_shape = past_k.GetTensorTypeAndShapeInfo().GetShape();
  auto past_k_size = past_k.GetTensorTypeAndShapeInfo().GetElementCount();
  auto past_k_type = past_k.GetTensorTypeAndShapeInfo().GetElementType();
  DataPtrWrapper past_k_data;
  GetInputTensorData(past_k_data, past_k_type, past_k);
  MY_LOG(2) << "past k shape: " << shape2str(past_k_shape) << std::endl;
  MY_LOG(2) << "past k size: " << past_k_size << std::endl;

  // Past Value data / shape
  auto past_v_shape = past_v.GetTensorTypeAndShapeInfo().GetShape();
  auto past_v_size = past_v.GetTensorTypeAndShapeInfo().GetElementCount();
  auto past_v_type = past_v.GetTensorTypeAndShapeInfo().GetElementType();
  DataPtrWrapper past_v_data;
  GetInputTensorData(past_v_data, past_v_type, past_v);
  MY_LOG(2) << "past v shape: " << shape2str(past_v_shape) << std::endl;
  MY_LOG(2) << "past v size: " << past_v_size << std::endl;

  const int32_t* seq_len_k = seqlens_k.GetTensorData<int32_t>();
  int total_seq_len =
      total_seqlen.GetTensorData<int32_t>()[0]; // chunk cache: 512
  MY_LOG(2) << "k seq len: " << seq_len_k[0] << std::endl;
  MY_LOG(2) << "total seq len: " << total_seq_len << std::endl;

  bool is_prefill = check_prefill(seq_len);
  MY_LOG(2) << "is_prefill: " << is_prefill << std::endl;
  int S = seq_len_k[0] + 1;
  const int past_seqlen = S - seq_len; // total_seq_len - seq_len;
  MY_LOG(2) << "past seqlen: " << past_seqlen << std::endl;

  __TIC__(CAL_POSID)
  /// pos_ids for RoPE
  std::vector<int64_t> pos_ids(seq_len == 1 ? batch_size
                                            : batch_size * seq_len);
  for (int b = 0; b < batch_size; b++) {
    for (auto s = 0; s < seq_len; s++) {
      if (past_seqlen + s < total_seq_len) {
        pos_ids[b * seq_len + s] = static_cast<int64_t>(past_seqlen) + s;
      } else {
        pos_ids[b * seq_len + s] = static_cast<int64_t>(1);
      }
    }
  }
  __TOC__(CAL_POSID)

  __TIC__(ALLOC_OUTPUT)
  MY_LOG(2) << "Getting outputs...\n";
  // Allocate output (primary output shape is same as query shape
  // [batch_size, sequence_length, hidden_size])
  std::vector<int64_t> output_shape(
      {static_cast<int64_t>(batch_size), static_cast<int64_t>(seq_len),
       static_cast<int64_t>(head_size * num_heads_)});
  DataPtrWrapper output_data;
  output_data.dtag = "bf16";

  size_t output_size = output_shape[0] * output_shape[1] * output_shape[2];
  if (!bo_sharing) {
    auto output = ctx.GetOutput(0, output_shape);
    auto output_data_type = output.GetTensorTypeAndShapeInfo().GetElementType();

    GetOutputTensorMutableData(output_data, output_data_type, output);
    MY_LOG(2) << "output shape: " << shape2str(output_shape) << std::endl;
    MY_LOG(2) << "output size: " << output_size << std::endl;
    MY_LOG(2) << "output data: " << output_data.toString();
  }

  MY_LOG(2) << "calculate shape for present k/v...\n";
  // calculate shape for present k/v
  int past_sequence_length = static_cast<int>(past_k_shape[2]);
  int total_sequence_length = total_seq_len > past_sequence_length
                                  ? total_seq_len
                                  : past_sequence_length;
  MY_LOG(2) << "past_sequence_length: " << past_sequence_length << std::endl;
  MY_LOG(2) << "total_sequence_length: " << total_sequence_length << std::endl;
  bool past_present_share_buffer =
      past_sequence_length == total_sequence_length;
  int T_buffer = total_sequence_length;
  std::vector<int64_t> present_k_shape(
      {static_cast<int64_t>(batch_size), static_cast<int64_t>(kv_num_heads_),
       static_cast<int64_t>(total_sequence_length),
       static_cast<int64_t>(head_size)});
  std::vector<int64_t> present_v_shape(
      {static_cast<int64_t>(batch_size), static_cast<int64_t>(kv_num_heads_),
       static_cast<int64_t>(total_sequence_length),
       static_cast<int64_t>(head_size)});
  MY_LOG(2) << "present k shape: " << shape2str(present_k_shape) << std::endl;
  MY_LOG(2) << "present v shape: " << shape2str(present_v_shape) << std::endl;
  MY_LOG(2) << "Getting present_key and present_value...\n";
  auto present_k = gqa_out[0];
  auto present_k_size = present_k.GetTensorTypeAndShapeInfo().GetElementCount();
  auto present_k_data_type =
      present_k.GetTensorTypeAndShapeInfo().GetElementType();
  DataPtrWrapper present_k_data;
  GetOutputTensorMutableData(present_k_data, present_k_data_type, present_k);
  MY_LOG(2) << "present_key data: " << present_k_data.toString();

  auto present_v = gqa_out[1];
  auto present_v_size = present_v.GetTensorTypeAndShapeInfo().GetElementCount();
  auto present_v_data_type =
      present_v.GetTensorTypeAndShapeInfo().GetElementType();
  DataPtrWrapper present_v_data;
  GetOutputTensorMutableData(present_v_data, present_v_data_type, present_v);
  MY_LOG(2) << "present_value data: " << present_v_data.toString();
  MY_LOG(2) << "present k shape: " << shape2str(present_k_shape) << std::endl;
  MY_LOG(2) << "present k size: " << present_k_size << std::endl;
  MY_LOG(2) << "present v shape: " << shape2str(present_v_shape) << std::endl;
  MY_LOG(2) << "present v size: " << present_v_size << std::endl;
  __TOC__(ALLOC_OUTPUT)

  int B = batch_size;         // batch
  int T = total_seq_len;      // total sequence length   //chunk cache: 512
  int past_S = seq_len_k[0];  // past sequence length    //64*(i-1)
  int N_q = num_heads_;       // num_heads
  int N_kv = kv_num_heads_;   // kv num_heads
  int group_num = N_q / N_kv; // group num
  int H = head_size;          // head size

  MY_LOG(2) << "B: " << batch_size << std::endl;
  MY_LOG(2) << "seq_len: " << seq_len << std::endl;
  MY_LOG(2) << "S: " << S << std::endl;
  MY_LOG(2) << "T_buffer: " << T_buffer << std::endl;
  MY_LOG(2) << "past_S: " << past_S << std::endl;
  MY_LOG(2) << "T: " << T << std::endl;
  MY_LOG(2) << "N_q: " << num_heads_ << std::endl;
  MY_LOG(2) << "N_kv: " << kv_num_heads_ << std::endl;
  MY_LOG(2) << "group_num: " << group_num << std::endl;
  MY_LOG(2) << "H: " << head_size << std::endl;

  if (is_prefill && ENV_PARAM(USE_AIE_GQA) == 1 && fused_env_param &&
      S <= mha_aie_kernel_info_.max_seq_length()) {
    MY_LOG(2) << "running AIE kernel" << std::endl;
    if (isBf16Model(qkv_data, output_data, present_k_data, present_v_data)) {
      int64_t S_pad = mha_aie_kernel_info_.try_pad_seq(S, 64);
      MY_LOG(2) << "S_pad: " << S_pad << std::endl;
      int64_t seq_len_pad = mha_aie_kernel_info_.try_pad_seq(seq_len, 64);
      MY_LOG(2) << "seq_len_pad: " << seq_len_pad << std::endl;

      uint16_t* mask_bo_map = softmax_mask.map<uint16_t*>();
      uint16_t* q_bo_map = bmm1_inputs[0].map<uint16_t*>();
      uint16_t* k_bo_map = bmm1_inputs[1].map<uint16_t*>();

      uint16_t* bf16_K_BNSH = nullptr;
      /// transpose V from [B, S, N_kv, H] to [B, N_kv, S, H]
      MY_LOG(2) << "Transposing V." << std::endl;

      uint16_t* bf16_V_BNSH = PrefillGQA_Allocator.get_buffer_generic<uint16_t>(
          kv_size * sizeof(uint16_t),
          PrefillGQA_Allocator::BufferType::AIE_V_T);
      __TIC__(TransposeV)
      transpose0213(bf16_V_BNSH, v_data_ptr, B, seq_len, N_kv, H, context);
      __TOC__(TransposeV)

      /// Getting attention mask from LUT
      MY_LOG(2) << "Filling attn mask." << std::endl;
      __TIC__(AllocFillAttnMask)
      assert(atten_mask_provider_ != nullptr);
      uint16_t* bf16_attention_mask = atten_mask_provider_->get_atten_mask(
          seq_len, seq_len_pad, past_seqlen, 64);
      __TOC__(AllocFillAttnMask)

      std::memcpy(mask_bo_map, bf16_attention_mask,
                  S * seq_len_pad * sizeof(uint16_t));
      /// If the attention mask is not in LUT, we will free it immediately after
      /// it was copied to bo.
      atten_mask_provider_->free_buffer(bf16_attention_mask, past_seqlen + 64);

      auto func_pad_concat_v = [&]() {
        __TIC__(PadConcatKV)
        pad_concat_kv(bmm2_inputs[1].map<uint16_t*>(),
                      past_v_data.cast<uint16_t>(), bf16_V_BNSH, N_kv,
                      past_seqlen, S_pad, H, true, T_buffer, seq_len);
        softmax_mask.sync(XCL_BO_SYNC_BO_TO_DEVICE);   ///////
        bmm2_inputs[1].sync(XCL_BO_SYNC_BO_TO_DEVICE); ////////
        __TOC__(PadConcatKV)
      };
      std::function<void(uint16_t*)> func_save_present_k =
          [&](uint16_t* k_ptr) {
            __TIC__(SavePresentK)
            save_present_kv(present_k_data.cast<uint16_t>(),
                            past_k_data.cast<uint16_t>(), k_ptr, N_kv,
                            past_seqlen, H, T, seq_len);
            __TOC__(SavePresentK)
          };
      std::function<void()> func_save_present_v = [&]() {
        __TIC__(SavePresentV)
        save_present_kv(present_v_data.cast<uint16_t>(),
                        past_v_data.cast<uint16_t>(), bf16_V_BNSH, N_kv,
                        past_seqlen, H, T, seq_len);
        __TOC__(SavePresentV)
      };

      // TODO: chunk cache v, past with pad, chache without pad
      std::shared_future<void> rst_pad_concat_v =
          std::async(std::launch::async, func_pad_concat_v);
      std::shared_future<void> rst_save_v =
          std::async(std::launch::async, func_save_present_v);

      if (ENV_PARAM(USE_AIE_RoPE) == 1 &&
          mha_aie_kernel_info_.is_seq_aie_supported(S)) {
        MY_LOG(2) << "Use AIE ROPE." << std::endl;
        uint16_t* rope_q_in_bo = rope_q_inputs[0].map<uint16_t*>();
        uint16_t* rope_k_in_bo = rope_k_inputs[0].map<uint16_t*>();
        __TIC__(PadQK)
        try_pad_qkv(rope_q_in_bo, q_data_ptr, N_q, seq_len, seq_len_pad, H);
        try_pad_qkv(rope_k_in_bo, k_data_ptr, N_kv, seq_len, seq_len_pad, H);
        __TOC__(PadQK)

        if (!bo_sharing) {
          const_cast<PrefillGQACustomOpKernel*>(this)->aie_execute_rope_mha(
              output_data.cast<uint16_t>(), N_q, N_kv, seq_len, seq_len_pad, S,
              S_pad, H, past_present_share_buffer, T, past_k_data, T,
              rst_pad_concat_v, func_save_present_k);
        } else {
          const_cast<PrefillGQACustomOpKernel*>(this)->aie_execute_rope_mha(
              nullptr, N_q, N_kv, seq_len, seq_len_pad, S, S_pad, H,
              past_present_share_buffer, T, past_k_data, T, rst_pad_concat_v,
              func_save_present_k);
        }
      } else {
        MY_LOG(2) << "Using ORT RoPE for Query." << std::endl;
        uint16_t* bf16_q_rope =
            PrefillGQA_Allocator.get_buffer_generic<uint16_t>(
                q_size * sizeof(uint16_t),
                PrefillGQA_Allocator::BufferType::AIE_Q_ROPE);
        uint16_t* bf16_k_rope =
            PrefillGQA_Allocator.get_buffer_generic<uint16_t>(
                kv_size * sizeof(uint16_t),
                PrefillGQA_Allocator::BufferType::AIE_K_ROPE);
        __TIC__(TransposeQK)
        uint16_t* bf16_q_data_transposed =
            PrefillGQA_Allocator.get_buffer_generic<uint16_t>(
                q_size * sizeof(uint16_t),
                PrefillGQA_Allocator::BufferType::AIE_Q_T);
        uint16_t* bf16_k_data_transposed =
            PrefillGQA_Allocator.get_buffer_generic<uint16_t>(
                kv_size * sizeof(uint16_t),
                PrefillGQA_Allocator::BufferType::AIE_K_T);
        transpose0213(bf16_q_data_transposed, q_data_ptr, B, seq_len, N_q, H,
                      context);
        transpose0213(bf16_k_data_transposed, k_data_ptr, B, seq_len, N_kv, H,
                      context); /////////
        __TOC__(TransposeQK)
        __TIC__(QueryBF16toFP32)
        float* fp32_q_data = PrefillGQA_Allocator.get_buffer_generic<float>(
            q_size * sizeof(float), PrefillGQA_Allocator::BufferType::AIE_F_Q);
        float* fp32_q_rope = PrefillGQA_Allocator.get_buffer_generic<float>(
            q_size * sizeof(float),
            PrefillGQA_Allocator::BufferType::AIE_F_Q_ROPE);
        vec_bf16_to_float(fp32_q_data, bf16_q_data_transposed, q_size);
        __TOC__(QueryBF16toFP32)
        __TIC__(RoPEQuery)
        RoPE(fp32_q_rope, fp32_q_data, pos_ids.data(), cos_cache, sin_cache, B,
             N_q, seq_len, H, context);
        __TOC__(RoPEQuery)
        __TIC__(QueryFP32toBF16)
        vec_float32_to_bf16(q_bo_map, fp32_q_rope, q_size);
        __TOC__(QueryFP32toBF16)

        __TIC__(KeyBF16toFP32)
        float* fp32_k_data = PrefillGQA_Allocator.get_buffer_generic<float>(
            kv_size * sizeof(float), PrefillGQA_Allocator::BufferType::AIE_F_K);
        float* fp32_k_rope = PrefillGQA_Allocator.get_buffer_generic<float>(
            kv_size * sizeof(float),
            PrefillGQA_Allocator::BufferType::AIE_F_K_ROPE);
        vec_bf16_to_float(fp32_k_data, bf16_k_data_transposed, kv_size);
        __TOC__(KeyBF16toFP32)
        __TIC__(RoPEKey)
        RoPE(fp32_k_rope, fp32_k_data, pos_ids.data(), cos_cache, sin_cache, B,
             N_kv, seq_len, H, context);
        __TOC__(RoPEKey)
        std::shared_future<void> rst_save_present_kv =
            std::async(std::launch::async, func_save_present_k, bf16_k_rope);

        __TIC__(KeyFP32toBF16)
        vec_float32_to_bf16(bf16_k_rope, fp32_k_rope, kv_size);
        __TOC__(KeyFP32toBF16)
        //  chunk cache
        __TIC__(PadConcatKV)
        pad_concat_kv(bmm1_inputs[1].map<uint16_t*>(),
                      past_k_data.cast<uint16_t>(), bf16_k_rope, N_kv,
                      past_seqlen, S_pad, H, true, T_buffer, seq_len);
        rst_pad_concat_v.wait();
        __TOC__(PadConcatKV)
        if (!bo_sharing) {
          const_cast<PrefillGQACustomOpKernel*>(this)->aie_execute_mha(
              output_data.cast<uint16_t>(), N_q, N_kv, seq_len, seq_len_pad, S,
              S_pad, H);
        } else {
          const_cast<PrefillGQACustomOpKernel*>(this)->aie_execute_mha(
              nullptr, N_q, N_kv, seq_len, seq_len_pad, S, S_pad, H);
        }
        rst_save_present_kv.get();
      }
      MY_LOG(2) << "save present k/v." << std::endl;
      /// save present k/v
      rst_save_v.wait();

    } else {
      throw std::runtime_error(
          "Not supported now, only support QKV with bfloat16 as inputs.");
    }
  }
  __TOC__(Compute)
  MY_LOG(2) << "- AMD GQA compute done ...\n";
  return bmm2_outputs;
}
} // namespace ort_prefill_gqa_custom_op
