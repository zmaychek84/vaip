/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */
#pragma once

#include "vaip/vaip.hpp"
#include <chrono>

#define ORT_API_MANUAL_INIT
#include <onnxruntime_cxx_api.h>
#undef ORT_API_MANUAL_INIT

#include <algorithm>
#include <cassert>
#include <functional>
#include <future>
#include <mutex>
#include <string>

#include <ryzenai/dynamic_dispatch/ops/bmm/bmm.hpp>
#include <ryzenai/dynamic_dispatch/ops/maskedsoftmax/maskedsoftmax.hpp>
#include <ryzenai/dynamic_dispatch/ops/mladfmharope/mladfmharope.hpp>
#include <xrt/xrt_bo.h>

#include "gqa_helper.hpp"

namespace vaip_gqa_custom_op {

using namespace vaip_core;

struct DataPtrWrapper {

  DataPtrWrapper() : T(nullptr), dtag("") {}

  void* T{nullptr};
  std::string dtag;

  bool is_float32() const { return dtag == "float32"; }

  bool is_bf16() const { return dtag == "bf16"; }
  std::string toString() {
    return dtag + ": " + std::to_string(reinterpret_cast<uintptr_t>(T)) + "\n";
  }

  template <typename U> U* cast() const {
    assert(dtag != "");
    if (is_type<U>()) {
      return static_cast<U*>(T);
    } else {
      throw std::runtime_error(
          "Type mismatch: cannot cast to the specified type.");
    }
  }

private:
  template <typename U> bool is_type() const {
    if constexpr (std::is_same<U, float>::value) {
      return is_float32();
    } else if constexpr (std::is_same<U, uint16_t>::value) {
      return is_bf16();
    } else {
      return false;
    }
  }
};

struct PrefillGQAAIEKernelInfo {

  PrefillGQAAIEKernelInfo() = default;

  int max_seq_length() const { return MAX_SEQ_LENGTH; }

  int64_t try_pad_seq(int64_t S, int chunk_size) const {
    assert(S >= 0 && S <= MAX_SEQ_LENGTH);
    int64_t S_padded = S;
    if (S <= MAX_SEQ_LENGTH)
      S_padded = (S % chunk_size) ? (S + chunk_size - (S % chunk_size)) : S;
    else
      S_padded = MAX_SEQ_LENGTH;
    return S_padded;
  }

  int64_t try_pad_total_seq(int64_t T) const {
    assert(T > 0 && T < MAX_SEQ_LENGTH);
    int64_t T_padded = T;
    if (T % 128 != 0) {
      T_padded = ((int)(T / 128) + 1) * 128;
    }
    return T_padded;
  }

  // check if the seq_len is supported by aie
  bool is_seq_aie_supported(int seq_len) const {
    return (supported_seqlen.find(seq_len) != supported_seqlen.end());
  }

private:
  std::set<int> supported_seqlen{64, 128, 192, 256, 320, 384, 448, 512};
};

struct PrefillAttenMaskProvider {
  PrefillAttenMaskProvider(const PrefillGQAAIEKernelInfo* aie_kernel_info)
      : aie_kernel_info_(aie_kernel_info) {
    // this will trigger LUT construction
    attn_mask_lut_ = &(AttnMask3DLUTSingleton::getInstance(64)); // TODO
  }

  void free_buffer(void* p, int S) {
    /// If the attention mask is in LUT, we do nothing here.
    if (attn_mask_lut_->hasLut(S))
      return;
    /// If the attention mask is not in LUT, we will free it.
    if (p != nullptr)
      allocator_.Free(p);
  }
  ~PrefillAttenMaskProvider() {}

  // define in cpp since we need to use log utility.
  uint16_t* get_atten_mask(int32_t S, int32_t S_pad, int32_t past_S,
                           int32_t kv_size);

private:
  const PrefillGQAAIEKernelInfo* aie_kernel_info_{nullptr};
  AttnMask3DLUTSingleton* attn_mask_lut_{nullptr};
  // use for mask allocation
  Ort::AllocatorWithDefaultOptions allocator_;
  std::vector<void*> free_list_{};
};

class MyCustomOp : public CustomOpImp {
public:
  MyCustomOp(std::shared_ptr<const PassContext> context,
             const std::shared_ptr<MetaDefProto>& meta_def,
             onnxruntime::Model* model);

  virtual ~MyCustomOp();

  void set_params();
  void LazyInit();
  void aie_execute_rope(const uint16_t* input, uint16_t* output, const int N,
                        const int S, const int H, const int past_S = 0);
  void aie_execute_rope_mha(uint16_t* output, const int N_q, const int N_kv,
                            const int seq_len, const int seq_len_pad,
                            const int S, const int S_pad, const int H,
                            bool past_present_share_buffer, int T_buffer,
                            DataPtrWrapper past_k_data, int past_k_size,
                            const int T, std::shared_future<void> cache_v,
                            std::function<void(uint16_t*)> save_k_func);
  void transpose0213(uint16_t* output_data, uint16_t* input_data, int D0,
                     int D1, int D2, int D3, OrtKernelContext* context) const;
  void get_rope_cache(const std::shared_ptr<MetaDefProto>& meta_def);
  void InitializeParams(const OrtApi& api, Ort::ConstKernelInfo& info,
                        Ort::ConstValue const_cos_, Ort::ConstValue const_sin,
                        int is_cos_const, int is_sin_const);

private:
  virtual void Compute(const OrtApi* api,
                       OrtKernelContext* context) const override final;
  int cnt;
  inline static bool dry_run_{false};

  inline static int headsize_{0};
  std::string mladf_version_ = "v0";

  inline static int instances__{0};
  inline static int compute_instances__{0};

  int64_t do_rotary_;
  int64_t kv_num_heads_;
  int64_t num_heads_;
  inline static bool is_packed_qkv_{false};
  static std::once_flag initFlag;

  // aie kernels from DD
  inline static ryzenai::bmm<uint16_t, uint16_t, uint16_t>* bmm1_{nullptr};
  inline static ryzenai::masked_softmax<uint16_t, uint16_t, uint16_t>* softmax_{
      nullptr};
  inline static ryzenai::bmm<uint16_t, uint16_t, uint16_t>* bmm2_{nullptr};
  inline static ryzenai::mha_rope<uint16_t, uint16_t, uint16_t>* rope_q_{
      nullptr};
  inline static ryzenai::mha_rope<uint16_t, uint16_t, uint16_t>* rope_k_{
      nullptr};
  // aie kernel bos
  inline static std::vector<xrt::bo> rope_q_inputs{};
  inline static std::vector<xrt::bo> rope_q_outputs{};
  inline static std::vector<xrt::bo> rope_k_inputs{};
  inline static std::vector<xrt::bo> rope_k_outputs{};
  inline static std::vector<xrt::bo> bmm1_inputs{};
  inline static std::vector<xrt::bo> bmm1_outputs{};
  inline static std::vector<xrt::bo> bmm2_inputs{};
  inline static std::vector<xrt::bo> bmm2_outputs{};
  inline static std::vector<xrt::bo> sm_inputs{};
  inline static xrt::bo softmax_input{};
  inline static xrt::bo softmax_mask{};
  inline static xrt::bo softmax_output{};
  inline static std::vector<xrt::bo> rope_inputs_{}, rope_outputs_{};
  /// RoPE
  static bool is_const_cache;
  inline static std::shared_ptr<uint16_t> cos_sin_cache__{nullptr};
  inline static size_t max_seq_length = 0;
  inline static size_t cs_1 = 0;
  PrefillGQAAIEKernelInfo mha_aie_kernel_info_;
  // attention provider
  inline static std::unique_ptr<PrefillAttenMaskProvider> atten_mask_provider_{
      nullptr};
};

class PrefillGQA_Allocator {
public:
  using BufferInfo = std::pair<void*, size_t>;

  // Enum to identify each buffer type for
  enum class BufferType {
    AIE_Q,
    AIE_K,
    AIE_V,
    AIE_Q_T,
    AIE_K_T,
    AIE_V_T,
    AIE_Q_ROPE,
    AIE_K_ROPE,
    AIE_K_T_P,
    AIE_ROPE_OR_PAD,
    AIE_V_T_P,
    AIE_KERNEL_OUT,
    AIE_Q_P,
    AIE_K_P,
    AIE_V_P,
    AIE_RPB_P,
    AIE_OUTPUT_P,
    AIE_F_Q,
    AIE_F_Q_ROPE,
    AIE_F_K,
    AIE_F_K_ROPE,
    /// Ort
    ORT_Q,
    ORT_K,
    ORT_V,
    ORT_PAST_K,
    ORT_PAST_V,
    ORT_OUTPUT,
    ORT_PRESENT_K,
    ORT_PRESENT_V,
    // Add more as needed
  };

  // Structure to hold buffer information and minimum size for
  // PrefillGQA_Allocator class
  struct BufferMeta {
    BufferInfo buffer{nullptr, 0};
    size_t min_size;
  };

  static PrefillGQA_Allocator& get_instance() {
    static PrefillGQA_Allocator self;
    return self;
  }

  PrefillGQA_Allocator(const PrefillGQA_Allocator&) = delete;
  PrefillGQA_Allocator& operator=(const PrefillGQA_Allocator&) = delete;

  template <typename T> T* get_buffer_generic(size_t sz, BufferType type) {
    if (buffer_map_.find(type) == buffer_map_.end()) {
      throw std::runtime_error("Buffer type not found in buffer_map_");
    }
    auto& meta = buffer_map_.at(type);
    size_t real_size = sz <= meta.min_size ? meta.min_size : sz;
    return static_cast<T*>(get_buffer(real_size, meta.buffer));
  }

private:
  // defined in cpp for logging
  void* get_buffer(size_t sz, BufferInfo& buffer);

  PrefillGQA_Allocator() {}

  void free_buffer(BufferInfo& info) {
    if (info.first) {
      allocator_.Free(info.first);
      info.first = nullptr;
    }
    info.second = 0;
  }

  void dealloc() {
    // first free items in free list
    for (auto& item : free_list_) {
      free_buffer(item);
    }
    free_list_.clear();

    for (auto& [type, meta] : buffer_map_) {
      free_buffer(meta.buffer);
    }
  }

  ~PrefillGQA_Allocator() { dealloc(); }
  Ort::AllocatorWithDefaultOptions allocator_;
  const float growth_factor_ = 1.5f;

  const size_t min_aie_q_size_ = 2048 * 4096 * sizeof(uint16_t);
  const size_t min_aie_k_size_ = 2048 * 1024 * sizeof(uint16_t);
  const size_t min_aie_v_size_ = 2048 * 1024 * sizeof(uint16_t);
  const size_t min_aie_q_t_size_ = 2048 * 4096 * sizeof(uint16_t);
  const size_t min_aie_k_t_size_ = 2048 * 1024 * sizeof(uint16_t);
  const size_t min_aie_v_t_size_ = 2048 * 1024 * sizeof(uint16_t);
  const size_t min_aie_q_rope_size_ = 2048 * 4096 * sizeof(uint16_t);
  const size_t min_aie_k_rope_size_ = 2048 * 4096 * sizeof(uint16_t);
  const size_t min_aie_k_t_p_size_ = 2048 * 4096 * sizeof(uint16_t);
  const size_t min_aie_rope_or_pad_size_ = 2048 * 4096 * sizeof(uint16_t);
  const size_t min_aie_v_t_p_size_ = 2048 * 4096 * sizeof(uint16_t);
  const size_t min_aie_kernel_out_size_ = 2048 * 4096 * sizeof(uint16_t);

  const size_t min_aie_q_p_size_ = 2048 * 4096 * sizeof(uint16_t);
  const size_t min_aie_k_p_size_ = 2048 * 4096 * sizeof(uint16_t);
  const size_t min_aie_v_p_size_ = 2048 * 4096 * sizeof(uint16_t);
  const size_t min_aie_rpb_p_size_ = 2048 * 2048 * sizeof(uint16_t);
  const size_t min_aie_output_p_size_ = 2048 * 4096 * sizeof(uint16_t);

  const size_t min_f_q_size_ = 2048 * 4096 * sizeof(float);
  const size_t min_f_q_rope_size_ = 2048 * 4096 * sizeof(float);
  const size_t min_f_k_size_ = 2048 * 1024 * sizeof(float);
  const size_t min_f_k_rope_size_ = 2048 * 1024 * sizeof(float);
  /// Ort
  const size_t min_q_size_ = 1 * 4096 * sizeof(float);
  const size_t min_k_size_ = 1 * 1024 * sizeof(float);
  const size_t min_v_size_ = 1 * 1024 * sizeof(float);
  const size_t min_past_kv_size_ = 32 * 2058 * 128 * sizeof(float);
  const size_t min_out_size_ = 2048 * 4096 * sizeof(float);
  const size_t min_present_k_size_ = 32 * 2058 * 128 * sizeof(float);
  const size_t min_present_v_size_ = 32 * 2058 * 128 * sizeof(float);
  std::vector<BufferInfo> free_list_{};

  // Buffer map to associate BufferType with BufferMeta
  std::unordered_map<BufferType, BufferMeta> buffer_map_{
      /// AIE Prefill
      {BufferType::AIE_Q, {{nullptr, 0}, min_aie_q_size_}},
      {BufferType::AIE_K, {{nullptr, 0}, min_aie_k_size_}},
      {BufferType::AIE_V, {{nullptr, 0}, min_aie_v_size_}},
      {BufferType::AIE_Q_T, {{nullptr, 0}, min_aie_q_t_size_}},
      {BufferType::AIE_K_T, {{nullptr, 0}, min_aie_k_t_size_}},
      {BufferType::AIE_V_T, {{nullptr, 0}, min_aie_v_t_size_}},
      {BufferType::AIE_Q_ROPE, {{nullptr, 0}, min_aie_q_rope_size_}},
      {BufferType::AIE_K_ROPE, {{nullptr, 0}, min_aie_k_rope_size_}},
      {BufferType::AIE_K_T_P, {{nullptr, 0}, min_aie_k_t_p_size_}},
      {BufferType::AIE_ROPE_OR_PAD, {{nullptr, 0}, min_aie_rope_or_pad_size_}},
      {BufferType::AIE_V_T_P, {{nullptr, 0}, min_aie_v_t_p_size_}},
      {BufferType::AIE_KERNEL_OUT, {{nullptr, 0}, min_aie_kernel_out_size_}},
      {BufferType::AIE_Q_P, {{nullptr, 0}, min_aie_q_p_size_}},
      {BufferType::AIE_K_P, {{nullptr, 0}, min_aie_k_p_size_}},
      {BufferType::AIE_V_P, {{nullptr, 0}, min_aie_v_p_size_}},
      {BufferType::AIE_RPB_P, {{nullptr, 0}, min_aie_rpb_p_size_}},
      {BufferType::AIE_OUTPUT_P, {{nullptr, 0}, min_aie_output_p_size_}},
      {BufferType::AIE_F_Q, {{nullptr, 0}, min_f_q_size_}},
      {BufferType::AIE_F_Q_ROPE, {{nullptr, 0}, min_f_q_rope_size_}},
      {BufferType::AIE_F_K, {{nullptr, 0}, min_f_k_size_}},
      {BufferType::AIE_F_K_ROPE, {{nullptr, 0}, min_f_k_rope_size_}},
      /// ORT
      {BufferType::ORT_Q, {{nullptr, 0}, min_q_size_}},
      {BufferType::ORT_K, {{nullptr, 0}, min_k_size_}},
      {BufferType::ORT_V, {{nullptr, 0}, min_v_size_}},
      {BufferType::ORT_PAST_K, {{nullptr, 0}, min_past_kv_size_}},
      {BufferType::ORT_PAST_V, {{nullptr, 0}, min_past_kv_size_}},
      {BufferType::ORT_OUTPUT, {{nullptr, 0}, min_out_size_}},
      {BufferType::ORT_PRESENT_K, {{nullptr, 0}, min_present_k_size_}},
      {BufferType::ORT_PRESENT_V, {{nullptr, 0}, min_present_v_size_}},
      // Add more buffers as needed
  };
};

} // namespace vaip_gqa_custom_op
