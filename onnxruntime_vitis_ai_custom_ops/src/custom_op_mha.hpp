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
#define ORT_API_MANUAL_INIT
#include <onnxruntime_cxx_api.h>
#undef ORT_API_MANUAL_INIT

#include <algorithm>
#include <cassert>
#include <future>
#include <mutex>
#include <string>

#include <ryzenai/dynamic_dispatch/ops/bmm/bmm.hpp>
#include <ryzenai/dynamic_dispatch/ops/maskedsoftmax/maskedsoftmax.hpp>
#include <xrt/xrt_bo.h>

namespace ort_mha_custom_op {

struct OrtTensor {
  std::vector<int64_t> shape;
  size_t size;
  void* data;
};

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

struct MHAAIEKernelInfo {

  MHAAIEKernelInfo() = default;

  int max_seq_length() const { return 2048; }

  int64_t try_pad_seq(int64_t S) const {
    assert(S >= 0 && S <= 2048);
    int64_t S_padded = S;
    if ((S - 256) <= 0)
      S_padded = 256;
    else if ((S - 512) <= 0)
      S_padded = 512;
    else if ((S - 1024) <= 0)
      S_padded = 1024;
    else // if ((M-2048)<=0) {
      S_padded = 2048;
    return S_padded;
  }

  // check if the seq_len is supported by aie
  bool is_seq_aie_supported(int seq_len) const {
    return (supported_seqlen.find(seq_len) != supported_seqlen.end());
  }

private:
  std::set<int> supported_seqlen{256, 512, 1024, 2048};
};

class MyCustomOpKernel {
public:
  MyCustomOpKernel(const OrtKernelInfo* info, const OrtApi& api);
  void set_params();
  void LazyInit();
  void aie_execute(OrtTensor& query_states, OrtTensor& key_states,
                   OrtTensor& value_states, OrtTensor& attention_mask,
                   OrtTensor& output);
  void transpose0213(uint16_t* output_data, uint16_t* input_data, int D0,
                     int D1, int D2, int D3, OrtKernelContext* context);
  void Compute(OrtKernelContext* context);

private:
  const OrtApi* api_;
  int64_t num_heads_;
  float mask_filter_value_;
  int64_t is_unidirectional_;
  std::string mladf_version_ = "v0";

  std::string m_node_name;
  Ort::Op mha_built_in{nullptr};
  Ort::Op transpose0213_built_in{nullptr};
  Ort::Logger m_logger{nullptr};
  static std::once_flag initFlag;

  // aie kernels from DD
  ryzenai::bmm<uint16_t, uint16_t, uint16_t>* bmm1_{nullptr};
  ryzenai::masked_softmax<uint16_t, uint16_t, uint16_t>* softmax_{nullptr};
  ryzenai::bmm<uint16_t, uint16_t, uint16_t>* bmm2_{nullptr};
  // aie kernel bos
  std::vector<xrt::bo> bmm1_inputs;
  std::vector<xrt::bo> bmm1_outputs;
  std::vector<xrt::bo> bmm2_inputs;
  std::vector<xrt::bo> bmm2_outputs;
  xrt::bo softmax_mask;
  bool dry_run_;

  // mha aie kernel info
  MHAAIEKernelInfo mha_aie_kernel_info_;
};

class MHA_Allocator {
public:
  using BufferInfo = std::pair<void*, size_t>;

  static MHA_Allocator& get_instance() {
    static MHA_Allocator self;
    return self;
  }

  MHA_Allocator(const MHA_Allocator&) = delete;
  MHA_Allocator& operator=(const MHA_Allocator&) = delete;
  /// AIE
  uint16_t* get_aie_q_t(size_t sz) {
    size_t real_size = sz <= min_aie_q_t_size_ ? min_aie_q_t_size_ : sz;
    return (uint16_t*)get_buffer(real_size, aie_q_t_);
  }

  uint16_t* get_aie_rpb(size_t sz) {
    size_t real_size = sz <= min_aie_rpb_size_ ? min_aie_rpb_size_ : sz;
    return (uint16_t*)get_buffer(real_size, aie_rpb_);
  }

  uint16_t* get_aie_output(size_t sz) {
    size_t real_size = sz <= min_aie_output_size_ ? min_aie_output_size_ : sz;
    return (uint16_t*)get_buffer(real_size, aie_output_);
  }
  /// ORT
  // get input buffers
  float* get_q(size_t sz) {
    size_t real_size = sz <= min_qkv_size_ ? min_qkv_size_ : sz;
    return (float*)get_buffer(real_size, q_);
  }

  float* get_k(size_t sz) {
    size_t real_size = sz <= min_qkv_size_ ? min_qkv_size_ : sz;
    return (float*)get_buffer(real_size, k_);
  }

  float* get_v(size_t sz) {
    size_t real_size = sz <= min_qkv_size_ ? min_qkv_size_ : sz;
    return (float*)get_buffer(real_size, v_);
  }

  float* get_past_k(size_t sz) {
    size_t real_size = sz <= min_past_kv_size_ ? min_past_kv_size_ : sz;
    return (float*)get_buffer(real_size, past_k_);
  }

  float* get_past_v(size_t sz) {
    size_t real_size = sz <= min_past_kv_size_ ? min_past_kv_size_ : sz;
    return (float*)get_buffer(real_size, past_v_);
  }

  // get output buffers
  float* get_output(size_t sz) {
    size_t real_size = sz <= min_out_size_ ? min_out_size_ : sz;
    return (float*)get_buffer(real_size, output_);
  }

  float* get_present_k(size_t sz) {
    size_t real_size = sz <= min_present_k_size_ ? min_present_k_size_ : sz;
    return (float*)get_buffer(real_size, present_k_);
  }

  float* get_present_v(size_t sz) {
    size_t real_size = sz <= min_present_v_size_ ? min_present_v_size_ : sz;
    return (float*)get_buffer(real_size, present_v_);
  }

private:
  // defined in cpp for logging
  void* get_buffer(size_t sz, BufferInfo& buffer);

  MHA_Allocator() {}

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
    /// AIE
    free_buffer(aie_q_t_);
    free_buffer(aie_rpb_);
    free_buffer(aie_output_);
    /// ORT
    free_buffer(q_);
    free_buffer(k_);
    free_buffer(v_);
    free_buffer(past_k_);
    free_buffer(past_v_);

    free_buffer(output_);
    free_buffer(present_k_);
    free_buffer(present_v_);
  }

  ~MHA_Allocator() { dealloc(); }
  Ort::AllocatorWithDefaultOptions allocator_;
  const float growth_factor_ = 1.5f;
  /// AIE Buffers
  BufferInfo aie_q_t_{nullptr, 0};
  BufferInfo aie_rpb_{nullptr, 0};
  BufferInfo aie_output_{nullptr, 0};
  const size_t min_aie_q_t_size_ = 2048 * 4096 * sizeof(uint16_t);
  const size_t min_aie_rpb_size_ = 2048 * 2048 * sizeof(uint16_t);
  const size_t min_aie_output_size_ = 2048 * 4096 * sizeof(uint16_t);

  /// ORT Buffers
  BufferInfo q_{nullptr, 0};
  BufferInfo k_{nullptr, 0};
  BufferInfo v_{nullptr, 0};
  BufferInfo past_k_{nullptr, 0};
  BufferInfo past_v_{nullptr, 0};
  BufferInfo output_{nullptr, 0};
  BufferInfo present_k_{nullptr, 0};
  BufferInfo present_v_{nullptr, 0};
  const size_t min_qkv_size_ = 1 * 4096 * sizeof(float);
  const size_t min_past_kv_size_ = 32 * 2058 * 128 * sizeof(float);
  const size_t min_out_size_ = 2048 * 4096 * sizeof(float);
  const size_t min_present_k_size_ = 32 * 2058 * 128 * sizeof(float);
  const size_t min_present_v_size_ = 32 * 2058 * 128 * sizeof(float);
  std::vector<BufferInfo> free_list_;
};

struct MyCustomOp : Ort::CustomOpBase<MyCustomOp, MyCustomOpKernel> {
  explicit MyCustomOp() {}

  OrtCustomOpInputOutputCharacteristic
  GetInputCharacteristic(size_t) const noexcept {
    // zero-points input is optional
    return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_OPTIONAL;
  }

  void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const {
    return new MyCustomOpKernel(info, api);
  };

  const char* GetName() const { return "AMDMultiHeadAttention"; };

  size_t GetInputTypeCount() const { return 8; };
  size_t GetOutputTypeCount() const { return 3; };

  ONNXTensorElementDataType GetInputType(size_t index = 0) const {
    if (index == 0)
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
    else if (index == 1)
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
    else if (index == 2)
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
    else if (index == 3)
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    else if (index == 4)
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    else if (index == 5)
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    else if (index == 6)
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
    else if (index == 7)
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
    else
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  };

  ONNXTensorElementDataType GetOutputType(size_t index = 0) const {
    if (index == 0)
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
    else if (index == 1)
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
    else if (index == 2)
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
    else
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  };
};

} // namespace ort_mha_custom_op