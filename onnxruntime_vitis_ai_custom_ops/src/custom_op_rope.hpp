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
#include <future>
#include <mutex>

#include <ryzenai/dynamic_dispatch/ops/mladfmharope/mladfmharope.hpp>

#include <xrt/xrt_bo.h>

namespace ort_rope_custom_op {

struct OrtTensor {
  std::vector<int64_t> shape;
  size_t size;
  void* data;
};

class MyCustomOpKernel {
public:
  MyCustomOpKernel(const OrtKernelInfo* info, const OrtApi& api);
  ~MyCustomOpKernel();
  // void set_params();
  void LazyInit();
  // void aie_execute(OrtTensor& query_states, OrtTensor& key_states,
  //                  OrtTensor& value_states, OrtTensor& attention_mask,
  //                  OrtTensor& output);
  void transpose0213(uint16_t* output_data, uint16_t* input_data, int D0,
                     int D1, int D2, int D3, OrtKernelContext* context);
  void Compute(OrtKernelContext* context);

private:
  int64_t num_heads_;
  float mask_filter_value_;
  int64_t is_unidirectional_;

  std::string m_node_name;
  Ort::Op op_built_in{nullptr};
  Ort::Op transpose0213_built_in{nullptr};
  Ort::Logger m_logger{nullptr};
  static std::once_flag initFlag;
  Ort::Op op_k{nullptr};
  const OrtApi* api_;

  // aie kernels from DD
  ryzenai::mha_rope<uint16_t, uint16_t, uint16_t>* rope_{nullptr};
  // aie kernel bos
  std::vector<xrt::bo> add_inputs_, add_outputs_;
  uint16_t* sin_ = nullptr;
  uint16_t* cos_ = nullptr;
  uint16_t* trig_max_len = nullptr;
  size_t max_seq_length = 0;
  size_t prefil_m = 0;
  size_t cs_1 = 0;
  int token_counter = 0;
  Ort::ConstValue const_cos_, const_sin_;
  bool dry_run_;
};

struct MyCustomOp : Ort::CustomOpBase<MyCustomOp, MyCustomOpKernel> {
  explicit MyCustomOp() {}

  void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const {
    return new MyCustomOpKernel(info, api);
  };

  const char* GetName() const { return "AMDRotaryEmbedding"; };

  size_t GetInputTypeCount() const { return 4; };
  size_t GetOutputTypeCount() const { return 1; };

  ONNXTensorElementDataType GetInputType(size_t index = 0) const {
    if (index == 0)
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
    else if (index == 1)
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    else if (index == 2)
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    else
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  };

  ONNXTensorElementDataType GetOutputType(size_t index = 0) const {
    if (index == 0)
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
    else
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  };
};

} // namespace ort_rope_custom_op