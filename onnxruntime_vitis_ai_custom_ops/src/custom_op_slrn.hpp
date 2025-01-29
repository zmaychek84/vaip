/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */
#define ORT_API_MANUAL_INIT
#include <onnxruntime_cxx_api.h>
#undef ORT_API_MANUAL_INIT

#include <algorithm>
#include <future>
#include <mutex>

// #include <ryzenai/dynamic_dispatch/ops/bmm/bmm.hpp>
#include <ryzenai/dynamic_dispatch/ops/mladfrmsnorm/mladfrmsnorm.hpp>

#include <xrt/xrt_bo.h>

namespace ort_slrn_custom_op {

struct OrtTensor {
  std::vector<int64_t> shape;
  size_t size;
  void* data;
};

class MyCustomOpKernel1 {
public:
  MyCustomOpKernel1(const OrtKernelInfo* info);
  ~MyCustomOpKernel1();
  // void set_params();
  void LazyInit();
  // void aie_execute(OrtTensor& query_states, OrtTensor& key_states,
  //                  OrtTensor& value_states, OrtTensor& attention_mask,
  //                  OrtTensor& output);
  void Compute(OrtKernelContext* context);

private:
  int64_t num_heads_;
  float mask_filter_value_;
  int64_t is_unidirectional_;

  std::string m_node_name;
  Ort::Op op_built_in{nullptr};
  Ort::Logger m_logger{nullptr};
  static std::once_flag initFlag;
  float epsilon_;
  int64_t axis_;
  int64_t stash_type_;
  bool dry_run_;

  Ort::Op op_k{nullptr};

  // aie kernels from DD
  ryzenai::rms_norm<uint16_t, uint16_t, uint16_t>* rms_norm_{nullptr};
  // aie kernel bos
  std::vector<Tensor> rms_norm_inputs_, rms_norm_outputs_;
  std::vector<size_t> supported_lengths{3072, 2048, 1920, 1792, 1664, 1536,
                                        1408, 1280, 1152, 1024, 800,  768,
                                        640,  512,  384,  256,  128};
  uint16_t* wts_ = nullptr;
  Ort::ConstValue m_weights;
};

struct MyCustomOp1 : Ort::CustomOpBase<MyCustomOp1, MyCustomOpKernel1> {
  explicit MyCustomOp1() {}

  OrtCustomOpInputOutputCharacteristic
  GetInputCharacteristic(size_t) const noexcept {
    // zero-points input is optional
    return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_OPTIONAL;
  }

  OrtCustomOpInputOutputCharacteristic
  GetOutputCharacteristic(size_t) const noexcept {
    // zero-points input is optional
    return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_OPTIONAL;
  }

  void* CreateKernel(const OrtApi&, const OrtKernelInfo* info) const {
    return new MyCustomOpKernel1(info);
  };

  const char* GetName() const { return "AMDSimplifiedLayerNormalization"; };

  size_t GetInputTypeCount() const { return 2; };
  size_t GetOutputTypeCount() const { return 1; };

  ONNXTensorElementDataType GetInputType(size_t index = 0) const {
    if (index == 0)
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
    else if (index == 1)
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    else
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  };

  ONNXTensorElementDataType GetOutputType(size_t index = 0) const {
    if (index == 0)
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
    else
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  };
};

} // namespace ort_slrn_custom_op