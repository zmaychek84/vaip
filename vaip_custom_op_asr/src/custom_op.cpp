/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

#include "onnxruntime_api.hpp"

#include <glog/logging.h>
#include <sstream>
//
#include "custom_op.hpp"

#include "vitis/ai/env_config.hpp"
#include <cmath>
#include <thread>
#include <unordered_map>
#include <vitis/ai/weak.hpp>

#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>

#ifdef _WIN32
// #  include "../../xrt_shared_context/xrt_shared_context.hpp"
// graph-engine should not include <xrt.h> in public header files.
// suppress warning, macro redefinition NOMINMAX
#  pragma warning(push)
#  pragma warning(disable : 4005)
#  pragma warning(pop)
#else
#endif
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_GEMM_ASR) >= n)
DEF_ENV_PARAM(DEBUG_GEMM_ASR, "0");

using TvmPackedFunc = ::tvm::runtime::PackedFunc;

namespace vaip_asr_custom_op {

MyCustomOp::MyCustomOp(std::shared_ptr<const PassContext> context,
                       const std::shared_ptr<MetaDefProto>& meta_def)
    : CustomOpImp(meta_def) {
  MY_LOG(1) << " Vitis AI ASR EP running " << meta_def->nodes_size()
            << " Nodes";
  // input shapes
  for (auto& asr_shapes : meta_def->asr_param().input_shapes()) {
    ort_input_shapes_.emplace_back(asr_shapes.shapes().begin(),
                                   asr_shapes.shapes().end());
    auto& a_shape = ort_input_shapes_.back();
    MY_LOG(1) << "input shape dump:";
    std::string str = "==>";
    for (auto item : a_shape) {
      str += " ";
      str += std::to_string(item);
    }
    MY_LOG(1) << str;
  }
  // output shapes
  for (auto& asr_shapes : meta_def->asr_param().output_shapes()) {
    ort_output_shapes_.emplace_back(asr_shapes.shapes().begin(),
                                    asr_shapes.shapes().end());
    auto& a_shape = ort_output_shapes_.back();
    MY_LOG(1) << "output shape dump: ";
    std::string str = "==>";
    for (auto item : a_shape) {
      str += " ";
      str += std::to_string(item);
    }
    MY_LOG(1) << str;
  }

  output_zero_copy_ = meta_def->asr_param().output_zero_copy();

  asr_mod_holder_ = context->get_context_resource("asr_module");
  asr_mod_ = static_cast<TvmModule*>(asr_mod_holder_.get());
  if (!asr_mod_) {
    LOG(FATAL) << "asr module is empty";
  }
}

MyCustomOp::~MyCustomOp() {}

void MyCustomOp::Compute(const OrtApi* api, OrtKernelContext* context) const {
  // there are two version of the global variable, be careful, because
  // ORT is remove all public/external variables for linking.
  if (Ort::Global<void>::api_ == nullptr) {
    Ort::Global<void>::api_ = api;
  }
  Ort::KernelContext ctx(context);

  if (output_zero_copy_) {
    RunWithoutOutputCopy(ctx);
  } else {
    RunWithOutputCopy(ctx);
  }
}

DLDataType GetDataType(ONNXTensorElementDataType type) {
  switch (type) {
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
    return {kDLUInt, 8, 1};
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
    return {kDLInt, 8, 1};
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
    return {kDLUInt, 16, 1};
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
    return {kDLInt, 16, 1};
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
    return {kDLUInt, 32, 1};
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
    return {kDLInt, 32, 1};
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
    return {kDLUInt, 64, 1};
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
    return {kDLInt, 64, 1};
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
    return {kDLFloat, 16, 1};
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
    return {kDLFloat, 32, 1};
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
    return {kDLFloat, 64, 1};
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
    return {kDLUInt, 1, 1};
  default:
    LOG(FATAL) << "Unsupported data type";
  }
}

DLDevice GetDLDevice(OrtMemoryInfoDeviceType device_type) {
  DLDevice context;
  switch (device_type) {
  case 0: // OrtDevice::CPU
    context = {kDLCPU, 0};
    break;
  default:
    LOG(FATAL) << "Unsupported device";
  }
  return context;
}

void OrtTensor2DLTensor(const AsrShapeVec& input_shapes,
                        Ort::KernelContext& context, std::vector<DLTensor>& dst,
                        std::vector<size_t>& dst_inds) {
  size_t num = input_shapes.size();
  dst.reserve(num);
  dst_inds.reserve(num);
  for (size_t i = 0; i < num; ++i) {
    auto& shape = input_shapes[i];
    auto input_tensor = context.GetInput(i);
    CHECK(input_tensor.IsTensor());

    auto ort_device_type = input_tensor.GetTensorMemoryInfo().GetDeviceType();
    const auto tensor_type =
        input_tensor.GetTensorTypeAndShapeInfo().GetElementType();

    DLTensor t;
    t.device = GetDLDevice(ort_device_type);
    t.dtype = GetDataType(tensor_type);
    t.strides = nullptr;
    t.byte_offset = 0;
    t.data = const_cast<void*>(input_tensor.GetTensorRawData());
    t.ndim = shape.size();
    t.shape = const_cast<int64_t*>(shape.data());
    dst.emplace_back(t);
    dst_inds.push_back(i);
  }
}

void TVMSetInputs(TvmModule& mod, std::vector<size_t>& inds,
                  std::vector<DLTensor>& inputs) {
  TvmPackedFunc set_input = mod.GetFunction("set_input", false);
  // TODO: input zero copy is causing accuracy problems, disable now
  for (size_t i = 0; i < inds.size(); ++i) {
    set_input(inds[i], &inputs[i]);
  }
}

void TVMSetOutputsZeroCopy(TvmModule& mod, std::vector<DLTensor>& outputs) {
  TvmPackedFunc set_output = mod.GetFunction("set_output_zero_copy", false);
  for (size_t i = 0; i < outputs.size(); ++i) {
    set_output(i, &outputs[i]);
  }
}

void TVMGetOutputs(TvmModule& mod, std::vector<DLTensor>& outputs) {
  TvmPackedFunc get_output = mod.GetFunction("get_output", false);
  for (size_t i = 0; i < outputs.size(); ++i) {
    get_output(i, &outputs[i]);
  }
}

void TVMRun(TvmModule& mod) {
  TvmPackedFunc run = mod.GetFunction("run", false);
  CHECK(run != nullptr);
  run();
}

static std::vector<DLTensor>
CreateOutputsTensorWithDeviceDataType(Ort::KernelContext& context,
                                      const AsrShapeVec& output_shapes) {
  // set output
  std::vector<DLTensor> output_tensors;
  // original tvm ep use TVMGetOutputShapes, we just use output_shapes_ which
  // came from onnxruntime
  for (size_t i = 0; i < output_shapes.size(); ++i) {
    auto& output_shape = output_shapes[i];
    auto output_tensor =
        context.GetOutput(i, output_shape.data(), output_shape.size());
    CHECK(output_tensor.IsTensor());

    DLTensor t;
    t.strides = nullptr;
    t.byte_offset = 0;
    t.ndim = output_shape.size();
    t.shape = const_cast<int64_t*>(output_shape.data());
    t.device = GetDLDevice(output_tensor.GetTensorMemoryInfo().GetDeviceType());
    t.dtype =
        GetDataType(output_tensor.GetTensorTypeAndShapeInfo().GetElementType());
    t.data = output_tensor.GetTensorMutableRawData();
    output_tensors.push_back(t);
  }
  return output_tensors;
}

void SetInputTensors(TvmModule& mod, const AsrShapeVec& input_shapes,
                     Ort::KernelContext& context) {
  std::vector<size_t> inds;
  std::vector<DLTensor> dl_tensors_inputs;
  OrtTensor2DLTensor(input_shapes, context, dl_tensors_inputs, inds);
  TVMSetInputs(mod, inds, dl_tensors_inputs);
}

void MyCustomOp::RunWithoutOutputCopy(Ort::KernelContext& context) const {
  SetInputTensors(*asr_mod_, ort_input_shapes_, context);
  auto output_tensors =
      CreateOutputsTensorWithDeviceDataType(context, ort_output_shapes_);
  TVMSetOutputsZeroCopy(*asr_mod_, output_tensors);
  // run
  TVMRun(*asr_mod_);
}

void MyCustomOp::RunWithOutputCopy(Ort::KernelContext& context) const {
  SetInputTensors(*asr_mod_, ort_input_shapes_, context);
  auto output_tensors =
      CreateOutputsTensorWithDeviceDataType(context, ort_output_shapes_);
  // run
  TVMRun(*asr_mod_);
  TVMGetOutputs(*asr_mod_, output_tensors);
}

} // namespace vaip_asr_custom_op
