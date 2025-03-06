/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */
#include "onnxruntime_api.hpp"

#include <glog/logging.h>
#include <sstream>
//
#include "./custom_op.hpp"

namespace vaip_qdqunsqueeze_custom_op {

MyCustomOp::MyCustomOp(std::shared_ptr<const PassContext> context,
                       const std::shared_ptr<MetaDefProto>& meta_def,
                       onnxruntime::Model* model)
    : CustomOpImp(context, meta_def, model) {}

MyCustomOp::~MyCustomOp() {}

void MyCustomOp::Compute(const OrtApi* api, OrtKernelContext* context) const {
  if (Ort::Global<void>::api_ == nullptr) {
    Ort::Global<void>::api_ = api;
  }

  Ort::KernelContext ctx(context);
  auto num_inputs = ctx.GetInputCount();
  auto num_outputs = ctx.GetOutputCount();
  auto input_tensor = ctx.GetInput(0);
  auto tensor_info = input_tensor.GetTensorTypeAndShapeInfo();
  auto element_num = tensor_info.GetElementCount();
  auto input_shape = tensor_info.GetShape();
  std::vector<int64_t> output_shape;
  /*
  - Get the input tensor shape and derive output shape
  - Due to Unsqueeze op we add an additional dimension to the output shape
  - copy the input elements to the output
  */

  const float* in_data_f = input_tensor.GetTensorData<float>();
  output_shape.push_back(static_cast<int64_t>(1));
  output_shape.insert(output_shape.end(), input_shape.begin(),
                      input_shape.end());
  auto output_tensor = ctx.GetOutput(0, output_shape);
  float* out_data_f = output_tensor.GetTensorMutableData<float>();

  std::memcpy(out_data_f, in_data_f, element_num * sizeof(float));
}
} // namespace vaip_qdqunsqueeze_custom_op
