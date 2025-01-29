/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */
#include "onnxruntime_api.hpp"

#include <glog/logging.h>
#include <sstream>
//
#include "./custom_op.hpp"

namespace vaip_concat_custom_op {

MyCustomOp::MyCustomOp(std::shared_ptr<const PassContext> context,
                       const std::shared_ptr<MetaDefProto>& meta_def,
                       onnxruntime::Model* model)
    : CustomOpImp(context, meta_def, model) {

  zp = stoi(meta_def->generic_param().at("out_zero_point"));
  scale = stof(meta_def->generic_param().at("out_scale"));
}

MyCustomOp::~MyCustomOp() {}

template <typename OutputType>
inline void QuantizeLinear(const float* Input, OutputType* Output, size_t N,
                           float Scale, float ZeroPoint, float MinimumValue,
                           float MaximumValue) {
  auto THREAD_NUM =
      std::max((int)std::thread::hardware_concurrency() / 2, (int)1);
  std::vector<std::future<int>> thr_fut(THREAD_NUM);
  size_t THREAD_WORKLOAD = size_t(ceil((float)N / THREAD_NUM));
  for (size_t i = 0U; i < THREAD_NUM; i++) {
    thr_fut[i] = std::async(
        std::launch::async,
        [&](size_t i) {
          size_t BASE_POS = i * THREAD_WORKLOAD;
          auto workload = std::min(THREAD_WORKLOAD + BASE_POS, N);
          float FloatValue;
          for (size_t n = BASE_POS; n < workload; n++) {
            FloatValue = std::nearbyintf(Input[n] / Scale) + ZeroPoint;
            FloatValue = std::max(FloatValue, MinimumValue);
            FloatValue = std::min(FloatValue, MaximumValue);
            Output[n] = (OutputType)(int32_t)FloatValue;
          }
          return 0;
        },
        i);
  }

  for (auto i = 0U; i < THREAD_NUM; i++) {
    thr_fut[i].wait();
  }
}
void MyCustomOp::Compute(const OrtApi* api, OrtKernelContext* context) const {
  // std::cout << " Custom CONCAT computed." <<std::endl;
  if (Ort::Global<void>::api_ == nullptr) {
    Ort::Global<void>::api_ = api;
  }

  Ort::KernelContext ctx(context);
  auto num_inputs = ctx.GetInputCount();
  auto num_outputs = ctx.GetOutputCount();
  // LOG(INFO) << "num_inputs " << num_inputs << " "   //
  //           << "num_outputs " << num_outputs << " " //
  //     ;
  // std::cout << (int) num_inputs << std::endl;

  auto input_tensor = ctx.GetInput(0);
  auto input_tensor1 = ctx.GetInput(1);
  auto input_tensor2 = ctx.GetInput(2);
  auto input_tensor3 = ctx.GetInput(3);

  // std::cout << "Got input tensor" << std::endl;

  auto tensor_info = input_tensor.GetTensorTypeAndShapeInfo();
  //  std::cout << "Got input tensor info" << std::endl;

  auto element_num = tensor_info.GetElementCount();
  // std::cout << "Got input tensor" << std::endl;
  // std::cout << (int) element_num << std::endl;

  const uint16_t* in_data_f = nullptr;
  const uint16_t* in_data_f1 = nullptr;
  const uint16_t* in_data_f2 = nullptr;
  const uint16_t* in_data_f3 = nullptr;

  auto in_ptr = input_tensor.GetTensorData<uint16_t>();
  in_data_f = in_ptr;
  auto in_ptr1 = input_tensor1.GetTensorData<uint16_t>();
  in_data_f1 = in_ptr1;
  auto in_ptr2 = input_tensor2.GetTensorData<uint16_t>();
  in_data_f2 = in_ptr2;
  auto in_ptr3 = input_tensor3.GetTensorData<uint16_t>();
  in_data_f3 = in_ptr3;

  std::vector<int64_t> output_shape({static_cast<int64_t>(1),
                                     static_cast<int64_t>(num_inputs),
                                     static_cast<int64_t>(element_num)});
  auto output_tensor = ctx.GetOutput(0, output_shape);

  uint16_t* out_data_f = nullptr;
  out_data_f = output_tensor.GetTensorMutableData<uint16_t>();

  for (int i = 0; i < element_num; i++) {
    out_data_f[i] = in_data_f[i];
  }
  for (int i = 0; i < element_num; i++) {
    out_data_f[i + element_num] = in_data_f1[i];
  }
  for (int i = 0; i < element_num; i++) {
    out_data_f[i + 2 * element_num] = in_data_f2[i];
  }
  for (int i = 0; i < element_num; i++) {
    out_data_f[i + 3 * element_num] = in_data_f3[i];
  }
}
} // namespace vaip_concat_custom_op
