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

#include "onnxruntime_api.hpp"

#include <glog/logging.h>
#include <sstream>
//
#include "./custom_op.hpp"

namespace vaip_qdq_op_custom_op {

MyCustomOp::MyCustomOp(std::shared_ptr<const PassContext> context,
                       const std::shared_ptr<MetaDefProto>& meta_def,
                       onnxruntime::Model* model)
    : CustomOpImp(context, meta_def, model) {
  // std::cout << " Custom QDQ_OP constructed." <<std::endl;

  if (meta_def->generic_param().at("zp_dtype") == "uint8") {
    maxval = 255;
    minval = 0;
  } else if (meta_def->generic_param().at("zp_dtype") == "uint16") {
    is_uint16 = true;
    maxval = 65535;
    minval = 0;
  } else if (meta_def->generic_param().at("zp_dtype") == "int8") {
    is_int8 = true;
    maxval = 127;
    minval = -128;
  } else if (meta_def->generic_param().at("zp_dtype") == "int16") {
    is_int16 = true;
    maxval = 32767;
    minval = -32768;
  }

  if (meta_def->generic_param().at("is_qop") == "1")
    is_qop = true;
  else
    is_qop = false;

  zp = stof(meta_def->generic_param().at("in_zero_point"));
  scale = stof(meta_def->generic_param().at("in_scale"));
}

MyCustomOp::~MyCustomOp() {
  // std::cout << " Custom QDQ_OP destructed." <<std::endl;
}

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
  // std::cout << " Custom QDQ_OP computed." <<std::endl;
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
  // std::cout << "Got input tensor" << std::endl;

  auto tensor_info = input_tensor.GetTensorTypeAndShapeInfo();
  //  std::cout << "Got input tensor info" << std::endl;

  auto element_num = tensor_info.GetElementCount();
  //  std::cout << "Got input tensor" << std::endl;
  // std::cout << (int) element_num << std::endl;

  const float* in_data_f = nullptr;
  const uint16_t* in_data_u16 = nullptr;
  const int16_t* in_data_i16 = nullptr;
  const uint8_t* in_data_u8 = nullptr;
  const int8_t* in_data_i8 = nullptr;

  if (is_qop) {
    auto in_ptr = input_tensor.GetTensorData<float>();
    in_data_f = in_ptr;
  } else if (is_int8) {
    auto in_ptr = input_tensor.GetTensorData<int8_t>();
    in_data_i8 = in_ptr;
  } else if (is_int16) {
    auto in_ptr = input_tensor.GetTensorData<int16_t>();
    in_data_i16 = in_ptr;
  } else if (is_uint16) {
    auto in_ptr = input_tensor.GetTensorData<uint16_t>();
    in_data_u16 = in_ptr;
  } else {
    auto in_ptr = input_tensor.GetTensorData<uint8_t>();
    in_data_u8 = in_ptr;
  }

  auto output_tensor = ctx.GetOutput(0, tensor_info.GetShape());

  float* out_data_f = nullptr;
  uint16_t* out_data_u16 = nullptr;
  int16_t* out_data_i16 = nullptr;
  uint8_t* out_data_u8 = nullptr;
  int8_t* out_data_i8 = nullptr;

  if (is_qop == false)
    out_data_f = output_tensor.GetTensorMutableData<float>();
  else if (is_int8)
    out_data_i8 = output_tensor.GetTensorMutableData<int8_t>();
  else if (is_int16)
    out_data_i16 = output_tensor.GetTensorMutableData<int16_t>();
  else if (is_uint16)
    out_data_u16 = output_tensor.GetTensorMutableData<uint16_t>();
  else
    out_data_u8 = output_tensor.GetTensorMutableData<uint8_t>();

  if (is_qop) {
    if (is_int8) {
      QuantizeLinear(in_data_f, out_data_i8, element_num, scale, zp, minval,
                     maxval);
    } else if (is_int16) {
      QuantizeLinear(in_data_f, out_data_i16, element_num, scale, zp, minval,
                     maxval);
    } else if (is_uint16) {
      QuantizeLinear(in_data_f, out_data_u16, element_num, scale, zp, minval,
                     maxval);
    } else {
      QuantizeLinear(in_data_f, out_data_u8, element_num, scale, zp, minval,
                     maxval);
    }
  } else {
    if (is_int8) {
      for (int i = 0; i < element_num; i++) {
        float temp = static_cast<float>(in_data_i8[i]);
        temp = (temp - zp) * scale;
        out_data_f[i] = temp;
      }
    } else if (is_int16) {
      for (int i = 0; i < element_num; i++) {
        float temp = static_cast<float>(in_data_i16[i]);
        temp = (temp - zp) * scale;
        out_data_f[i] = temp;
      }
    } else if (is_uint16) {
      for (int i = 0; i < element_num; i++) {
        float temp = static_cast<float>(in_data_u16[i]);
        temp = (temp - zp) * scale;
        out_data_f[i] = temp;
      }
    } else {
      for (int i = 0; i < element_num; i++) {
        float temp = static_cast<float>(in_data_u8[i]);
        temp = (temp - zp) * scale;
        out_data_f[i] = temp;
      }
    }
  }
}
} // namespace vaip_qdq_op_custom_op
