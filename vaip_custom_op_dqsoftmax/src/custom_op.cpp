/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

#include "onnxruntime_api.hpp"

#include "./custom_op.hpp"
#include <glog/logging.h>
#include <sstream>

namespace vaip_dqsoftmax_custom_op {

MyCustomOp::MyCustomOp(std::shared_ptr<const PassContext> context,
                       const std::shared_ptr<MetaDefProto>& meta_def,
                       onnxruntime::Model* model)
    : CustomOpImp(context, meta_def, model) {
  zp = (uint16_t)stoi(meta_def->generic_param().at("in_zero_point"));
  scale = stof(meta_def->generic_param().at("in_scale"));
  h_w = stoi(meta_def->generic_param().at("h_w"));
  channel = stoi(meta_def->generic_param().at("channel"));
}

MyCustomOp::~MyCustomOp() {}

// static std::string shape_to_string(const std::vector<int64_t>& shape) {
//   std::ostringstream str;
//   str << "[";
//   int c = 0;
//   for (auto s : shape) {
//     if (c != 0) {
//       str << ",";
//     }
//     str << s;
//     c = c + 1;
//   }
//   str << "]";
//   return str.str();
// }

template <typename DTYPE>
void dqsoftmax(DTYPE* input, float* output, float scale_x, int zp_x, int c_sz,
               int h_w) {
  for (int h = 0; h < h_w; h++) {
    DTYPE* input_iter = input + h * c_sz;
    float* output_iter = output + h * c_sz;
    std::vector<float> exp_vec(c_sz);

    float sum_exp = 0.0f;

    int i = 0;

#ifdef _WIN32
    // Vectorized part
    __m512 sum_exp_vec = _mm512_setzero_ps();

    for (; i <= c_sz - 16; i += 16) {
      __m512 input_vec = _mm512_cvtepi32_ps(
          _mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)&input_iter[i])));
      __m512 dq_v_vec = _mm512_mul_ps(input_vec, _mm512_set1_ps(scale_x));
      __m512 exp_vec_avx = _mm512_exp_ps(dq_v_vec);
      _mm512_storeu_ps(&exp_vec[i], exp_vec_avx);
      sum_exp_vec = _mm512_add_ps(sum_exp_vec, exp_vec_avx);
    }

    // Horizontal sum of sum_exp_vec
    float sum_array[16];
    _mm512_storeu_ps(sum_array, sum_exp_vec);
    for (int j = 0; j < 16; ++j) {
      sum_exp += sum_array[j];
    }
#endif

    // Handle the remaining elements
    for (; i < c_sz; ++i) {
      float dq_v = static_cast<float>(input_iter[i]) * scale_x;
      float exp_val = std::exp(dq_v);
      exp_vec[i] = exp_val;
      sum_exp += exp_val;
    }

    float sum_exp_inv = 1.0f / sum_exp;
    i = 0;

#ifdef _WIN32
    // Vectorized part for softmax
    __m512 sum_exp_inv_vec = _mm512_set1_ps(sum_exp_inv);

    for (i = 0; i <= c_sz - 16; i += 16) {
      __m512 exp_vec_avx = _mm512_loadu_ps(&exp_vec[i]);
      __m512 softmax_vec = _mm512_mul_ps(exp_vec_avx, sum_exp_inv_vec);
      _mm512_storeu_ps(&output_iter[i], softmax_vec);
    }
#endif

    // Handle the remaining elements
    for (; i < c_sz; ++i) {
      output_iter[i] = exp_vec[i] * sum_exp_inv;
    }
  }
}

void MyCustomOp::Compute(const OrtApi* api, OrtKernelContext* context) const {
  if (Ort::Global<void>::api_ == nullptr) {
    Ort::Global<void>::api_ = api;
  }

  Ort::KernelContext ctx(context);
  auto num_inputs = ctx.GetInputCount();
  auto num_outputs = ctx.GetOutputCount();

  auto input_tensor = ctx.GetInput(0);

  auto tensor_info = input_tensor.GetTensorTypeAndShapeInfo();
  auto in_data = input_tensor.GetTensorData<uint16_t>();

  auto output_tensor = ctx.GetOutput(0, tensor_info.GetShape());
  auto out_data = output_tensor.GetTensorMutableData<float>();

  dqsoftmax(in_data, out_data, scale, zp, channel, h_w);
}
} // namespace vaip_dqsoftmax_custom_op
