/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

#include "ort_tensor_buffer.hpp"

#include <cmath>
#include <vitis/ai/weak.hpp>

namespace vaip_dpu_custom_op {

std::shared_ptr<vart::TensorBuffer>
OrtTensorBuffer::create(std::unique_ptr<xir::Tensor> tensor, void* data) {
  return std::make_shared<OrtTensorBuffer>(std::move(tensor), data);
}

OrtTensorBuffer::OrtTensorBuffer(std::unique_ptr<xir::Tensor> tensor,
                                 void* data)
    : vart::TensorBuffer{tensor.release()}, data_{data},
      tensor_{const_cast<xir::Tensor*>(vart::TensorBuffer::get_tensor())} {}
OrtTensorBuffer::~OrtTensorBuffer() {}

std::pair<std::uint64_t, std::size_t>
OrtTensorBuffer::data(const std::vector<std::int32_t> idx) {
  if (idx.size() == 0) {
    return {reinterpret_cast<uint64_t>(data_), tensor_->get_data_size()};
  }
  auto dims = tensor_->get_shape();
  auto offset = 0;
  for (auto k = 0u; k < dims.size(); k++) {
    auto stride = 1;
    for (auto m = k + 1; m < dims.size(); m++) {
      stride *= dims[m];
    }
    offset += idx[k] * stride;
  }

  auto dtype_size = tensor_->get_data_type().bit_width / 8;
  auto elem_num = tensor_->get_element_num();

  return std::make_pair(reinterpret_cast<uint64_t>(data_) + offset * dtype_size,
                        (elem_num - offset) * dtype_size);
}

} // namespace vaip_dpu_custom_op
