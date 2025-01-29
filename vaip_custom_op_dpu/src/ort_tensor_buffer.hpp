/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

#pragma once

#include <vart/tensor_buffer.hpp>
#include <xir/tensor/tensor.hpp>

namespace vaip_dpu_custom_op {

class OrtTensorBuffer : public vart::TensorBuffer {
public:
  static std::shared_ptr<vart::TensorBuffer>
  create(std::unique_ptr<xir::Tensor> tensor, void* data);

public:
  explicit OrtTensorBuffer(std::unique_ptr<xir::Tensor> tensor, void* data);
  virtual ~OrtTensorBuffer();
  OrtTensorBuffer(const OrtTensorBuffer& other) = delete;
  OrtTensorBuffer& operator=(const OrtTensorBuffer& rhs) = delete;

private:
  virtual std::pair<std::uint64_t, std::size_t>
  data(const std::vector<std::int32_t> idx) override;

private:
  void* data_;
  std::unique_ptr<xir::Tensor> tensor_;
};

} // namespace vaip_dpu_custom_op
