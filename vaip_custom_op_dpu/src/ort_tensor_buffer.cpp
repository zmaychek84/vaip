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
