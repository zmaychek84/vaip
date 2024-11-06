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

#include "vaip/vaip.hpp"
#include <glog/logging.h>
#include <vitis/ai/dim_calc.hpp>
static std::vector<int32_t>
get_dst_shape(const std::vector<int32_t>& src_shape,
              const gsl::span<const int32_t>& order) {
  auto dim_size = src_shape.size();
  auto dst_shape = std::vector<int32_t>(); // src OIHW dst OHWI
  dst_shape.reserve(dim_size);
  for (auto idx : order) {
    dst_shape.push_back(src_shape[idx]);
  }
  return dst_shape;
}
static int get_element_num(const std::vector<int32_t> shape) {
  auto ret = 1;
  for (auto dim : shape) {
    ret *= dim;
  }
  return ret;
}
static std::vector<int32_t>
trans_shape_i64_to_i32(const gsl::span<const int64_t>& shape) {
  auto ret = std::vector<int32_t>(shape.size());
  for (auto i = 0u; i < shape.size(); ++i) {
    ret[i] = (int32_t)shape[i];
  }
  return ret;
}

static std::vector<int32_t> trans_idx(const std::vector<int32_t>& input_idx,
                                      const std::vector<int32_t>& order) {
  auto sz = input_idx.size();
  auto output_idx = std::vector<int32_t>(sz, 0);
  for (auto i = 0u; i < sz; ++i) {
    output_idx[i] = input_idx[order[i]];
  }
  return output_idx;
}

template <typename T>
void transpose_data(const gsl::span<const int64_t>& shape,
                    const gsl::span<const int64_t>& perm,
                    const gsl::span<const T>& data, bool flip_hw, T* ret) {
  auto src_shape = trans_shape_i64_to_i32(shape);
  auto order = trans_shape_i64_to_i32(perm);
  CHECK_EQ(src_shape.size(), order.size());
  auto dst_shape = get_dst_shape(src_shape, order);

  auto size = get_element_num(src_shape);
  CHECK_EQ(data.size(), size);
  auto src_dim_calc = std::make_unique<vitis::ai::DimCalc>(src_shape);
  auto dst_dim_calc = std::make_unique<vitis::ai::DimCalc>(dst_shape);
  if (flip_hw) {
    for (auto i = 0u; i < data.size(); ++i) {
      auto src_idx = src_dim_calc->index(i);
      src_idx[2] = src_shape[2] - src_idx[2] - 1;
      src_idx[3] = src_shape[3] - src_idx[3] - 1;
      auto dst_idx = trans_idx(src_idx, order);
      auto dst_offset = dst_dim_calc->offset(dst_idx);
      ret[dst_offset] = data[i];
    }
  } else {
    for (auto i = 0u; i < data.size(); ++i) {
      auto src_idx = src_dim_calc->index(i);
      auto dst_idx = trans_idx(src_idx, order);
      auto dst_offset = dst_dim_calc->offset(dst_idx);
      ret[dst_offset] = data[i];
    }
  }
  return;
}
template void transpose_data<char>(const gsl::span<const int64_t>&,
                                   const gsl::span<const int64_t>&,
                                   const gsl::span<const char>&, bool, char*);
template void transpose_data<float>(const gsl::span<const int64_t>&,
                                    const gsl::span<const int64_t>&,
                                    const gsl::span<const float>&, bool,
                                    float*);
template void transpose_data<uint16_t>(const gsl::span<const int64_t>&,
                                       const gsl::span<const int64_t>&,
                                       const gsl::span<const uint16_t>&, bool,
                                       uint16_t*);
template void transpose_data<int16_t>(const gsl::span<const int64_t>&,
                                      const gsl::span<const int64_t>&,
                                      const gsl::span<const int16_t>&, bool,
                                      int16_t*);
