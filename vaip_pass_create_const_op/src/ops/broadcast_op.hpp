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

#pragma once

template <typename T, typename T2>
bool Broadcast_op_tmpl(GTensorView<T> output, GTensorView<T> input_a,
                       GTensorView<T> input_b, T2 calc) {
  auto get_shape = [](std::vector<int64_t> shape) {
    if (shape.size() == 0)
      return std::vector<int64_t>(1, 1);
    return shape;
  };
  auto shape_a = get_shape(input_a.shape);
  auto shape_b = get_shape(input_b.shape);
  auto shape_o = get_shape(output.shape);

  auto calc_len = [](const std::vector<int64_t>& shape) {
    int64_t len = 1;
    for (auto s : shape)
      len *= s;
    return len;
  };

  if (shape_a.size() != shape_o.size())
    shape_a.resize(shape_o.size(), 1);
  if (shape_b.size() != shape_o.size())
    shape_b.resize(shape_o.size(), 1);

  CHECK_EQ(calc_len(shape_o), calc_len(shape_a) * calc_len(shape_b));
  for (size_t i = 0; i < shape_o.size(); i++) {
    if (shape_a[i] == 1 || shape_b[i] == 1) {
      CHECK_EQ(shape_a[i] * shape_b[i], shape_o[i]);
    }
  }

  auto get_strides = [](const std::vector<int64_t>& a) {
    auto ret = std::vector<int64_t>(a.size() + 1);
    auto index = a.size();
    ret[index] = 1;
    for (auto i = 0u; i < a.size(); ++i) {
      ret[index - 1] = ret[index] * a[index - 1];
      index = index - 1;
    }
    return ret;
  };

  auto a_strides = get_strides(shape_a);
  auto b_strides = get_strides(shape_b);
  auto c_i = std::vector<int64_t>(shape_o.size(), 0);
  size_t index_c = 0;
  int64_t index_a = 0;
  int64_t index_b = 0;

  auto tick = [=, &c_i, &index_a, &index_b, &index_c]() {
    index_c++;
    index_a++;
    index_b++;
    if (index_c >= output.data.size()) {
      return;
    }
    for (auto dim = shape_o.size() - 1; dim >= 0; dim--) {
      c_i[dim]++;
      CHECK_LE(c_i[dim], shape_o[dim]) << " dim= " << dim;
      if (c_i[dim] == shape_o[dim]) {
        c_i[dim] = 0;
      } else {
        if (shape_a[dim] == 1) {
          index_a = index_a - a_strides[dim + 1];
        }
        if (shape_b[dim] == 1) {
          index_b = index_b - b_strides[dim + 1];
        }
        break;
      }
    }
  };

  for (; index_c < output.data.size(); tick()) {
    output.data[index_c] = calc(input_a.data[index_a], input_b.data[index_b]);
  }

  return true;
}
