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

static std::ostream& operator<<(std::ostream& s,
                                const gsl::span<const int64_t>& v) {
  s << "[";
  for (auto c = 0u; c < v.size(); ++c) {
    if (c != 0) {
      s << ",";
    }
    s << v[c];
  }
  s << "]";
  return s;
}

static std::ostream& operator<<(std::ostream& s,
                                const std::vector<int64_t>& v) {
  s << "[";
  for (auto c = 0u; c < v.size(); ++c) {
    if (c != 0) {
      s << ",";
    }
    s << v[c];
  }
  s << "]";
  return s;
}
static std::vector<int64_t>
get_dst_shape(const std::vector<int64_t>& src_shape,
              const gsl::span<const int64_t>& order) {
  auto dim_size = src_shape.size();
  auto dst_shape = std::vector<int64_t>(); // src OIHW dst OHWI
  dst_shape.reserve(dim_size);
  for (auto idx : order) {
    dst_shape.push_back(src_shape[idx]);
  }
  return dst_shape;
}

std::vector<int32_t> trans_idx(const std::vector<int32_t>& input_idx,
                               const gsl::span<const int64_t>& order) {
  auto sz = input_idx.size();
  auto output_idx = std::vector<int32_t>(sz, 0);
  for (auto i = 0u; i < sz; ++i) {
    output_idx[i] = input_idx[order[i]];
  }
  return output_idx;
}

template <typename T> static auto Transpose_tmpl() {
  return [](IPass& self, const Node& node, GTensorView<T> output,
            GTensorView<T> input) -> bool {
    CHECK_EQ(input.data.size(), output.data.size());
    auto input_args = node_get_input_node_args(node);
    CHECK_EQ(input_args.size(), 1) << "Transpose input_arg size must 1";
    auto& input_arg = *input_args[0];
    auto pshape = node_arg_get_shape_i64(input_arg);
    CHECK(pshape != nullptr)
        << node_arg_as_string(input_arg) << " shape absent";
    auto input_shape = *pshape;
    auto perm = node_get_attr_ints(node, "perm");
    CHECK_EQ(input_shape.size(), perm.size());
    auto output_shape = get_dst_shape(input_shape, perm);
    MY_LOG(1) << "transpose perm " << perm << " "
              << "input_shape " << input_shape << " output_shape "
              << output_shape << ", " << node_as_string(node);
    auto src_shape = trans_shape_i64_to_i32(input_shape);
    auto dst_shape = trans_shape_i64_to_i32(output_shape);
    auto src_dim_calc = std::make_unique<vitis::ai::DimCalc>(src_shape);
    auto dst_dim_calc = std::make_unique<vitis::ai::DimCalc>(dst_shape);
    for (auto i = 0u; i < input.data.size(); ++i) {
      auto src_idx = src_dim_calc->index(i);
      auto dst_idx = trans_idx(src_idx, perm);
      auto dst_offset = dst_dim_calc->offset(dst_idx);
      output.data[dst_offset] = input.data[i];
    }
    return true;
  };
}

template <typename... T>
static std::unique_ptr<BaseRule> Transpose(IPass& pass) {
  return std::make_unique<ConstantFoldRule>(pass, "Transpose",
                                            Transpose_tmpl<T>()...);
}
