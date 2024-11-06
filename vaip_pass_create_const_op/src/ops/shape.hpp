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

template <typename T> auto Shape_tmpl() {
  return [](IPass& self, const Node& node, GTensorView<int64_t> output,
            GTensorView<T> input) -> bool {
    auto input_args = node_get_input_node_args(node);
    CHECK_EQ(input_args.size(), 1) << "Shape input_arg size must 1";
    auto& input_arg = *input_args[0];
    if (node_arg_is_dynamic_shape(input_arg)) {
      return false;
    }
    if (node_arg_is_scalar(input_arg)) {
      return false;
    }
    auto pshape = node_arg_get_shape_i64(input_arg);
    CHECK(pshape != nullptr)
        << node_arg_as_string(input_arg) << " shape absent";
    auto input_shape = *pshape;
    auto rank = (int64_t)input_shape.size();
    auto start = node_get_attr_int_with_default(node, "start", 0);
    auto end = node_get_attr_int_with_default(node, "end", rank);
    if (end < 0) {
      end = rank + end;
    }
    for (auto i = start; i < end; ++i) {
      CHECK_GT(input_shape[i], 0);
      output.data[i] = input_shape[i];
    }
    return true;
  };
}

template <typename... T> static std::unique_ptr<BaseRule> Shape(IPass& pass) {
  return std::make_unique<ConstantFoldRule>(pass, "Shape", Shape_tmpl<T>()...);
}
