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

static void copy_fix_info_from_input(IPass& pass, const Node& node,
                                     size_t arg_index) {
  auto inputs = node_get_inputs(node);
  CHECK_LT(arg_index, inputs.size()) << node_as_string(node);
  auto& input_node_arg = *inputs[arg_index].node_arg;
  auto input_name = node_arg_get_name(input_node_arg);
  if (pass.has_fix_info(input_name.c_str())) {
    auto node_arg_name = node_get_output_name(node);
    pass.set_fix_info(node_arg_name.c_str(),
                      pass.get_fix_info(input_name.c_str()));
  }
}
template <typename T> static auto Reshape_tmpl() {
  return [](IPass& self, const Node& node, GTensorView<T> output,
            GTensorView<T> input) -> bool {
    copy_fix_info_from_input(self, node, 0u);
    std::copy(input.data.begin(), input.data.end(), output.data.begin());
    return true;
  };
}

template <typename... T> static std::unique_ptr<BaseRule> Reshape(IPass& pass) {
  return std::make_unique<ConstantFoldRule>(pass, "Reshape",
                                            Reshape_tmpl<T>()...);
}
