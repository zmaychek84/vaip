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

#include "vaip/vaip.hpp"
#include "vaip/xir_headers.hpp"
#include <unordered_map>

namespace vaip_pass_to_xir_ops {
using namespace vaip_core;

class ToXirRule {
public:
  using action_t = std::function<bool(ToXirRule* self, const Graph& /*graph*/,
                                      const Node& /*node*/,
                                      NodeAttributesBuilder& /*attrs*/)>;
  explicit ToXirRule(const std::string& onnx_op_type,
                     const std::string xir_op_type, IPass& pass);

  bool apply(Graph& graph, const Node& node);
  // all inputs & outputs must not be scalar
  ToXirRule& check_scalar();
  // all to xir op not support zero shape
  ToXirRule& check_node_input_zero_shape();

  ToXirRule& rename(
      const std::string from, const std::string to,
      std::function<AttributeProtoPtr(ToXirRule* self, const Node& node)>
          or_else = [](ToXirRule* self, const Node& node) { return nullptr; });
  ToXirRule& rename_and_convert_HW_to_WH(const std::string from,
                                         const std::string to);
  ToXirRule& add_attr_s(const std::string name, const std::string value);
  ToXirRule& add_attr_i(const std::string name, int64_t value);
  ToXirRule& remove_input(size_t index) {
    remove_inputs_.push_back(index);
    return *this;
  }
  ToXirRule& copy(const std::string from) { return rename(from, from); }
  ToXirRule& action(const action_t& action) {
    actions_.push_back(action);
    return *this;
  }

  ToXirRule& constant_input_arg_to_attr(size_t arg_index,
                                        const std::string& attr_name,
                                        const std::string& fun);

  ToXirRule& set_shape_1_for_scalar();

  ToXirRule& set_xir_op_type(const std::string& xir_op_type);

  std::string debug_string() { return onnx_op_type_ + " -> " + xir_op_type_; }

private:
  std::vector<const NodeArg*> get_input_args(const Node& node);
  std::vector<const NodeArg*> get_output_args(const Node& node);

  // all op not support dynamic shape
  void check_dynamic_shape();
  void check_xir_support_data_type();

private:
  std::string onnx_op_type_;
  std::string xir_op_type_;
  std::vector<action_t> actions_;
  std::vector<size_t> remove_inputs_;

public:
  IPass& pass_;
};
void to_xir_ops_pass(IPass& self, onnxruntime::Graph& graph);
} // namespace vaip_pass_to_xir_ops
