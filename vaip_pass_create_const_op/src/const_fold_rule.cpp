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

#include "const_fold_rule.hpp"
#include "vaip/node.hpp"
#include <glog/logging.h>
#include <memory>

#include <fstream>

#include <vitis/ai/env_config.hpp>
#include <vitis/ai/weak.hpp>

namespace vaip_pass_create_const_op {

ConstantFoldRule::ConstantFoldRule(std::nullptr_t, IPass& pass,
                                   const std::string& op_type,
                                   std::vector<action_t>&& action)
    : pass_{pass}, op_type_{op_type}, action_{std::move(action)} {
  MY_LOG(1) << "rule is created: @" << (void*)this;
}

ConstantFoldRule::~ConstantFoldRule() {
  MY_LOG(1) << "rule is destroyed: @" << (void*)this;
}

static TensorView get_constant_data(IPass& pass, const Node* node) {
  auto data = pass.get_const_data<char>(*node);
  return {data, node_get_output_element_type(*node),
          node_get_output_shape(*node, 0)};
}

bool is_constant_op(const Node* node) {
  return node_is_op(*node, "const", "com.xilinx");
}

bool ConstantFoldRule::compute(const Node& node, TensorView output,
                               const std::vector<TensorView>& inputs) {
  auto ret = false;
  for (auto& f : action_) {
    ret = ret || f(pass_, node, output, inputs);
  }
  return ret;
}

static size_t shape_to_size(const std::vector<int64_t>& shape) {
  int64_t r = 1;
  for (auto v : shape) {
    r = r * v;
  }
  CHECK_GT(r, 0);
  return (size_t)r;
}

bool ConstantFoldRule::apply_once(onnxruntime::Graph* graph,
                                  const onnxruntime::Node* node) {
  auto op_type = VAIP_ORT_API(node_op_type)(*node);
  auto ret = false;
  if (op_type == op_type_) {
    auto inputs = node_get_inputs(*node);
    std::vector<TensorView> input_data;
    input_data.reserve(inputs.size());
    auto can_const_fold = true;
    for (auto input : inputs) {
      if (input.node == nullptr) {
        MY_LOG(2) << "cancel constant fold, input might be a graph's input"
                  << node_as_string(*node)
                  << " input: " << node_arg_as_string(*input.node_arg);
        return false;
      }
      if (is_constant_op(input.node)) {
        input_data.push_back(get_constant_data(pass_, input.node));
      } else if (op_type == "Shape") {
        // Shape op does not read the input data. it is ok to give a nullptr.
        if (node_arg_is_unknown_shape(*input.node_arg)) {
          can_const_fold = false;
        } else {
          input_data.push_back(TensorView{
              gsl::span<char>(), 0u, *node_arg_get_shape_i64(*input.node_arg)});
        }
      } else {
        can_const_fold = false;
      }
    }
    if (can_const_fold) {
      auto name = VAIP_ORT_API(node_get_name)(*node);
      auto op_type = std::string("const");
      auto description =
          std::string("constant folding from ") + name + " " + op_type;
      auto input_args = std::vector<const NodeArg*>{};
      auto output_args = node_get_output_node_args(*node);
      auto domain = std::string("com.xilinx");
      if (output_args.size() > 1) {
        LOG(WARNING) << "constant folding only support single output yet.";
        return false;
      }
      auto& arg = *output_args[0];
      auto shape = node_arg_get_shape_i64(arg);
      CHECK(shape != nullptr) << node_arg_as_string(arg) << " shape absent";
      auto data_type = VAIP_ORT_API(node_arg_get_element_type)(arg);
      if (node_arg_is_dynamic_shape(arg)) {
        return false;
      }
      auto size_of_value_type = sizeof(int8_t);
      switch (data_type) {
      case ONNX_NAMESPACE::TensorProto_DataType_INT8: {
        size_of_value_type = sizeof(int8_t);
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_INT64: {
        size_of_value_type = sizeof(int64_t);
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT: {
        size_of_value_type = sizeof(float);
        break;
      }
      default:
        // NOTE: add more info
        LOG(WARNING) << "unsupported constant folding: "
                     << node_as_string(*node);
        return false;
      }
      MY_LOG(1) << "start constant folding: " << node_as_string(*node);
      auto tmp_data =
          std::vector<char>(shape_to_size(*shape) * size_of_value_type);
      auto my_data = TensorView{tmp_data, node_get_output_element_type(*node),
                                node_get_output_shape(*node, 0)};
      ret = compute(*node, my_data, input_data);
      if (ret) {
        MY_LOG(1) << "constant folding success: " << node_as_string(*node);
        auto& new_node = NodeBuilder(*graph, pass_)
                             .set_op_type("const")
                             .clone_shape(*node)
                             .clone_data_type(*node)
                             .set_anchor_point1(*node)
                             .build();
        pass_.create_const(new_node, tmp_data);
      } else {
        LOG(WARNING) << "constant folding failure: " << node_as_string(*node);
      }
    } else {
      ret = false;
    }
  }
  return ret;
}

} // namespace vaip_pass_create_const_op
