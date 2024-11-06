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

#include "glog/logging.h"
#include "vaip/vaip.hpp"
#include "vitis/ai/env_config.hpp"
DEF_ENV_PARAM(DEBUG_CONST_FOLDING, "0")
DEF_ENV_PARAM(XLNX_ENABLE_OLD_QDQ, "1")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_CONST_FOLDING) >= n)
namespace vaip_pass_create_const_op {
using namespace vaip_core;

static void create_const_op(IPass& pass, Graph& graph, const std::string& name,
                            const TensorProto* input) {

  auto shape = tensor_proto_get_shape(*input);
  auto node_arg = VAIP_ORT_API(graph_get_node_arg)(graph, name);
  CHECK(node_arg != nullptr) << "cannot get node arg: name=" << name;
  auto data_type =
      data_type_to_string(VAIP_ORT_API(node_arg_get_element_type)(*node_arg));
  auto op_type = std::string("const");
  auto attrs = NodeAttributesBuilder(); //
  attrs.add("data_type", data_type);
  if (!shape.empty()) {
    attrs.add("shape", shape);
  } else {
    attrs.add("shape", std::vector<int64_t>{});
  }
  auto tensor_data_type = VAIP_ORT_API(tensor_proto_data_type)(*input);
  if (tensor_data_type == onnx::TensorProto_DataType_UINT8 &&
      ENV_PARAM(XLNX_ENABLE_OLD_QDQ) == 1) {
    MY_LOG(1) << name
              << " : cancel create const op , not support uint8 data type";
    return;
  }
  auto node_arg1 = vaip_cxx::NodeArgConstRef::from_node_arg(graph, *node_arg);
  auto data = node_arg1.const_data_as_raw();
  // some node_arg maybe has name, but don't has data, such as Resize op's
  // roi/scales. for this scene, create_const maybe lead to graph_resolve error.
  // test case: issue #874, yolov3-coco_quantized_model.onnx
  if (data.empty()) {
    MY_LOG(1) << name << ": don't contain data, cancel create const op";
    return;
  }
  auto type = VAIP_ORT_API(node_arg_get_element_type)(*node_arg);
  pass.create_const(name.c_str(), data, shape, type);
  graph_add_node(graph, name, op_type, "convert from initialized tensor.", {},
                 {node_arg}, attrs.build(), "com.xilinx");
  VAIP_ORT_API(graph_remove_node)(graph, {nullptr, node_arg});
  return;
}

void create_const_ops(IPass& pass, Graph& graph) {
  auto constants = VAIP_ORT_API(graph_get_all_initialized_tensors)(graph);
  for (auto constant : constants) {
    create_const_op(pass, graph, constant.first, (constant.second));
  }
}

} // namespace vaip_pass_create_const_op
