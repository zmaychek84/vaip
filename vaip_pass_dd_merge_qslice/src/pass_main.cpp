/*
 *     The Xilinx Vitis AI Vaip in this distribution are provided under the
 * following free and permissive binary-only license, but are not provided in
 * source code form.  While the following free and permissive license is similar
 * to the BSD open source license, it is NOT the BSD open source license nor
 * other OSI-approved open source license.
 *
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

#include "vaip/dd/dd_utils.hpp"
#include "vaip/pattern_zoo.hpp"
#include "vaip/vaip.hpp"
namespace {
using namespace vaip_core;

static void add_node_attr_qslice(NodeBuilder& building_node,
                                 onnxruntime::Graph* graph, binder_t* binder) {
  std::vector<std::string> nodes;
  for (auto& ni : *binder) {
    if ((*node_arg_is_constant)(*graph, *ni.second.node_arg)) {
      continue;
    }
    nodes.push_back(node_arg_get_name(*ni.second.node_arg));
  }
  building_node.add("nodes", nodes);
}

const NodeArg& add_tensor(onnxruntime::Graph* graph, const NodeArg& clone) {
  auto value = std::vector<uint8_t>(16, 0);
  auto shape = std::vector<int64_t>{16};
  auto name = node_arg_get_name(clone) + "_qdq";
  auto new_tensor = tensor_proto_new_u8(name, shape, value);
  VAIP_ORT_API(graph_add_initialized_tensor)(*graph, *new_tensor);
  return VAIP_ORT_API(node_arg_new)(*graph, name, &shape,
                                    ONNX_NAMESPACE::TensorProto_DataType_UINT8);
}

struct MergeQSlice {
  MergeQSlice(IPass& self) : self_{self} {}
  std::unique_ptr<Rule> create_rule(IPass* self) {
    auto q = vaip::pattern_zoo::get_pattern("m_qslice");
    CHECK(q != nullptr) << "Pattern returned is null";
    return Rule::create_rule(
        q, [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
          auto ni_input = binder["input_0"];
          auto ni_output = binder["q"];

          auto new_node = NodeBuilder(*graph, self_);
          new_node.set_op_type("QSlice", "com.xilinx");
          auto start_node = binder["constant_0"];
          auto a_s_node = binder["a_s"];
          auto a_z_node = binder["a_z"];

          auto a_scale =
              node_arg_get_const_data_as_float(*graph, *a_s_node.node_arg);
          auto a_zp = vaip::dd::get_zp_from_node(*graph, *a_z_node.node_arg);
          auto start_data =
              node_arg_get_const_data_as_i64s(*graph, *start_node.node_arg);
          LOG(INFO) << "Modified QSlice Op";
          int64_t slice_idx = start_data[0] == 0 ? 0 : 1;
          const NodeArg& qdq_node_arg =
              add_tensor(graph, *binder["dq"].node_arg);
          new_node.set_input_node_args({ni_input.node_arg, &qdq_node_arg});
          new_node.add("orig_output_shape",
                       *node_arg_get_shape_i64(*ni_output.node_arg));
          add_node_attr_qslice(new_node, graph, &binder);
          new_node.clone_shape(*ni_output.node_arg);
          new_node.set_anchor_point1(*ni_output.node);
          std::vector<std::string> in_dtypes = {"uint16", "uint8"};
          std::vector<std::string> out_dtypes = {"uint16"};
          new_node.add("q_scale", a_scale);
          new_node.add("q_zp", (int64_t)a_zp);
          new_node.add("in_dtypes", in_dtypes);
          new_node.add("out_dtypes", out_dtypes);
          new_node.add("slice_idx", slice_idx);

          new_node.build();
          return true;
        });
  }
  void process(IPass& self, Graph& graph) { create_rule(&self)->apply(&graph); }

public:
  IPass& self_;
};
} // namespace

DEFINE_VAIP_PASS(MergeQSlice, vaip_pass_dd_merge_qslice)