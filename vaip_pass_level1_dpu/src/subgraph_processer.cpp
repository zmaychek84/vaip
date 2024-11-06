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

#include "./subgraph_processer.hpp"
#include "compile_model.hpp"
#include "graph.hpp"
#include "node_arg.hpp"
#include "vaip/anchor_point.hpp"
#include "vaip/node.hpp"
#include "vitis/ai/env_config.hpp"
#include <iterator>
#include <regex>
#include <utility>
#include <vaip/my_ort.h>
#include <vaip/vaip_ort_api.h>
DEF_ENV_PARAM(DEBUG_LEVEL1_DPU, "0")
DEF_ENV_PARAM(VAIP_COMPILE_RESERVE_CONST_DATA, "0")
DEF_ENV_PARAM(XLNX_ENABLE_OUTPUT_DEPAD, "0")
DEF_ENV_PARAM(XLNX_ENABLE_GRAPH_ENGINE_DEPAD, "1")
DEF_ENV_PARAM(XLNX_ENABLE_GRAPH_ENGINE_PAD, "1")
DEF_ENV_PARAM(XLNX_ENABLE_CHECK_DPU_SG_4D, "0")
DEF_ENV_PARAM(XLNX_ENABLE_OP_NAME_PROTECTION, "1")
DEF_ENV_PARAM(DISABLE_MATMUL_DPU_SG, "0")
DEF_ENV_PARAM(DISABLE_RESIZE_DPU_SG, "0")
DEF_ENV_PARAM(DISABLE_CONCAT_DPU_SG, "0")
DEF_ENV_PARAM(DISABLE_GEMM_DPU_SG, "0")
DEF_ENV_PARAM(XLNX_ENABLE_BATCH, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_LEVEL1_DPU) >= n)

namespace vaip_level1_dpu {
using namespace vaip_core;

static std::ostream&
operator<<(std::ostream& s,
           const google::protobuf::RepeatedPtrField<std::string>& v) {
  s << "[";
  for (auto c = 0; c < v.size(); ++c) {
    if (c != 0) {
      s << ",";
    }
    s << v[c];
  }
  s << "]";
  return s;
}
static std::ostream& operator<<(std::ostream& s, std::vector<std::string>& v) {
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
static bool ends_with_str(const std::string& str, const std::string& suffix) {
  if (str.size() < suffix.size()) {
    return false;
  }
  auto ends = str.substr(str.size() - suffix.size());
  return suffix == ends;
}

static std::string
tensor_name_remove_xir_suffix_protect(const std::string& name,
                                      std::string& head, std::string& tail) {
  std::size_t start_pos = name.find(head);
  if (start_pos != std::string::npos) {
    start_pos += head.length();
    std::size_t end_pos = name.find(tail, start_pos);
    if (end_pos != std::string::npos) {
      return name.substr(start_pos, end_pos - start_pos);
    }
  }
  return "";
}

static std::string tensor_name_remove_xir_suffix(const std::string& name) {
  std::regex pattern(R"(_reshaped_\d+_inserted_fix_\d+$)");
  std::smatch match;
  if (std::regex_search(name, match, pattern)) {
    return name.substr(0, match.position());
  }
  pattern = R"(_reshaped_\d+_inserted_fix_\d+_merged$)";
  if (std::regex_search(name, match, pattern)) {
    return name.substr(0, match.position());
  }
  auto pos = name.find("_recomputation_");
  if (pos != std::string::npos) {
    return name.substr(0, pos);
  }
  pos = name.find("_fix_reshaped_inserted_fix_");
  if (pos != std::string::npos) {
    return name.substr(0, pos);
  }

  pos = name.find("(TransferMatMulToConv2d)_inserted_fix");
  if (pos != std::string::npos) {
    return name.substr(0, pos);
  }
  // testcase : nightly_84_opset17_u8s8_p1.json : botnet26t_256
  pos = name.find("_input_inserted_reshape");
  if (pos != std::string::npos) {
    return name.substr(0, pos);
  }
  // testcase : nightly_84_opset17_u8s8_p1.json : lambda_resnet50ts
  pos = name.find("_transfered_to_4d");
  if (pos != std::string::npos) {
    return name.substr(0, pos);
  }
  pos = name.find("_reshaped_inserted_fix_");
  if (pos != std::string::npos) {
    return name.substr(0, pos);
  }

  pos = name.find("_inserted_fix_");
  if (pos != std::string::npos) {
    return name.substr(0, pos);
  }

  pos = name.find("_FixShifter");
  if (pos != std::string::npos) {
    return name.substr(0, pos);
  }

  // testcase: pr#1178
  pos = name.find("_insert_conv2d_fix");
  if (pos != std::string::npos) {
    return name.substr(0, pos);
  }

  auto new_suffix = std::string("_new");
  if (ends_with_str(name, new_suffix)) {
    return name.substr(0, name.size() - new_suffix.size());
  }

  auto suffix = std::string("_fix");
  auto is_ends_fix = ends_with_str(name, suffix);
  if (is_ends_fix) {
    return name.substr(0, name.size() - suffix.size());
  } else {
    return name;
  }
}

// we assume the output of a DPU subgraph is most likely anchored to a
// `fix` op, but the data type of `fix` op is float, actually, the
// above QuantizeLinear is the right anchor, then
//
//    return value != anchor_point->get_origin_name()
//
// we need create a new anchor point for this case, to correct the
// disagreement between `vaip` and `xcompiler`
//
// Unfortunately sometimes `xcompiler` add a `fix` op internally, so
// that the anchor point is not a `fix` op any more, it could be a
// `relu` or anything, in this case, then
//
//    return value == anchor_point->get_origin_name()
//
//
static std::unique_ptr<AnchorPoint>
find_q_dq_anchor_point(const IPass& pass,
                       const onnxruntime::Graph& onnx_dot_onnx_graph,
                       const std::string& origin_node_name) {
  auto ret = std::unique_ptr<AnchorPoint>();
  auto guess_origin_name = origin_node_name;
  auto onnx_node =
      VAIP_ORT_API(graph_producer_node)(onnx_dot_onnx_graph, origin_node_name);
  // potentially `onnx_node` could be nullptr when DPU subgraph
  // connect to DPU subgraph, e.g. test case `"EfficientNet_int"`
  //
  // when processing blob.20, it is anchored to the `Conv` op because
  // `fix` is inserted by xcompiler.
  //
  // after fusing subgraph `subgraph_blob.20_vaip_12`, `blob.20` is deleted by
  // ::fuse` see issue #560 for more details.
  //
  // then we process subgraph `subgraph_blob.28_vaip_18`, it cannot
  // find the anchor point `blob.20`.
  //
  // in such case `onnx_node` == nullptr
  if (onnx_node == nullptr) {
    return nullptr;
  }

  auto data_type = node_get_output_element_type(*onnx_node);
  if (onnx::TensorProto_DataType_UINT8 == data_type ||
      onnx::TensorProto_DataType_INT8 == data_type ||
      onnx::TensorProto_DataType_INT16 == data_type ||
      onnx::TensorProto_DataType_UINT16 == data_type ||
      onnx::TensorProto_DataType_BFLOAT16 == data_type) {
    // When xcompiler is correctly anchored to the `fix` op, we need to return
    // itself;
    ret = AnchorPoint::create(pass, node_get_output_node_arg(*onnx_node),
                              {AnchorPoint::IDENTITY_OP});
  } else if (node_op_type(*onnx_node) == "DequantizeLinear") {
    // when xcompiler wrongly anchor to a `fix` op, we need to cancel fix2float;

    // The last op of DequantizeLinear is not necessarily QuantizeLinear, such
    // as Squeeze(fix), now it is judged that as long as uint8 or int8 can be
    // output
    auto inputs = node_get_inputs(*onnx_node);
    CHECK_LE(inputs.size(), 3u);
    auto input = inputs[0];
    // remove this check becasue  DequantizeLinear input[0] maybe is graph
    // input, graph input is node_arg , now is GraphInput -> DequantizeLinear
    // not QuantizeLinear -> DequantizeLinear
    if (input.node == nullptr) {
      return nullptr;
    }
    auto input_data_type = node_arg_get_element_type(*input.node_arg);
    if (onnx::TensorProto_DataType_UINT8 == input_data_type ||
        onnx::TensorProto_DataType_INT8 == input_data_type ||
        onnx::TensorProto_DataType_INT16 == input_data_type ||
        onnx::TensorProto_DataType_UINT16 == input_data_type) {
      ret = AnchorPoint::create(pass, *input.node_arg,
                                {AnchorPoint::IDENTITY_OP});
    }
  } else {
    // in case that xcompiler does not anchor to a `fix` op, we try
    // out best to guess if the node is followed by a sole consumers
    // and the consumer is QuantizeLinear.
    auto consumers = graph_get_consumer_nodes(onnx_dot_onnx_graph,
                                              node_get_output_name(*onnx_node));
    if (consumers.size() == 1u) {
      // in case xcompiler anchor to an op other than `fix`, we dont need to
      // cancel out fix2float.
      if (node_op_type(*consumers[0]) == "QuantizeLinear") {
        ret = AnchorPoint::create(pass, node_get_output_node_arg(*consumers[0]),
                                  {AnchorPoint::IDENTITY_OP});
      }
    }
  }
  return ret;
}

struct PadCheckingException : public std::exception {
public:
  virtual const char* what() const throw() { return m.c_str(); }
  std::string m;
};

static void throw_exception(const std::string& msg) {
  PadCheckingException e;
  e.m = msg;
  throw e;
}

/**
 * we support padd if and only if last dim need pad
 */
static std::pair<std::string, std::vector<int>>
is_pad_needed(const std::vector<int32_t>& shape, std::vector<int32_t>& stride) {
  auto size = shape.size();
  std::vector<int32_t> ret(size * 2, 0);
  if (size != stride.size()) {
    return std::make_pair("size is not same.", ret);
  }
  // for 3 dims,
  // shape = [2, 224, 3]
  // strides = [224*3, 3, 1],
  // we don't need pad
  //
  // for 4 dims,
  // shape = [2, 224,224, 3]
  // strides = [224*224*4,224*4, 4, 1],
  // when we need pad
  //
  // for 3 dims
  // shape = [2,224, 3]
  // strides = [224*4, 4, 1],
  // then we need pad
  //
  // for 2 dims
  // shape = [2, 3]
  // strides = [2*4, 1],
  // then we need pad
  //
  // for 1 dims
  // shape = [4]
  // strides = [1,2],
  // then we need pad
  // so that we don't support dim=1
  if (size < 2) {
    return std::make_pair("shape dim >=2.", ret);
  }
  if (stride[size - 1u] != 1) {
    return std::make_pair("first element is not one.", ret);
  }
  auto sum = stride[size - 2]; // initialize the sum
  for (auto i = size - 2; i > 0; i--) {
    sum = sum * shape[i];      // sum = 300, 300*299, 300*299*3,
                               // only support padding at the last dimentions.
    if (sum != stride[i - 1]) {
      std::string msg = "stride[" + std::to_string(i - 1) + "] expect " +
                        std::to_string(sum) + ",but got " +
                        std::to_string(stride[i - 1]);
      return std::make_pair(msg, ret);
    }
  }
  // when size = 2, ret[3] = stride[0] - shape[1]
  // when size = 1, ret[1] = stride[-1] - shape[1],  so we don't support dim=1
  ret[2 * size - 1] = stride[size - 2] - shape[size - 1];
  return std::make_pair("", ret);
}

static std::pair<bool, std::vector<int>>
check_need_to_need_pad_last_dim_2(const xir::Tensor* tensor) {
  auto stride = tensor->template get_attr<std::vector<int32_t>>("stride");
  auto shape = tensor->get_shape();
  auto [msg, padding] = is_pad_needed(shape, stride);
  if (msg != "") {
    throw_exception(msg + " " + tensor->to_string());
  }
  auto ret = false;
  if (padding[padding.size() - 1] != 0) {
    ret = true;
  }
  return std::make_pair(ret, padding);
}
static std::pair<bool, std::vector<int>>
check_need_to_need_pad_last_dim(const xir::Subgraph* subgraph,
                                const xir::Tensor* tensor,
                                TensorBufferType tensor_type) {
  auto ret = false;
  std::vector<int> padding;
  if (tensor_type == vaip_core::XIR_INPUT) {
    if (ENV_PARAM(XLNX_ENABLE_GRAPH_ENGINE_PAD) == 0) {
      auto fanout_ops = tensor->get_producer()->get_fanout_ops();
      for (auto& op : fanout_ops) {
        if (subgraph->has_op(op)) {
          if (op->get_type() == "upload" &&
              op->get_output_tensor()->has_attr("stride")) {
            std::tie(ret, padding) =
                check_need_to_need_pad_last_dim_2(op->get_output_tensor());
            break;
          }
        }
      }
    }
  } else if (tensor_type == vaip_core::XIR_OUTPUT) {
    if (tensor->get_producer()->get_type() == "download") {
      // if XLNX_ENABLE_GRAPH_ENGINE_DEPAD == 0:check fail and assign to cpu:
      // if XLNX_ENABLE_GRAPH_ENGINE_DEPAD > 0 ： graph-engine do depad
      if (ENV_PARAM(XLNX_ENABLE_GRAPH_ENGINE_DEPAD) == 0) {
        auto input_tensors = tensor->get_producer()->get_input_tensors();
        CHECK_EQ(input_tensors.size(), 1);
        if (input_tensors[0]->has_attr("stride")) {
          std::tie(ret, padding) =
              check_need_to_need_pad_last_dim_2(input_tensors[0]);
        }
      } else {
        return {false, {}};
      }
    }
  } else {
    LOG(FATAL) << "unsupported tensor_type " << tensor_type;
  }
  return std::make_pair(ret, padding);
}

static std::unique_ptr<AnchorPoint>
append_pad_with_channel_last_dim(const IPass& pass, const AnchorPoint* ret,
                                 const std::string& tensor_name,
                                 const std::vector<int> padding) {
  // pad at last dim,append 0 at the end
  AnchorPointPadOpAttr pad_attr;
  pad_attr.mutable_paddings()->Assign(padding.begin(), padding.end());
  return AnchorPoint::create(pass, ret->get_proto(), tensor_name,
                             {"pad", pad_attr});
}

bool DPUSubgraphProcessor::is_shape_compatible(
    const std::string& tensor_name_on_xir_xmodel,
    const std::string& tensor_name_on_compiled_xmodel) {
  auto tensor_on_xir_xmodel =
      xir_xmodel_->get_tensor(tensor_name_on_xir_xmodel);
  auto tensor_on_compiled_xmodel =
      compiled_xmodel_->get_tensor(tensor_name_on_compiled_xmodel);
  if (tensor_on_xir_xmodel == nullptr || tensor_on_compiled_xmodel == nullptr) {
    // test case: ipu.modelzoo, model id: efficientformerv2_s0.snap_dist_in1k
    LOG(WARNING)
        << "tensor_name_on_xir_xmodel or tensor_name_on_compiled_xmodel not "
           "found, tensor_name_on_xir_xmodel: "
        << tensor_name_on_xir_xmodel
        << " tensor_name_on_compiled_xmodel: " << tensor_name_on_compiled_xmodel
        << ". ";
    report_comment_
        << "tensor_name_on_xir_xmodel or tensor_name_on_compiled_xmodel not "
           "found, tensor_name_on_xir_xmodel: "
        << tensor_name_on_xir_xmodel
        << " tensor_name_on_compiled_xmodel: " << tensor_name_on_compiled_xmodel
        << ". ";
    return false;
  }
  auto shape_on_xir_xmodel = tensor_on_xir_xmodel->get_shape();
  auto shape_on_compiled_xmodel = tensor_on_compiled_xmodel->get_shape();
  CHECK(!shape_on_xir_xmodel.empty());
  CHECK(!shape_on_compiled_xmodel.empty());
  // we assume the first dimention is the batch size.
  auto ret = shape_on_xir_xmodel[0] == shape_on_compiled_xmodel[0];
  if (!ret) {
    LOG(WARNING) << "please check xcompiler. tensor shapes are not compatible:"
                 << " tensor on xir.xmodel: "
                 << tensor_on_xir_xmodel->to_string()
                 << " tensor on compiled.xmodel: "
                 << tensor_on_compiled_xmodel->to_string();
    report_comment_
        << "please check xcompiler. tensor shapes are not compatible:"
        << " tensor on xir.xmodel: " << tensor_on_xir_xmodel->to_string()
        << " tensor on compiled.xmodel: "
        << tensor_on_compiled_xmodel->to_string();
    return ret;
  }
  if (ENV_PARAM(XLNX_ENABLE_BATCH)) {
    ret = ret && (shape_on_compiled_xmodel[0] == 1);
    if (!ret) {
      LOG(WARNING) << "please check xcompiler."
                   << " shape_on_compiled_xmodel[0] should be == 1."
                   << " tensor on compiled.xmodel: "
                   << tensor_on_compiled_xmodel->to_string();
      report_comment_ << "please check xcompiler."
                      << " shape_on_compiled_xmodel[0] should be == 1."
                      << " tensor on compiled.xmodel: "
                      << tensor_on_compiled_xmodel->to_string();
    }
  }
  return ret;
}

static bool node_arg_is_graph_input(const Graph& graph,
                                    const std::string& node_arg_name) {
  auto graph_inputs = graph_get_inputs(graph);
  bool ret = std::any_of(graph_inputs.begin(), graph_inputs.end(),
                         [&node_arg_name](const NodeArg* node_arg) -> bool {
                           return node_arg_get_name(*node_arg) == node_arg_name;
                         });
  return ret;
}
static bool node_arg_check_data_type(const Graph& graph,
                                     const std::string& node_arg_name) {
  auto node_arg = VAIP_ORT_API(graph_get_node_arg)(graph, node_arg_name);
  CHECK(node_arg != nullptr) << "cannot find node arg: " << node_arg_name;
  auto element_type = node_arg_get_element_type(*node_arg);
  return element_type == onnx::TensorProto_DataType_INT8 ||
         element_type == onnx::TensorProto_DataType_UINT8 ||
         element_type == onnx::TensorProto_DataType_UINT16 ||
         element_type == onnx::TensorProto_DataType_INT16;
}
static bool
node_arg_is_graph_input_and_check_data_type(const Graph& graph,
                                            const std::string& node_arg_name) {
  return node_arg_is_graph_input(graph, node_arg_name) &&
         node_arg_check_data_type(graph, node_arg_name);
}
static bool is_graph_input_to_dq(const Graph& graph,
                                 const std::string& node_arg_name) {
  auto onnx_node = VAIP_ORT_API(graph_producer_node)(graph, node_arg_name);
  // origin node is DequanzerLinear op
  if (onnx_node == nullptr) {
    return false;
  }
  if (node_op_type(*onnx_node) != "DequantizeLinear") {

    return false;
  }
  auto inputs = node_get_inputs(*onnx_node);
  auto input = inputs[0]; // graph input
  auto input_node_arg_name = node_arg_get_name(*input.node_arg);
  // DequantizerLinear input is graph input
  auto ret =
      node_arg_is_graph_input_and_check_data_type(graph, input_node_arg_name);
  MY_LOG(1) << "found anchor point to graph input, Graph Input -> "
               "DequantizeLinear, node_arg_name : "
            << node_arg_name;
  return ret;
}
// fillin onnx.onnx -> compiled.xmodel anchor_point
std::unique_ptr<AnchorPoint>
DPUSubgraphProcessor::create_anchor_point_by_xir_tensor(
    const IPass& pass, const onnxruntime::Graph& onnx_dot_onnx_graph,
    const xir::Tensor* tensor, TensorBufferType tensor_type,
    const xir::Subgraph* subgraph) {
  MY_LOG(1) << "search anchor point for xir tensor: " << tensor->get_name();
  // TODO : pad-fix
  std::string new_tensor_name = "";
  if (ENV_PARAM(XLNX_ENABLE_OP_NAME_PROTECTION)) {
    std::string head = "(";
    std::string tail = ")";
    new_tensor_name =
        tensor_name_remove_xir_suffix_protect(tensor->get_name(), head, tail);
  } else {
    new_tensor_name = tensor_name_remove_xir_suffix(tensor->get_name());
  }
  auto ret_is_shape_compatible =
      is_shape_compatible(new_tensor_name, tensor->get_name());
  if (!ret_is_shape_compatible) {
    // if shape is not compatible it means that when we map a
    // tensor name on `compiled.xmodel` to a tensor name on
    // `xir.xmodel`, the shape is not same. xcompiler needs to
    // guarantee the shapes should be same. For now, we only
    // check whether the first dimention is same or not, it
    // might still be risky depending on xcompiler's
    // implementation.
    return nullptr;
  }
  if (new_tensor_name != tensor->get_name()) {
    // it is possible that the alias already exists, because when
    // DPU subgraph connects to a DPU subgraph,
    //
    // 1. when processing the first DPU subgraph, the output of
    // tensor alias for download op
    //
    // 2. when process the second DPU subgraph, the input tensor
    // alias shares the same `new_tensor_name` after removing xir
    // suffix
    if (AnchorPoint::find_anchor_point(pass, tensor->get_name()) == nullptr) {
      auto alias = AnchorPoint::alias1(pass, onnx_dot_onnx_graph,
                                       new_tensor_name, tensor->get_name());
      alias->insert_into_context(const_cast<IPass&>(pass));
    }
  }
  // onnx.onnx graph -> compiled.xmodel
  auto ret = AnchorPoint::find_anchor_point(
      const_cast<IPass&>(pass), onnx_dot_onnx_graph, tensor->get_name());

  // for example, in issue #560, the following log is printed out
  // anchor point to `fix` OP:
  //   blob.20_vaip_12_inserted_fix_85 <-- transpose@layoutransform --
  //   blob.20
  if (ret == nullptr) {
    // test case: ipu.modelzoo, model id: efficientformerv2_s0.snap_dist_in1k
    MY_LOG(1) << "can't find anchor point, tensor_name: " << tensor->get_name();
    // it is a fatal error if not found, we cancel processing of this
    // subgraph, this dpu subgraph will be dispatched to CPU EP.
    return nullptr;
  }

  MY_LOG(1) << "anchor point to `fix` OP: " << ret->op_debug_string();

  auto [need_pad_channel_in_last_dim, padding] =
      check_need_to_need_pad_last_dim(subgraph, tensor, tensor_type);
  if (need_pad_channel_in_last_dim) {
    ret = append_pad_with_channel_last_dim(pass, ret.get(), tensor->get_name(),
                                           padding);
    MY_LOG(1) << "ret anchor point :  " << ret->op_debug_string();
  }

  auto q_dq_anchor_point = find_q_dq_anchor_point(pass, onnx_dot_onnx_graph,
                                                  ret->origin_node_arg_name());

  if (q_dq_anchor_point) {
    MY_LOG(1) << "find xir q_dq_anchor_point: "
              << q_dq_anchor_point->op_debug_string();
    // for example, in issue #560, find xir q_dq_anchor_point:
    //   onnx::DequantizeLinear_346 <-- identity@fuse_DPU --
    //   onnx::DequantizeLinear_346
    //
    //  onnx::DequantizeLinear_346 is output of the following node
    //
    //  [onnx::DequantizeLinear_327:(ty=3,shape=[1,64,56,62])]
    //  QuantizeLinear
    //  [onnx::QuantizeLinear_324:(ty=1,shape=[1,64,56,62]),ortshared_1_0_1_2_token_158:(ty=1,shape=[]),onnx::QuantizeLinear_326:(ty=3,shape=[])]
    //
  } else {
    MY_LOG(1) << "find xir q_dq_anchor_point: nullptr";
  }
  if (q_dq_anchor_point) {
    auto q_dq_origin_node_ptr = VAIP_ORT_API(graph_producer_node)(
        onnx_dot_onnx_graph, q_dq_anchor_point->origin_node_arg_name());
    CHECK(q_dq_origin_node_ptr != nullptr)
        << "logical error:" << q_dq_anchor_point->origin_node_arg_name();
    const auto& q_dq_origin_node = *q_dq_origin_node_ptr;
    // append QuantizeLinear node;
    ret = ret->append(pass, *q_dq_anchor_point);
    // because xcompiler hook to `fix` op wrongly, we need to correct
    // it as below, i.e. the origin node of `ret` is replaced
    //
    // ret =
    //     blob.20_vaip_12_inserted_fix_85 <-- identity@fuse_DPU --
    //     blob.20_vaip_12_inserted_fix_85 <-- transpose@layoutransform --
    //     onnx::DequantizeLinear_346 <-- identity@fuse_DPU --
    //     onnx::DequantizeLinear_346
    MY_LOG(1) << "find xir anchor point: "
              << "origin_node_name " << ret->origin_node_arg_name() << " " //
              << node_as_string(q_dq_origin_node)                          //
              << "new_tensor_name " << new_tensor_name << " "              //
              << "ret = " << ret->op_debug_string() << "\n"                //
              << "q_dq_anchor_point = " << ret->op_debug_string() << "\n";
  } else {

    if (is_graph_input_to_dq(onnx_dot_onnx_graph,
                             ret->origin_node_arg_name())) {
      // for Graph Input (int8/uint8) -> DequantizerLinear
      // is bewteen find QDQ and not find QDQ
      // anchor point origin node is DequantizerLinear and DequantizerLinear
      // input is graph input and graph input data type is int8/uint8
      auto onnx_node = VAIP_ORT_API(graph_producer_node)(
          onnx_dot_onnx_graph, ret->origin_node_arg_name());
      CHECK(onnx_node != nullptr); // onnx_node is DequantizerLinear op
      auto inputs = node_get_inputs(*onnx_node);
      auto input = inputs[0];      // input[0] is graph input

      ret = ret->append(pass, *AnchorPoint::create(pass, *input.node_arg,
                                                   {AnchorPoint::IDENTITY_OP}));

    } else if ( // new_tensor_name == tensor->get_name()  cancel this check
                // because new_tensor_name != tensor->get_name() when enable
                // protection. and this check include in node_arg_is_graph_input
                // , so can ben cancel
        new_tensor_name == ret->origin_node_arg_name() &&
        node_arg_is_graph_input_and_check_data_type(
            onnx_dot_onnx_graph, ret->origin_node_arg_name())) {
      // testcase , onnx model is int8/uint8 inputs and outputs
      // anchor point origin node is graph input
      // graph input element type must be int8 or uint8, becasue xir DPU
      // Subgraph input data is int8.
      // if graph input element type is uint8, XIR will treat uint8 data as int8
      // because the quantization tool has applied some tricky processing. so
      // graph input data need to make some tricky modifications similar to what
      // is done in the VAI_Q quantization tool. zero_point = -128
      MY_LOG(1) << "find anchor point to graph input " << tensor->get_name();
      // nothing to do

    } else if (tensor->has_attr("fix_point")) {
      auto fix_point = tensor->template get_attr<int>("fix_point");
      AnchorPointFixAttr attr;
      attr.set_fix_point(fix_point);
      // testcase: dla102(disable convert_ending_blacklist_ops_to_unknown_op
      // pass),#1048
      // testcase: HP YV8Seg_int_io.onnx
      // /proj/xcdhdstaff2/yiminghu/share/hp_ipu/BBM2_0906_anchor_point_debug/YV8Seg_int_io.onnx
      ret = AnchorPoint::create(pass, ret->get_proto(), tensor->get_name(),
                                {"float2fix", attr});
    } else {
      // testcase:
      // /group/dphi_software/software/workspace/runfengw/ipu/MEP_models/Model_C1/Model_C1.quant.onnx
      // The upload/download of quantize-linear/quantize-linear op has no
      // fix-point
      return nullptr;
    }
  }
  // cannot insert it into database, otherwise, duplicated anchor point
  // detected.
  //
  // ret->insert_into_context(const_cast<IPass&>(pass));
  MY_LOG(1) << "found anchor point for xir tensor: " << tensor->get_name()
            << ";" << ret->op_debug_string();
  return ret;
} // namespace vaip_level1_dpu

std::vector<std::unique_ptr<AnchorPoint>>
DPUSubgraphProcessor::create_anchor_points_by_xir_tensors(
    const IPass& pass, const onnxruntime::Graph& graph,
    const std::vector<const xir::Tensor*>& tensors,
    TensorBufferType tensor_type, const xir::Subgraph* subgraph) {
  auto ret = std::vector<std::unique_ptr<AnchorPoint>>{};
  std::transform(
      tensors.begin(), tensors.end(), std::back_inserter(ret),
      [&graph, &pass, &tensor_type, &subgraph,
       this](const xir::Tensor* tensor) {
        auto anchor_point = create_anchor_point_by_xir_tensor(
            pass, graph, tensor, tensor_type, subgraph);
        if (anchor_point == nullptr) {
          return std::unique_ptr<AnchorPoint>{nullptr};
        }
        CHECK_EQ(tensor->get_name(), anchor_point->get_proto().name())
            << anchor_point->op_debug_string();
        return std::unique_ptr<AnchorPoint>{std::move(anchor_point)};
      });
  return ret;
}

static bool
any_of_is_nullptr(const std::vector<std::unique_ptr<AnchorPoint>>& v) {
  return std::find(v.begin(), v.end(), nullptr) != v.end();
}

static std::vector<TensorBufferParam>
create_xir_tensor_buffers(const std::vector<const xir::Tensor*>& tensors,
                          TensorBufferType type) {
  auto ret = std::vector<TensorBufferParam>{};
  std::transform(tensors.begin(), tensors.end(), std::back_inserter(ret),
                 [&type](const xir::Tensor* tensor) {
                   auto tb = TensorBufferParam();
                   tb.set_tb_type(type);
                   tb.set_tensor_name(tensor->get_name());
                   return tb;
                 });
  return ret;
}

static TensorBufferParam
create_onnx_tensor_buffer(const onnxruntime::Graph& graph,
                          const std::string& node_arg_name, int idx,
                          TensorBufferType type) {
  auto tb = TensorBufferParam();
  tb.set_tb_type(type);
  tb.set_tensor_name(node_arg_name);
  tb.set_onnx_index(idx);
  if (type == vaip_core::ONNX_OUTPUT) {
    auto node_arg = VAIP_ORT_API(graph_get_node_arg)(graph, node_arg_name);
    auto onnx_shape = node_arg_get_shape_i64(*node_arg);
    CHECK(onnx_shape != nullptr)
        << node_arg_as_string(*node_arg) << " shape absent";
    tb.mutable_onnx_shape()->CopyFrom({onnx_shape->begin(), onnx_shape->end()});
  }
  return tb;
}
static std::vector<TensorBufferParam> create_onnx_tensor_buffers(
    const onnxruntime::Graph& graph,
    const google::protobuf::RepeatedPtrField<std::string>& node_arg_names,
    TensorBufferType type) {
  auto ret = std::vector<TensorBufferParam>();
  ret.reserve(node_arg_names.size());
  auto idx = 0;
  for (auto name : node_arg_names) {
    ret.push_back(create_onnx_tensor_buffer(graph, name, idx++, type));
  }
  return ret;
}

static void add_tensor_info_to_report(
    const std::vector<TensorBufferParam>& buffers,
    google::protobuf::RepeatedPtrField<vaip_core::XirTensorProto>* report) {
  for (const auto& buffer : buffers) {
    vaip_core::XirTensorProto new_item;
    new_item.set_name(buffer.tensor_name());
    new_item.mutable_shape()->Add(buffer.onnx_shape().begin(),
                                  buffer.onnx_shape().end());
    report->Add()->CopyFrom(new_item);
  }
}

static void add_anchor_point_to_report(
    const std::vector<std::unique_ptr<AnchorPoint>>& anchor_points,
    google::protobuf::RepeatedPtrField<vaip_core::AnchorPointProto>* report) {
  for (const auto& anchor_point : anchor_points) {
    if (!anchor_point) {
      vaip_core::AnchorPointProto new_item;
      new_item.set_name("nullptr");
      report->Add()->CopyFrom(new_item);
    } else {
      report->Add()->CopyFrom(anchor_point->get_proto());
    }
  }
}

static std::vector<std::string> get_origin_names_by_anchor_points(
    const std::vector<std::unique_ptr<AnchorPoint>>& my_anchor_points) {
  auto name_set = std::set<std::string>();
  for (auto& ap : my_anchor_points) {
    name_set.emplace(ap->origin_node_arg_name());
  }
  auto ret = std::vector<std::string>();
  ret.reserve(name_set.size());
  for (auto name : name_set) {
    ret.push_back(name);
  }
  return ret;
}

const TensorBufferParam*
find_tb_by_tensor_name(const std::vector<vaip_core::TensorBufferParam>& tbs,
                       const std::string& tensor_name) {
  const TensorBufferParam* ret = nullptr;
  auto it =
      std::find_if(tbs.begin(), tbs.end(), [&tensor_name](const auto& tb) {
        return tb.tensor_name() == tensor_name;
      });
  if (it != tbs.end()) {
    ret = &*it;
  }
  return ret;
}

static int
idx_anchor_point(const std::vector<std::unique_ptr<AnchorPoint>>& anchor_points,
                 const std::string& origin_name) {

  auto it =
      std::find_if(anchor_points.begin(), anchor_points.end(),
                   [&origin_name](const auto& anchor_point) {
                     return anchor_point->origin_node_arg_name() == origin_name;
                   });
  auto ret = it - anchor_points.begin();
  if (it == anchor_points.end()) {
    ret = -1;
  }
  return (int)ret;
}

void schedule_fill_in_attr(MetaSchedule* schedule,
                           const AnchorPointProto* proto) {
  if (proto->op_type() == "pad") {
    schedule->mutable_op()->set_is_pad(true);
    *schedule->mutable_op()->mutable_padding() =
        proto->attribute().pad_attr().paddings();
  } else if (proto->op_type() == "transpose") {
    schedule->mutable_op()->set_is_layout_transform(true);
    *schedule->mutable_op()->mutable_order() =
        proto->attribute().transpose_attr().order();
  } else if (proto->op_type() == "float2fix") {
    schedule->mutable_op()->set_float2fix(true);
    schedule->mutable_op()->set_fix_point(
        (int32_t)proto->attribute().fix_attr().fix_point());
  } else if (proto->op_type() == "fix2float") {
    schedule->mutable_op()->set_fix2float(true);
    schedule->mutable_op()->set_fix_point(
        (int32_t)proto->attribute().fix_attr().fix_point());
  } else if (proto->op_type() == "quantize_linear") {
    schedule->mutable_op()->set_quantize_linear(true);
    schedule->mutable_op()->set_scale(proto->attribute().qdq_attr().scale());
    schedule->mutable_op()->set_zero_point(
        proto->attribute().qdq_attr().zero_point());
  } else if (proto->op_type() == "dequantize_linear") {
    schedule->mutable_op()->set_dequantize_linear(true);
    schedule->mutable_op()->set_scale(proto->attribute().qdq_attr().scale());
    schedule->mutable_op()->set_zero_point(
        proto->attribute().qdq_attr().zero_point());
  } else if (proto->op_type() == "identity") {
    // nothing to do
  } else if (proto->op_type() == "transpose_immune") {
    // This type is generated by create_broadcast_op_transpose_immune_rule.It
    // unsqueeze the dimensions of the original data.It has no effect on the
    // layout of the data
  } else {
    LOG(FATAL) << "not support op type : " << proto->op_type();
  }
}

static std::vector<int64_t>
reverse_order(const google::protobuf::RepeatedField<int64_t>& order) {
  std::vector<int64_t> ret;
  ret.resize(order.size());
  for (auto index = 0; index < order.size(); ++index) {
    ret[order[index]] = index;
  }
  return ret;
}

static void schedule_fill_in_attr_reverse(MetaSchedule* schedule,
                                          const AnchorPointProto* proto) {

  if (proto->op_type() == "transpose") {
    schedule->mutable_op()->set_is_layout_transform(true);
    auto order = reverse_order(proto->attribute().transpose_attr().order());
    schedule->mutable_op()->mutable_order()->CopyFrom(
        {order.begin(), order.end()});
  } else if (proto->op_type() == "float2fix") {
    schedule->mutable_op()->set_fix2float(true);
    schedule->mutable_op()->set_fix_point(
        (int32_t)proto->attribute().fix_attr().fix_point());
  } else if (proto->op_type() == "fix2float") {
    schedule->mutable_op()->set_float2fix(true);
    schedule->mutable_op()->set_fix_point(
        (int32_t)proto->attribute().fix_attr().fix_point());
  } else if (proto->op_type() == "quantize_linear") {
    schedule->mutable_op()->set_dequantize_linear(true);
    schedule->mutable_op()->set_scale(proto->attribute().qdq_attr().scale());
    schedule->mutable_op()->set_zero_point(
        proto->attribute().qdq_attr().zero_point());
  } else if (proto->op_type() == "dequantize_linear") {
    schedule->mutable_op()->set_quantize_linear(true);
    schedule->mutable_op()->set_scale(proto->attribute().qdq_attr().scale());
    schedule->mutable_op()->set_zero_point(
        proto->attribute().qdq_attr().zero_point());
  } else if (proto->op_type() == "identity") {
    // nothing to do
  } else if (proto->op_type() == "pad") {
    bool vaip_do_depad = ENV_PARAM(XLNX_ENABLE_OUTPUT_DEPAD) == 1;
    if (vaip_do_depad) {

    } else {
      // nothing to do
    }
  } else if (proto->op_type() == "transpose_immune") {
    // This type is generated by create_broadcast_op_transpose_immune_rule.It
    // unsqueeze the dimensions of the original data.It has no effect on the
    // layout of the data
  } else {
    LOG(FATAL) << "not support op type : " << proto->op_type();
  }
}

static bool need_reverse(const vaip_core::TensorBufferParam& from_tb,
                         const std::string& origin_name) {
  return from_tb.tb_type() == vaip_core::XIR_INPUT ||
         from_tb.tb_type() == vaip_core::XIR_OUTPUT ||
         from_tb.tensor_name() != origin_name;
}

static MetaSchedule create_schedule(const IPass& pass,
                                    const vaip_core::TensorBufferParam& from_tb,
                                    const vaip_core::TensorBufferParam& to_tb,
                                    const AnchorPoint* ap) {
  auto ret = MetaSchedule{};
  *ret.mutable_from_tb_param() = from_tb;
  *ret.mutable_to_tb_param() = to_tb;
  auto anchor_point = ap->optimize(pass);
  if (need_reverse(from_tb, anchor_point->origin_node_arg_name())) {
    anchor_point->for_each([&ret](const AnchorPointProto& p) {
      schedule_fill_in_attr_reverse(&ret, &p);
    });
  } else {
    anchor_point->for_each(
        [&ret](const AnchorPointProto& p) { schedule_fill_in_attr(&ret, &p); });
  }
  return ret;
}

std::vector<MetaSchedule> create_input_meta_schedules(
    const IPass& pass,
    const std::vector<vaip_core::TensorBufferParam>& onnx_input_tbs,
    const std::vector<vaip_core::TensorBufferParam>& xir_input_tbs,
    const std::vector<std::unique_ptr<AnchorPoint>>& anchor_points) {
  auto ret = std::vector<MetaSchedule>{};
  for (auto i = 0u; i < xir_input_tbs.size(); ++i) {
    auto& anchor_point = anchor_points[i];
    auto onnx_input_tb = find_tb_by_tensor_name(
        onnx_input_tbs, anchor_point->origin_node_arg_name());
    CHECK(onnx_input_tb != nullptr);
    ret.push_back(create_schedule(pass, *onnx_input_tb, xir_input_tbs[i],
                                  anchor_point.get()));
  }
  return ret;
}

static std::vector<MetaSchedule> create_onnx_output_meta_schedules(
    const IPass& pass,
    const std::vector<vaip_core::TensorBufferParam>& xir_output_tbs,
    const std::vector<vaip_core::TensorBufferParam>& onnx_input_tbs,
    const std::vector<vaip_core::TensorBufferParam>& onnx_output_tbs,
    const std::vector<std::unique_ptr<AnchorPoint>>&
        onnx_output_anchor_points) {
  auto ret = std::vector<MetaSchedule>{};
  ret.resize(onnx_output_tbs.size());
  for (auto i = 0u; i < onnx_output_tbs.size(); ++i) {
    auto& onnx_output_tb = onnx_output_tbs[i];
    auto& onnx_output_anchor_point = onnx_output_anchor_points[i];

    auto from_tb = find_tb_by_tensor_name(
        xir_output_tbs, onnx_output_anchor_point->get_proto().name());
    if (from_tb == nullptr) {
      from_tb = find_tb_by_tensor_name(
          onnx_input_tbs, onnx_output_anchor_point->get_proto().name());
    }
    CHECK(from_tb != nullptr);
    ret[i] = create_schedule(pass, *from_tb, onnx_output_tb,
                             onnx_output_anchor_points[i].get());
  }
  return ret;
}

static std::vector<const Node*> map_anchor_points_to_onnx_nodes(
    const Graph& graph,
    const std::vector<std::unique_ptr<AnchorPoint>>& anchor_points) {
  auto ret = std::vector<const Node*>{};
  for (auto& ap : anchor_points) {
    auto origin_node_arg_name = ap->origin_node_arg_name();
    auto node = VAIP_ORT_API(graph_producer_node)(graph, origin_node_arg_name);
    if (node) { // maybe anchor_points is graph inputs
      // CHECK(node != nullptr) << "cannot find : " << origin_node_arg_name;
      ret.push_back(node);
    }
  }
  return ret;
}

DPUSubgraphProcessor::DPUSubgraphProcessor(onnxruntime::Graph& onnx_graph,
                                           IPass& self, xir::Graph* xir_xmodel,
                                           xir::Graph* compiled_xmodel)
    : onnx_graph_{onnx_graph}, self_{self}, xir_xmodel_{xir_xmodel},
      compiled_xmodel_{compiled_xmodel} {}

std::unique_ptr<std::vector<const Node*>>
calculate_path(const Node* node, const std::vector<const Node*>& end_nodes) {
  auto ret = std::vector<const Node*>{node};
  auto is_node_supported = [](const Node* n) {
    const char* supported_op[] = {"QuantizeLinear", "DequantizeLinear",
                                  "Transpose"};
    const char* supported_custom_op[] = {"VitisQuantizeLinear",
                                         "VitisDequantizeLinear"};
    auto ret = false;
    for (auto op_type : supported_op) {
      ret = ret || node_is_op(*n, op_type, "ai.onnx");
    }
    for (auto op_type : supported_custom_op) {
      ret = ret || node_is_op(*n, op_type, "com.vai.quantize");
    }
    return ret;
  };
  for (auto inputs = node_get_inputs(*ret.back());
       inputs.size() >= 1 &&
       inputs[0].node != nullptr /* only support SISO, "QuantizeLinear" has two
                                    optional constant inputs.*/
       ;) {
    auto found = std::find(end_nodes.begin(), end_nodes.end(),
                           inputs[0].node) != end_nodes.end();
    if (found) {
      ret.push_back(inputs[0].node);
      break;
    }
    auto ok = is_node_supported(inputs[0].node); // only support Q,DQ,Transpose

    if (ok) {
      ret.push_back(inputs[0].node);
      inputs = node_get_inputs(*ret.back());
    } else {
      // test case: ipu.modelzoo, id: cspresnet50
      MY_LOG(1) << "not supported: " << node_as_string(*inputs[0].node);
      return nullptr;
    }
  }
  return std::make_unique<std::vector<const Node*>>(ret);
}

static const std::unique_ptr<AnchorPoint>& find_known_anchor_points(
    const std::vector<std::unique_ptr<AnchorPoint>>& known_aps,
    const std::string& origin_name) {
  auto it = std::find_if(known_aps.begin(), known_aps.end(),
                         [&origin_name](const auto& ap) {
                           return ap->origin_node_arg_name() == origin_name;
                         });
  CHECK(it != known_aps.end());
  return *it;
}

static std::vector<std::unique_ptr<AnchorPoint>>
fill_in_onnx_output_tb_anchor_point(
    const IPass& pass, const Graph& onnx_graph_,
    const std::vector<vaip_core::TensorBufferParam>& xir_output_tbs,
    const std::vector<vaip_core::TensorBufferParam>& onnx_input_tbs,
    std::vector<vaip_core::TensorBufferParam>& onnx_output_tbs,
    const std::vector<std::unique_ptr<AnchorPoint>>& xir_output_anchor_points) {

  auto ret = std::vector<std::unique_ptr<AnchorPoint>>();
  ret.reserve(onnx_output_tbs.size());
  auto known_anchor_points = std::vector<std::unique_ptr<AnchorPoint>>();
  known_anchor_points.reserve(xir_output_anchor_points.size() +
                              onnx_input_tbs.size());
  for (auto& ap : xir_output_anchor_points) {
    known_anchor_points.push_back(
        std::unique_ptr<AnchorPoint>{AnchorPoint::create(ap->get_proto())});
  }
  for (auto& onnx_input_tb : onnx_input_tbs) {
    auto ap = AnchorPoint::find_anchor_point(
        const_cast<IPass&>(pass), onnx_graph_, onnx_input_tb.tensor_name());
    CHECK(ap != nullptr);
    CHECK_EQ(onnx_input_tb.tensor_name(), ap->get_proto().name());
    known_anchor_points.push_back(std::unique_ptr<AnchorPoint>{std::move(ap)});
  }
  // todo onnx_output-> onnx_output
  auto end_nodes =
      map_anchor_points_to_onnx_nodes(onnx_graph_, known_anchor_points);
  for (auto& tb : onnx_output_tbs) {

    // onnx_output origin_name is equals tensor_name (with in onnx.onnx)
    auto idx = idx_anchor_point(known_anchor_points, tb.tensor_name());
    if (idx >= 0) {
      ret.push_back(std::unique_ptr<AnchorPoint>{
          AnchorPoint::create(known_anchor_points[idx]->get_proto())});
    } else {
      auto begin_node =
          VAIP_ORT_API(graph_producer_node)(onnx_graph_, tb.tensor_name());
      CHECK(begin_node != nullptr) << "cannot find node: " << tb.tensor_name();
      auto path_nodes_ptr = calculate_path(begin_node, end_nodes);
      if (path_nodes_ptr == nullptr) {
        MY_LOG(1) << "path nodes is nullptr, " << node_as_string(*begin_node);
        ret.push_back(std::unique_ptr<AnchorPoint>{nullptr});
        break;
      }
      auto path_nodes = *path_nodes_ptr.get();
      for (auto& node : path_nodes) {
        LOG(INFO) << "path ==== "
                  << node_arg_get_name(node_get_output_node_arg(*node));
      }
      auto anchor_point =
          AnchorPoint::create_from_siso_path(pass, onnx_graph_, path_nodes);
      // there is a ancher point link, from one of `end_nodes` (E) to a
      // TensorBufferParam (G) so that G's anchor point is
      //
      //     E -> ... -> G
      //
      // path_nodes, means that is a path on the origin graph between E and T
      // (the residual output onnx TB.)
      //
      //     E -> .. -> T
      //
      // we create a *reverse* anchor point, `anchor_point` via `path_nodes`,
      // means
      //
      //    T -> ... -> E
      //
      // we get combined `combine_anchor_point` via `AnchorPoint::append`,
      // we get an anchor point path
      //
      //    T -> ... -> ... G
      //
      // now we know how to create a meta schedual from G to T.
      //
      MY_LOG(1) << "anchor_point = " << anchor_point->op_debug_string() << "\n"
                << anchor_point->get_proto().DebugString();
      auto path_end_node_arg_name = node_get_output_name(*path_nodes.back());
      auto& known_anchor_point =
          find_known_anchor_points(known_anchor_points, path_end_node_arg_name);
      MY_LOG(1) << "known anchor_point "
                << known_anchor_point->op_debug_string() << "\n"
                << known_anchor_point->get_proto().DebugString();
      auto combine_anchor_point =
          known_anchor_point->append(pass, *anchor_point);
      CHECK_EQ(combine_anchor_point->get_proto().name(),
               known_anchor_point->get_proto().name());
      MY_LOG(1) << "combinded anchor_point = "
                << combine_anchor_point->op_debug_string() << "\n"
                << combine_anchor_point->get_proto().DebugString();
      ret.push_back(
          std::unique_ptr<AnchorPoint>{std::move(combine_anchor_point)});
    }
  }

  return ret;
}

ProcessInfo
DPUSubgraphProcessor::find_xir_anchor_point(const xir::Subgraph* subgraph) {
  try {
    return find_xir_anchor_point2(subgraph);
  } catch (const std::exception& e) {
    LOG(WARNING) << "Unexcepted exception: "
                 << "(Exception type: " << typeid(e).name() << ") " << e.what();
    report_.set_status(std::string("unknown error"));
    report_.set_comments(std::string(e.what()));
    return ProcessInfo(false);
  }
}

std::unique_ptr<MetaDefProto>
DPUSubgraphProcessor::process(const xir::Subgraph* subgraph,
                              const ProcessInfo& above_context) {
  try {
    return process_internal(subgraph, above_context);
  } catch (const std::exception& e) {
    LOG(WARNING) << "Unexcepted exception: "
                 << "(Exception type: " << typeid(e).name() << ") " << e.what();
    report_.set_status(std::string("unknown error"));
    report_.set_comments(std::string(e.what()));
    return nullptr;
  }
}

bool DPUSubgraphProcessor::check_4d_dimention_tensors(
    const std::vector<const xir::Tensor*>& tensors) {
  bool ret = false;
  if (std::all_of(tensors.begin(), tensors.end(),
                  [](const xir::Tensor* tensor) {
                    return tensor->get_shape().size() == 4;
                  })) {
    ret = true;
  }
  report_comment_ << tensors[0]->to_string();
  return ret;
}

bool DPUSubgraphProcessor::check_4d_dimention(const xir::Subgraph* subgraph) {
  bool input_ok =
      check_4d_dimention_tensors(subgraph->get_sorted_input_tensors());
  bool output_ok =
      check_4d_dimention_tensors(subgraph->get_sorted_output_tensors());
  return input_ok && output_ok;
}

static bool check_xir_sg_nested(onnxruntime::Graph& onnx_graph_,
                                const std::unique_ptr<MetaDefProto>& meta_def) {
  // This is coupled with fused op type,Although this op type is provided in
  // ort-provider
  for (int i = 0; i < meta_def->nodes_size(); i++) {
    auto onnx_node_arg_name = meta_def->nodes(i);
    auto node =
        VAIP_ORT_API(graph_producer_node)(onnx_graph_, onnx_node_arg_name);
    if (node) {
      if (node_is_op(*node, "super_layer", "com.xilinx")) {
        return true;
      }
    } else {
      LOG(FATAL) << onnx_node_arg_name << " producer node is nullptr";
    }
  }
  return false;
}

ProcessInfo
DPUSubgraphProcessor::find_xir_anchor_point2(const xir::Subgraph* subgraph) {
  // change to sorted because potential errors introduced by different order
  report_.set_subgrpah_name(subgraph->get_name());
  if (subgraph->get_sorted_input_tensors().size() == 0) {
    MY_LOG(1) << "cancel subgraph processing , DPU subgraph has no upload op. "
                 "DPU subgraph : "
              << subgraph->get_name();
    report_.set_status("xir_dpu_subgraph_no_upload_ops");
    return ProcessInfo(false);
  }
  auto xir_input_tbs = create_xir_tensor_buffers(
      subgraph->get_sorted_input_tensors(), vaip_core::XIR_INPUT);
  add_tensor_info_to_report(xir_input_tbs,
                            report_.mutable_xir_subgraph_inputs());
  auto xir_output_tbs = create_xir_tensor_buffers(
      subgraph->get_sorted_output_tensors(), vaip_core::XIR_OUTPUT);
  add_tensor_info_to_report(xir_output_tbs,
                            report_.mutable_xir_subgraph_outputs());
  if (subgraph->has_attr("fallback_to_cpu") &&
      subgraph->get_attr<bool>("fallback_to_cpu")) {
    report_.set_status("fallback_to_cpu");
    return ProcessInfo(false);
  }
  if (ENV_PARAM(XLNX_ENABLE_CHECK_DPU_SG_4D) && !check_4d_dimention(subgraph)) {
    report_.set_status("Fail_for_Only_support_4-D_tensor");
    report_.set_comments(report_comment_.str());
    return ProcessInfo(false);
  }
  if (ENV_PARAM(DEBUG_LEVEL1_DPU)) {
    LOG(INFO) << "processing subgraph: " << subgraph->get_name();
    for (auto tb : xir_input_tbs) {
      LOG(INFO) << "xir input tb : " << tb.DebugString();
    }
    for (auto tb : xir_output_tbs) {
      LOG(INFO) << "xir output tb : " << tb.DebugString();
    }
  }

  auto xir_input_anchor_points = create_anchor_points_by_xir_tensors(
      self_, onnx_graph_, subgraph->get_sorted_input_tensors(),
      vaip_core::XIR_INPUT, subgraph);
  add_anchor_point_to_report(xir_input_anchor_points,
                             report_.mutable_xir_input_anchor_points());
  if (any_of_is_nullptr(xir_input_anchor_points)) {
    MY_LOG(1) << "cancel subgraph processing, at least there is one of xir "
                 "input tensor cannot be anchored to the original graph.";
    report_.set_status("xir_input_anchor_point_is_null");
    report_.set_comments(report_comment_.str());
    return ProcessInfo(false);
  }
  auto xir_output_anchor_points = create_anchor_points_by_xir_tensors(
      self_, onnx_graph_, subgraph->get_sorted_output_tensors(),
      vaip_core::XIR_OUTPUT, subgraph);
  add_anchor_point_to_report(xir_output_anchor_points,
                             report_.mutable_xir_output_anchor_points());
  if (any_of_is_nullptr(xir_output_anchor_points)) {
    MY_LOG(1) << "cancel subgraph processing, at least there is one of xir "
                 "output tensor cannot be anchored to the original graph.";
    report_.set_status("xir_output_anchor_point_is_null");
    report_.set_comments(report_comment_.str());
    return ProcessInfo(false);
  }
  if (ENV_PARAM(DEBUG_LEVEL1_DPU)) {
    for (auto& ap : xir_input_anchor_points) {
      MY_LOG(1) << "xir input anchor point:" << ap->op_debug_string() << "\n"
                << ap->get_proto().DebugString();
    }
    for (auto& ap : xir_output_anchor_points) {
      MY_LOG(1) << "xir output anchor point " << ap->op_debug_string() << "\n"
                << ap->get_proto().DebugString();
    }
  }
  auto inputs = get_origin_names_by_anchor_points(xir_input_anchor_points);
  auto outputs = get_origin_names_by_anchor_points(xir_output_anchor_points);

  return ProcessInfo(inputs, outputs, xir_input_tbs, xir_input_anchor_points,
                     xir_output_tbs, xir_output_anchor_points);
}

std::unique_ptr<MetaDefProto>
DPUSubgraphProcessor::process_internal(const xir::Subgraph* subgraph,
                                       const ProcessInfo& above_context) {

  auto inputs = above_context.inputs;
  auto outputs = above_context.outputs;
  auto xir_input_tbs = above_context.xir_input_tbs;
  auto xir_output_tbs = above_context.xir_output_tbs;
  report_.mutable_try_fuse()->mutable_inputs()->Add(inputs.begin(),
                                                    inputs.end());
  report_.mutable_try_fuse()->mutable_outputs()->Add(outputs.begin(),
                                                     outputs.end());
  LOG_IF(INFO, ENV_PARAM(DEBUG_LEVEL1_DPU))
      << "try fuse subgraph ( " << subgraph->get_name()
      << " ) : inputs : " << inputs << " outputs " << outputs;
  auto [meta_def, fuse_error] = self_.try_fuse(
      onnx_graph_, subgraph->get_name(), inputs, outputs, {}, "DPU");
  if (meta_def == nullptr) {
    for (size_t i = 0; i < fuse_error.body.size(); i++) {
      auto node = fuse_error.body[i];
      report_.mutable_try_fuse()->mutable_body()->Add(
          std::string(node_get_first_output_name(*node)));
    }
    for (auto argument : fuse_error.arguments) {
      report_.mutable_try_fuse()->add_arguments(argument);
    }
    for (auto return_value : fuse_error.return_values) {
      report_.mutable_try_fuse()->add_return_values(return_value);
    }
  } else {
    report_.mutable_try_fuse()->mutable_body()->Assign(
        meta_def->nodes().begin(), meta_def->nodes().end());
    report_.mutable_try_fuse()->mutable_arguments()->CopyFrom(
        meta_def->inputs());
    report_.mutable_try_fuse()->mutable_return_values()->CopyFrom(
        meta_def->outputs());
  }
  if (meta_def == nullptr) {
    MY_LOG(1) << "cancel subgraph processing , subgraph ("
              << subgraph->get_name() << ") try fuse failed";
    report_.set_status("try_fuse_failed");
    report_.mutable_try_fuse_loop_path()->Assign(fuse_error.path.begin(),
                                                 fuse_error.path.end());
    report_.set_comments(fuse_error.comments);
    return nullptr;
  }
  bool is_xir_sg_nested = check_xir_sg_nested(onnx_graph_, meta_def);
  if (is_xir_sg_nested) {
    MY_LOG(1) << "cancel subgraph processing , subgraph ("
              << subgraph->get_name() << ") nested another subgraph";
    report_.set_status("subgraph nested");
    report_.mutable_try_fuse_loop_path()->Assign(fuse_error.path.begin(),
                                                 fuse_error.path.end());
    report_.set_comments(fuse_error.comments);
    return nullptr;
  }
  LOG_IF(INFO, ENV_PARAM(DEBUG_LEVEL1_DPU))
      << "try fuse subgrah return : arguments :" << meta_def->inputs()
      << " return values : " << meta_def->outputs();
  LOG_IF(INFO, ENV_PARAM(DEBUG_LEVEL1_DPU))
      << "try fuse subgrah return nodes :" << meta_def->nodes();
  LOG_IF(INFO, ENV_PARAM(DEBUG_LEVEL1_DPU))
      << "try fuse subgrah return constant_initializers  :"
      << meta_def->constant_initializers();

  if (ENV_PARAM(DISABLE_MATMUL_DPU_SG)) {
    for (int i = 0; i < meta_def->nodes_size(); i++) {
      auto onnx_node_arg_name = meta_def->nodes(i);
      auto pos = onnx_node_arg_name.find("MatMul");
      if (pos != std::string::npos) {
        MY_LOG(1) << "cancel subgraph processing , subgraph ("
                  << subgraph->get_name() << ") contain MatMul name";
        return nullptr;
      }
    }
  }

  if (ENV_PARAM(DISABLE_RESIZE_DPU_SG)) {
    for (int i = 0; i < meta_def->nodes_size(); i++) {
      auto onnx_node_arg_name = meta_def->nodes(i);
      auto pos = onnx_node_arg_name.find("Resize");
      if (pos != std::string::npos) {
        MY_LOG(1) << "cancel subgraph processing , subgraph ("
                  << subgraph->get_name() << ") contain Resize name";
        return nullptr;
      }
    }
  }

  if (ENV_PARAM(DISABLE_CONCAT_DPU_SG)) {
    for (int i = 0; i < meta_def->nodes_size(); i++) {
      auto onnx_node_arg_name = meta_def->nodes(i);
      auto pos = onnx_node_arg_name.find("Concat");
      if (pos != std::string::npos) {
        MY_LOG(1) << "cancel subgraph processing , subgraph ("
                  << subgraph->get_name() << ") contain Concat name";
        return nullptr;
      }
    }
  }

  if (ENV_PARAM(DISABLE_GEMM_DPU_SG)) {
    for (int i = 0; i < meta_def->nodes_size(); i++) {
      auto onnx_node_arg_name = meta_def->nodes(i);
      auto pos = onnx_node_arg_name.find("Gemm");
      if (pos != std::string::npos) {
        MY_LOG(1) << "cancel subgraph processing , subgraph ("
                  << subgraph->get_name() << ") contain Gemm name";
        return nullptr;
      }
    }
  }

  auto onnx_input_tbs = create_onnx_tensor_buffers(
      onnx_graph_, meta_def->inputs(), vaip_core::ONNX_INPUT);
  auto onnx_output_tbs = create_onnx_tensor_buffers(
      onnx_graph_, meta_def->outputs(), vaip_core::ONNX_OUTPUT);

  if (ENV_PARAM(DEBUG_LEVEL1_DPU)) {
    for (auto tb : onnx_input_tbs) {
      LOG(INFO) << "onnx input tb : " << tb.DebugString();
    }
    for (auto tb : onnx_output_tbs) {
      LOG(INFO) << "onnx output tb : " << tb.DebugString();
    }
  }
  auto input_schedules =
      create_input_meta_schedules(self_, onnx_input_tbs, xir_input_tbs,
                                  *above_context.xir_input_anchor_points);

  for (auto i = 0u; i < xir_input_tbs.size(); ++i) {
    LOG_IF(INFO, ENV_PARAM(DEBUG_LEVEL1_DPU))
        << "input anchor point : "
        << (*above_context.xir_input_anchor_points)[i]
               ->get_proto()
               .DebugString()
        << "\n input schedule : " << input_schedules[i].DebugString();
  }

  auto onnx_output_anchor_points = fill_in_onnx_output_tb_anchor_point(
      self_, onnx_graph_, xir_output_tbs, onnx_input_tbs, onnx_output_tbs,
      *above_context.xir_output_anchor_points);
  add_anchor_point_to_report(onnx_output_anchor_points,
                             report_.mutable_onnx_output_anchor_points());
  if (any_of_is_nullptr(onnx_output_anchor_points)) {
    MY_LOG(1) << "cancel subgraph processing, at least there is one of onnx "
                 "output tensor cannot be anchored to the original graph.";
    report_.set_status("onnx_output_anchor_point_is_null");
    report_.set_comments(report_comment_.str());
    return nullptr;
  }
  auto onnx_output_schedules = create_onnx_output_meta_schedules(
      self_, xir_output_tbs, onnx_input_tbs, onnx_output_tbs,
      onnx_output_anchor_points);

  for (auto i = 0u; i < onnx_output_tbs.size(); ++i) {
    LOG_IF(INFO, ENV_PARAM(DEBUG_LEVEL1_DPU))
        << "output anchor point : "
        << onnx_output_anchor_points[i]->get_proto().DebugString()
        << "\n output schedule : " << onnx_output_schedules[i].DebugString();
  }

  auto dpu_param = meta_def->mutable_dpu_param();
  // here is always get xclbin in pass_dpu_param
  // here xclbin is fullpath for now , it is a contract between dpu pass and GE.
  // maybe GE does not care.

  auto xclbin = self_.get_context()
                    ->xclbin_path_to_cache_files(
                        self_.get_pass_proto().pass_dpu_param().xclbin())
                    .string();
  CHECK(!xclbin.empty()) << "please setting xclbin in PassDpuParam";
  dpu_param->set_xclbin(xclbin);

  std::string subfix =
      ENV_PARAM(VAIP_COMPILE_RESERVE_CONST_DATA) == 1 ? "_fat" : "";

  auto xcompiler_fingerprint = get_xcompiler_fingerprint(
      *self_.get_context(), self_.get_pass_proto().pass_dpu_param());
  dpu_param->set_compiled_xmodel(std::string("compiled.") +
                                 xcompiler_fingerprint + subfix + ".xmodel");
  dpu_param->set_subgraph_name(subgraph->get_name());
  for (auto& schedule : input_schedules) {
    dpu_param->add_input_schedule()->MergeFrom(schedule);
  }
  for (auto& schedule : onnx_output_schedules) {
    dpu_param->add_output_schedule()->MergeFrom(schedule);
  }
  report_.set_status("OK");
  return std::move(meta_def);
}
} // namespace vaip_level1_dpu
