/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

#include <algorithm>
#include <cstdint>
#include <glog/logging.h>

#include "./export_to_xir.hpp"

#include "vitis/ai/env_config.hpp"
#include <memory>
#include <sstream>
#include <vaip/my_ort.h>

#include <xir/op/op_def.hpp>

using namespace ::onnxruntime;
///
#include "./xir_hack.inc"

DEF_ENV_PARAM(DEBUG_EXPORT_TO_XIR, "0")

using namespace ::xir;
namespace vaip_core {
static void build_op(IPass& pass, const onnxruntime::Graph& graph,
                     xir::Graph* xir_graph, const Node& node);
static std::vector<int> node_arg_get_shape(const NodeArg& node_arg) {
  auto shape = node_arg_get_shape_i64(node_arg);
  CHECK(shape != nullptr) << node_arg_as_string(node_arg) << " shape absent";
  auto shape_vector = std::vector<int>();
  shape_vector.reserve(shape->size());
  for (auto i : *shape) {
    shape_vector.push_back((int)i);
  }
  return shape_vector;
}

static xir::DataType onnx_data_type_to_xir_data_type(int type) {
  auto ret = xir::create_data_type<int>();
  switch (type) {
  case onnx::TensorProto_DataType_BOOL:
    // test case: /home/public/bevdet/LS_int.onnx
    ret = xir::DataType(xir::DataType::INT, 32);
    break;
  case onnx::TensorProto_DataType_FLOAT:
    ret = xir::DataType(xir::DataType::FLOAT, 32);
    break;
  case onnx::TensorProto_DataType_INT8:
    ret = xir::DataType(xir::DataType::INT, 8);
    break;
  case onnx::TensorProto_DataType_INT16:
    ret = xir::DataType(xir::DataType::INT, 16);
    break;
  case onnx::TensorProto_DataType_INT32:
    ret = xir::DataType(xir::DataType::INT, 32);
    break;
  case onnx::TensorProto_DataType_UINT16:
    ret = xir::DataType(xir::DataType::UINT, 16);
    break;
  case onnx::TensorProto_DataType_INT64:
    ret = xir::DataType(xir::DataType::INT, 64);
    break;
  case onnx::TensorProto_DataType_UINT8:
    ret = xir::DataType(xir::DataType::UINT, 8);
    break;
  case onnx::TensorProto_DataType_FLOAT16:
    // It seems that FP16 is float16, and BFLOAT16 is bf16. but xir don't
    // support it or don't care now. test case: adobe fp16 model.
    ret = xir::DataType(xir::DataType::FLOAT, 16);
    break;
  case onnx::TensorProto_DataType_BFLOAT16:
    // It seems that FP16 is float16, and BFLOAT16 is bf16. but xir don't
    // support it or don't care now. test case:
    // autodpu_bf16/vovnet_bfloat16/1_23_512_1920_hsigmoid_False_False
    // /quantize_result/VoVNet_int.onnx(issue #1020).
    ret = xir::DataType(xir::DataType::BFLOAT, 16);
    break;
  case onnx::TensorProto_DataType_INT4:
    // test case: PSU
    ret = xir::DataType(xir::DataType::INT, 4);
    break;
  case onnx::TensorProto_DataType_UINT4:
    ret = xir::DataType(xir::DataType::UINT, 4);
    break;
  case -1:
    // test case: mlperf-retinanet
    ret = xir::DataType(xir::DataType::UNKNOWN, 32);
    break;
  default:
    // TODO
    LOG(FATAL) << "datatype is not supported: " << type;
  }
  return ret;
}
static xir::DataType node_arg_get_xir_dtype(const NodeArg& node_arg) {
  return onnx_data_type_to_xir_data_type(
      VAIP_ORT_API(node_arg_get_element_type)(node_arg));
}

static void build_type(const NodeArg& node_arg, xir::Attrs* attrs,
                       const std::string& domain) {
  auto shape = std::vector<int>();
  // test case: model #34 Pruned PSMNet
  if (node_arg_is_unknown_shape(node_arg) && domain != "com.xilinx") {
    shape = {1}; // support unknown shape if domain != "com.xilinx"
    attrs->set_attr<int>("is_unknown_shape_and_not_xilinx_domain", 1);
  } else {
    attrs->set_attr<int>("is_unknown_shape_and_not_xilinx_domain", 0);
    shape = node_arg_get_shape(node_arg);
    auto is_scalar = shape.empty();
    if (is_scalar) {
      shape = {1}; // support scalar for onnx::Ops
    }
    auto is_dynamic_size =
        (!is_scalar) && std::any_of(shape.begin(), shape.end(),
                                    [](int64_t v) { return v <= 0; });
    if (is_scalar) {
      attrs->set_attr<int>("is_scalar", 1);
    } else {
      attrs->set_attr<int>("is_scalar", 0);
    }
    if (is_dynamic_size) {
      attrs->set_attr<int>("is_dynamic_size", 1);
      attrs->set_attr<std::vector<int>>("origin_shape", shape);
    } else {
      attrs->set_attr<int>("is_dynamic_size", 0);
      attrs->set_attr<std::vector<int>>("origin_shape", shape);
    }
    for (auto& s : shape) {
      // test case: /home/public/bevdet/LS_int.onnx
      if (s <= 0) {
        s = 1;
      }
    }
  }
  attrs->set_attr<std::vector<int>>("shape", shape);
  attrs->set_attr<std::string>("data_type", // TODO: is is convertible.
                               node_arg_get_xir_dtype(node_arg).to_string());
}

void build_data_op(xir::Graph* xir_graph, const NodeArg& node_arg) {
  std::string type = "data";
  std::string domain = "com.xilinx";
  auto attrs = xir::Attrs::create();
  build_type(node_arg, attrs.get(), domain);
  auto input_ops_map = std::map<std::string, std::vector<xir::Op*>>{};
  xir::Subgraph* subgraph = nullptr;
  auto name = node_arg_get_name(node_arg);
  LOG_IF(INFO, ENV_PARAM(DEBUG_EXPORT_TO_XIR) >= 1)
      << "add data xir op:" << name;
  auto op =
      xir_graph->add_op(name, type, std::move(attrs), input_ops_map, subgraph);
  CHECK(op != nullptr);
}

static void build_input_data(xir::Graph* xir_graph,
                             const onnxruntime::Graph& graph) {
  auto inputs = graph_get_inputs(graph);
  for (auto input : inputs) {
    build_data_op(xir_graph, *input);
  }
}

static std::vector<int>
ints_as_vector_int(const ONNX_NAMESPACE::AttributeProto& attr_proto) {
  auto ret = std::vector<int>();
  auto ints = VAIP_ORT_API(attr_proto_get_ints)(attr_proto);
  ret.reserve(ints.size());
  for (auto i : ints) {
    i = std::clamp(i, (int64_t)std::numeric_limits<int32_t>::min(),
                   (int64_t)std::numeric_limits<int32_t>::max());
    ret.push_back((int)i);
  }
  return ret;
}
static void build_constant(xir::Graph* xir_graph,
                           const onnxruntime::Graph& graph) {
  auto constants = VAIP_ORT_API(graph_get_all_initialized_tensors)(graph);
  int counter = 0;
  for (auto constant : constants) {
    auto name = constant.first;
    if (xir_graph->get_op(name)) {
      // for some reasons, onnx subgraph and parent graph might have same
      // constant intializers.
      continue;
    }
    std::string type = "const";
    auto attrs = xir::Attrs::create();
    auto tensor_proto_ptr = constant.second;
    if (tensor_proto_ptr == nullptr) {
      continue;
    }
    auto& tensor_proto = *tensor_proto_ptr;
    auto tensor_proto_shape = tensor_proto_get_shape(tensor_proto);
    auto shape = std::vector<int>();
    for (auto s : tensor_proto_shape) {
      shape.push_back((int)s);
    }
    if (shape.empty()) {
      // xir does not support scalar value. use [1] to mimic scalar
      shape.push_back(1);
    }
    // for some unknown reason, some shape are zero.
    for (auto& s : shape) {
      if (s <= 0) { // resize op, roi might be zero shape , it is strange.
        s = 1;
      }
    }
    attrs->set_attr<std::vector<int>>("shape", shape);
    attrs->set_attr<std::string>(
        "data_type", onnx_data_type_to_xir_data_type(
                         VAIP_ORT_API(tensor_proto_data_type)(tensor_proto))
                         .to_string());
    auto raw_values = tensor_proto_as_raw(graph, tensor_proto);
    attrs->set_attr<std::vector<char>>(
        "data", std::vector<char>(raw_values.begin(), raw_values.end()));
    auto input_ops_map = std::map<std::string, std::vector<xir::Op*>>{};
    xir::Subgraph* subgraph = nullptr;
    LOG_IF(INFO, ENV_PARAM(DEBUG_EXPORT_TO_XIR) >= 1)
        << "add const xir op:" << name;
    auto op = xir_graph->add_op(name, type, std::move(attrs), input_ops_map,
                                subgraph);
    CHECK(op != nullptr);
    counter++;
  }
  LOG_IF(INFO, ENV_PARAM(DEBUG_EXPORT_TO_XIR) >= 1)
      << "there are " << counter << " const ops";
}

static void add_op_attr(const onnxruntime::Graph& graph,
                        const xir::OpDef& op_def,
                        const AttributeProto& attr_proto,
                        xir::Attrs* xir_attrs) {
  auto& name = VAIP_ORT_API(attr_proto_get_name)(attr_proto);
  switch (VAIP_ORT_API(attr_proto_get_type)(attr_proto)) {
  case onnx::AttributeProto_AttributeType_FLOAT:
    xir_attrs->set_attr<float>(name,
                               VAIP_ORT_API(attr_proto_get_float)(attr_proto));
    break;
  case onnx::AttributeProto_AttributeType_INT: {
    auto modified = false;
    auto i_val = VAIP_ORT_API(attr_proto_get_int)(attr_proto);
    for (auto& attr_def : op_def.attrs()) {
      if (attr_def.name == name) {
        if (attr_def.data_type == std::type_index(typeid(bool))) {
          xir_attrs->set_attr<bool>(name, i_val != 0);
          modified = true;
        } else if (attr_def.data_type == std::type_index(typeid(int32_t))) {
          i_val =
              std::clamp(i_val, (int64_t)std::numeric_limits<int32_t>::min(),
                         (int64_t)std::numeric_limits<int32_t>::max());
          xir_attrs->set_attr<int32_t>(name, (int32_t)i_val);
          modified = true;
        } else if (attr_def.data_type == std::type_index(typeid(float))) {
          xir_attrs->set_attr<float>(name, (float)i_val);
          modified = true;
        } else if (attr_def.data_type == std::type_index(typeid(int64_t))) {
          xir_attrs->set_attr<int64_t>(name, (int64_t)i_val);
          modified = true;
        } else {
          LOG(FATAL) << "Unsupprted type, name is : " << name;
        }
        break;
      }
    }
    if (!modified) {
      xir_attrs->set_attr<int32_t>(name, (int32_t)i_val);
    }
    break;
  }
  case onnx::AttributeProto_AttributeType_STRING: {
    xir_attrs->set_attr<std::string>(
        name, VAIP_ORT_API(attr_proto_get_string)(attr_proto));
    break;
  }
  case onnx::AttributeProto_AttributeType_TENSOR: {
    auto& tensor_proto = VAIP_ORT_API(attr_proto_get_tensor)(attr_proto);
    auto raw = tensor_proto_as_raw(graph, tensor_proto);
    auto data_type = VAIP_ORT_API(tensor_proto_data_type)(tensor_proto);
    // if ((attr_proto.t().data_type() == onnx::TensorProto::FLOAT ||
    //      attr_proto.t().data_type() == onnx::TensorProto::INT8 ||
    //      attr_proto.t().data_type() == onnx::TensorProto::INT64 ||
    //      attr_proto.t().data_type() == onnx::TensorProto::INT32)) {
    if (data_type == 1 || data_type == 3 || data_type == 7 || data_type == 6) {
      xir_attrs->set_attr<std::vector<char>>(
          name, std::vector<char>(raw.begin(), raw.end()));
    } else {
      LOG(FATAL) << "onnx::AttributeProto::TENSOR is not supported yet. "
                 << attr_proto_as_string(attr_proto)
                 << " op: " << op_def.name();
    }
    break;
  }
  case onnx::AttributeProto_AttributeType_GRAPH: {
    // LOG(FATAL) << "onnx::AttributeProto::GRAPH is not supported yet. "
    //          << attr_proto_as_string(attr_proto);
    // we just ignore "super layer" here, "body" is a graph.
    break;
  }
  case onnx::AttributeProto_AttributeType_SPARSE_TENSOR: {
    LOG(FATAL) << "onnx::AttributeProto::SPARSE_TENSOR is not supported yet. "
               << attr_proto_as_string(attr_proto);
    break;
  }
  case onnx::AttributeProto_AttributeType_TYPE_PROTO: {
    LOG(FATAL) << "onnx::AttributeProto::TYPE_PROTO is not supported yet. "
               << attr_proto_as_string(attr_proto);
    break;
  }
  case onnx::AttributeProto_AttributeType_FLOATS: {
    auto floats_val = VAIP_ORT_API(attr_proto_get_floats)(attr_proto);
    xir_attrs->set_attr<std::vector<float>>(
        name, std::vector<float>(floats_val.begin(), floats_val.end()));
    break;
  }
  case onnx::AttributeProto_AttributeType_INTS: {
    xir_attrs->set_attr<std::vector<int>>(name, ints_as_vector_int(attr_proto));
    break;
  }
  case onnx::AttributeProto_AttributeType_STRINGS: {
    auto strings_val = VAIP_ORT_API(attr_proto_get_strings)(attr_proto);
    xir_attrs->set_attr<std::vector<std::string>>(
        name, std::vector<std::string>(strings_val.begin(), strings_val.end()));
    break;
  }
  case onnx::AttributeProto_AttributeType_TENSORS: {
    LOG(FATAL) << "onnx::AttributeProto::TENSORS is not supported yet. "
               << attr_proto_as_string(attr_proto);
    break;
  }
  case onnx::AttributeProto_AttributeType_GRAPHS: {
    LOG(FATAL) << "onnx::AttributeProto::GRAPHS is not supported yet. "
               << attr_proto_as_string(attr_proto);
    break;
  }
  case onnx::AttributeProto_AttributeType_SPARSE_TENSORS: {
    LOG(FATAL) << "onnx::AttributeProto::SPARSE_TENSORS is not supported yet. "
               << attr_proto_as_string(attr_proto);
    break;
  }
  case onnx::AttributeProto_AttributeType_TYPE_PROTOS: {
    LOG(FATAL) << "onnx::AttributeProto::TYPE_PROTOS is not supported yet. "
               << attr_proto_as_string(attr_proto);
    break;
  }
  default:
    LOG(FATAL) << "unsupported attr type: " << attr_proto_as_string(attr_proto);
  }
} // namespace vaip_core

static void convert_attrs(const onnxruntime::Graph& graph,
                          const xir::OpDef& op_def, const Node& node,
                          xir::Attrs* xir_attrs) {
  auto domain = VAIP_ORT_API(node_op_domain)(node);
  auto node_attrs = node_get_attributes(node);
  for (auto attr : node_attrs) {
    LOG_IF(INFO, ENV_PARAM(DEBUG_EXPORT_TO_XIR) >= 2)
        << " convert attr:" << VAIP_ORT_API(attr_proto_get_name)(*attr);
    add_op_attr(graph, op_def, *attr, xir_attrs);
  }
}

const xir::OpDef& the_extra_node_op_def() {
  static auto ret = xir::OpDef("extra").add_attr(
      xir::AttrDefBuilder<std::vector<std::string>>::build(
          "subgraph_path",
          AttrDef::
              REQUIRED /* it does not matter whether it is required or not */,
          0, ""));
  return ret;
}

static void
convert_extra_attrs(const onnxruntime::Graph& graph,
                    const std::vector<AttributeProtoPtr>& extra_node_attrs,
                    xir::Attrs* xir_attrs) {
  const xir::OpDef& extra_node_op_def = the_extra_node_op_def();
  for (auto& attr : extra_node_attrs) {
    add_op_attr(graph, extra_node_op_def, *attr, xir_attrs);
  }
}

void build_fused_op(IPass& pass, const onnxruntime::Graph& origin_graph,
                    xir::Graph* xir_graph, const Node& node) {
  LOG_IF(INFO, ENV_PARAM(DEBUG_EXPORT_TO_XIR) >= 1)
      << "build fused op: node=" << node_as_string(node);
  auto& graph = VAIP_ORT_API(node_get_function_body)(node);
  LOG_IF(INFO, ENV_PARAM(DEBUG_EXPORT_TO_XIR) >= 1)
      << "graph=" << vaip_core::graph_as_string(graph);
  build_constant(xir_graph, graph);
  // GraphViewer graph_viewer(graph);
  for (auto node_idx : graph_get_node_in_topoligical_order(graph)) {
    auto node1 = VAIP_ORT_API(graph_get_node)(graph, node_idx);
    build_op(pass, origin_graph, xir_graph, *node1);
  }
}

static std::unordered_map<
    std::string,
    std::function<void(const onnxruntime::Graph&, const xir::OpDef&,
                       const Node&, xir::Attrs*)>>
    attrs_translation_table = {};

static bool op_has_same_shape(const xir::Op* op, const Node& node) {
  auto op_shape = op->get_output_tensor()->get_shape();
  auto onnx_shape = node_get_output_shape(node, 0);
  auto ok = op_shape.size() == onnx_shape.size();
  auto size = op_shape.size();
  for (auto i = 0u; i < size && ok; ++i) {
    if (onnx_shape[i] == -1)
      break; // -1 means shape need to be derived
    ok = ok && op_shape[i] == onnx_shape[i];
  }
  return ok;
}

static std::string xir_op_as_string(const xir::Op* op) {
  std::ostringstream str;
  str << "{";
  str << "type=" << op->get_type();
  str << ",name=" << op->get_name();
  str << ",shape=" << container_as_string(op->get_output_tensor()->get_shape());
  str << "}";
  return str.str();
}
static std::vector<NodeArg*>
const_cast_vector_node_args(std::vector<const NodeArg*> node_args) {
  auto ret = std::vector<NodeArg*>();
  ret.reserve(node_args.size());
  for (auto arg : node_args) {
    ret.push_back(const_cast<NodeArg*>(arg));
  }
  return ret;
}

static void build_selector_op(xir::Graph* xir_graph, const xir::Op* op,
                              const NodeArg& node_arg, size_t index) {
  auto attrs = xir::Attrs::create();
  auto type = std::string("Selector");
  std::string domain = "";
  auto input_ops_map = std::map<std::string, std::vector<xir::Op*>>{
      {"input", {const_cast<xir::Op*>(op)}}};
  attrs->set_attr<int32_t>("index", (int32_t)index);
  build_type(node_arg, attrs.get(), domain);
  xir::Subgraph* subgraph = nullptr;
  auto op_name = node_arg_get_name(node_arg);
  xir_graph->add_op(op_name, type, std::move(attrs), input_ops_map, subgraph);
}

static void build_op(IPass& pass, const onnxruntime::Graph& graph,
                     xir::Graph* xir_graph, const Node& node) {
  std::string type = convert_to_xir_op_type(VAIP_ORT_API(node_op_domain)(node),
                                            VAIP_ORT_API(node_op_type)(node));

  auto& op_def = *op_def_factory()->get_op_def(type);
  if (VAIP_ORT_API(node_type_is_fused)(node)) {
    build_fused_op(pass, graph, xir_graph, node);
  } else {
    // test case: /home/public/bevdet/LS_int.onnx
    // op : Split
    auto node_args = node_get_output_node_args(node);
    LOG_IF(INFO, ENV_PARAM(DEBUG_EXPORT_TO_XIR) >= 1)
        << "add xir op: node=" << node_as_string(node);
    std::string op_name;
    if (node_args.size() == 1) {
      op_name = node_arg_get_name(*node_args[0]);
    } else {
      for (auto arg : node_args) {
        op_name = op_name + ":" + node_arg_get_name(*arg);
      }
    }
    auto attrs = xir::Attrs::create();
    auto input_node_args = node_get_input_node_args(node);
    auto input_ops_map = get_ops_map(
        xir_graph, op_def, const_cast_vector_node_args(input_node_args));
    xir::Subgraph* subgraph = nullptr;
    auto convert_attrs_func =
        std::function<void(const onnxruntime::Graph&, const xir::OpDef&,
                           const Node&, xir::Attrs*)>();
    convert_attrs_func = convert_attrs;

    auto it = attrs_translation_table.find(type);
    if (it != attrs_translation_table.end()) {
      convert_attrs_func = it->second;
    }

    convert_attrs_func(graph, op_def, node, attrs.get());

    convert_extra_attrs(graph, pass.node_extra_attrs(op_name.c_str()),
                        attrs.get());
    auto domain = VAIP_ORT_API(node_op_domain)(node);
    if (node_args.size() == 1) {
      build_type(*node_args[0], attrs.get(), domain);
    } else {
      // for multiply outputs OP, the shape and data_type does not matter any
      // more, because Selector op accepts any size of inputs, so we just copy
      // the first output's shape and datatype.
      build_type(*node_args[0], attrs.get(), domain);
    }
    if (ENV_PARAM(DEBUG_EXPORT_TO_XIR) >= 1) {
      LOG(INFO) << " xir op type=" << type;
      for (auto& i : input_ops_map) {
        LOG(INFO) << "\t" << i.first << "=";
        for (auto& j : i.second) {
          LOG(INFO) << "\t\t" << j->get_name() << " :: "
                    << container_as_string(j->get_output_tensor()->get_shape());
        }
      }
    }
    if (type == "const") {
      CHECK(pass.has_const(op_name.c_str()))
          << "cannot find constant data: " << op_name;
      auto data = pass.get_const_data<char>(op_name.c_str());
      attrs->set_attr<std::vector<char>>(
          "data", std::vector<char>(data.begin(), data.end()));
    };
    auto is_scalar =
        attrs->has_attr("is_scalar") && attrs->get_attr<int>("is_scalar");
    auto is_dynamic_size = attrs->has_attr("is_dynamic_size") &&
                           attrs->get_attr<int>("is_dynamic_size");
    auto is_unknown_shape_and_not_xilinx_domain =
        attrs->has_attr("is_unknown_shape_and_not_xilinx_domain") &&
        attrs->get_attr<int>("is_unknown_shape_and_not_xilinx_domain");
    auto op = xir_graph->add_op(op_name, type, std::move(attrs), input_ops_map,
                                subgraph);
    if (node_args.size() > 1) {
      size_t index = 0;
      for (auto arg : node_args) {
        build_selector_op(xir_graph, op, *arg, index);
        index = index + 1;
      }
    }
    if (is_scalar || is_dynamic_size ||
        is_unknown_shape_and_not_xilinx_domain) {
      // skip shape checking for scalar or dynamic size or
      // unknown_shape_and_not_xilinx_domain ops.
    } else {
      CHECK(op_has_same_shape(op, node))
          << " node=" << node_as_string(node)
          << " xir_op=" << xir_op_as_string(op) << "\n\t"
          << "Ususually, it is because that we do not handle layout tranform "
             "properly for Op 'xilinx:"
          << VAIP_ORT_API(node_op_type)(node)
          << "' please check layout_transform_pass.cpp";
    }
    if (ENV_PARAM(DEBUG_EXPORT_TO_XIR) >= 1) {
      LOG(INFO) << " xir op shape="
                << container_as_string(op->get_output_tensor()->get_shape());
    }
    CHECK(op != nullptr);
  }
} // namespace vaip_core

// static void remove_undefined_attrs(const xir::Op* op) {
//   auto attrs = op->get_attrs();
//   auto opdef = op->get_opdef();
//   for (auto& key : attrs->get_keys()) {
//     auto& opdef_attrs = opdef->attrs();
//     auto found = std::find_if(
//         opdef_attrs.begin(), opdef_attrs.end(),
//         [&key](const xir::AttrDef& def) -> bool { return def.name == key; });
//     if (found == opdef_attrs.end()) {
//       attrs->del_attr(key);
//     }
//   }
//   const_cast<xir::Op*>(op)->set_attrs(std::move(attrs));
// }

std::unique_ptr<xir::Graph> export_to_xir(IPass& pass,
                                          onnxruntime::Graph& graph) {
  graph_resolve(graph, true);
  auto nodes = graph_nodes(graph);
  auto xir_graph = xir::Graph::create(VAIP_ORT_API(graph_get_name)(graph));
  auto xir_graph_p = xir_graph.get();
  build_input_data(xir_graph.get(), graph);
  build_constant(xir_graph.get(), graph);
  auto graph_outputs = graph_get_outputs(graph);
  std::vector<const Node*> leaf_nodes;
  leaf_nodes.reserve(graph_outputs.size());
  for (auto n : graph_nodes(graph)) {
    CHECK(n != nullptr);
    auto node_outputs = node_get_output_node_args(*n);
    auto found = std::any_of(node_outputs.begin(), node_outputs.end(),
                             [&graph_outputs](const NodeArg* x) {
                               return std::find(graph_outputs.begin(),
                                                graph_outputs.end(),
                                                x) != graph_outputs.end();
                             });
    if (found) {
      leaf_nodes.push_back(n);
    }
  }
  VAIP_ORT_API(graph_reverse_dfs_from)
  (
      graph,      //
      leaf_nodes, //
      [](const Node* n) {
        LOG_IF(INFO, false) << "\n\tnode leave: " << node_as_string(*n);
      }, //
      [&graph, &pass, xir_graph_p](const Node* n) {
        build_op(pass, graph, xir_graph_p, *n);
      }, //
      nullptr);
  return xir_graph;
}
} // namespace vaip_core
