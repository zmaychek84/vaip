/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

// clang-format off
#include "prepare_metadata.hpp"
#include <glog/logging.h>
#include <vaip/vaip_ort_api.h>
#include <numeric>
#include <filesystem>
#include <fstream>
#include "vaip/dd_metadata.pb.h"
#include <google/protobuf/util/json_util.h>
#include "vaip/dd/dd_utils.hpp"
// clang-format on

namespace dd {

using namespace vaip_core;

// static int get_element_size(int elem_type) {
//   auto size_of_value_type = sizeof(int8_t);
//   switch (elem_type) {
//   case ONNX_NAMESPACE::TensorProto_DataType_INT8:
//   case ONNX_NAMESPACE::TensorProto_DataType_UINT8: {
//     size_of_value_type = sizeof(int8_t);
//     break;
//   }
//   case ONNX_NAMESPACE::TensorProto_DataType_INT64: {
//     size_of_value_type = sizeof(int64_t);
//     break;
//   }
//   case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
//   case ONNX_NAMESPACE::TensorProto_DataType_INT32: {
//     size_of_value_type = sizeof(float);
//     break;
//   }
//   case ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16:
//   case ONNX_NAMESPACE::TensorProto_DataType_UINT16:
//   case ONNX_NAMESPACE::TensorProto_DataType_INT16: {
//     size_of_value_type = 2;
//     break;
//   }
//   default:
//     LOG(FATAL) << "unsupported element type: " << elem_type;
//   }
//   return static_cast<int>(size_of_value_type);
// }

static size_t get_size_of_type(const std::string& type) {
  static const std::unordered_map<std::string, size_t> elem_size{
      {"int4", 1},  {"uint4", 1},  {"int8", 1},     {"uint8", 1},
      {"int16", 2}, {"uint16", 2}, {"int32", 4},    {"uint32", 4},
      {"int64", 8}, {"uint64", 8}, {"bfloat16", 2}, {"float32", 4},
      {"float", 4}, {"double", 8}, {"float64", 8}};
  if (elem_size.find(type) == elem_size.end()) {
    throw std::runtime_error("get_size_of_type - Invalid type : " + type);
  }
  auto sz = elem_size.at(type);
  return sz;
}

// static std::string get_element_type_str(int elem_type) {
//   std::string str_type;
//   switch (elem_type) {
//   case ONNX_NAMESPACE::TensorProto_DataType_INT8:
//     str_type = "int8";
//     break;
//   case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
//     str_type = "uint8";
//     break;
//   case ONNX_NAMESPACE::TensorProto_DataType_INT64:
//     str_type = "int64";
//     break;
//   case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
//     str_type = "float";
//     break;
//   case ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16:
//     str_type = "bfloat16";
//     break;
//   case ONNX_NAMESPACE::TensorProto_DataType_UINT16:
//     str_type = "uint16";
//     break;
//   case ONNX_NAMESPACE::TensorProto_DataType_INT16:
//     str_type = "int16";
//     break;
//   case ONNX_NAMESPACE::TensorProto_DataType_INT32:
//     str_type = "int32";
//     break;
//   default:
//     LOG(FATAL) << "unsupported element type: " << elem_type;
//   }
//   return str_type;
// }

AuxTensorInfo get_tensor_aux_info(const NodeArg& node_arg,
                                  const std::string& node_arg_dtype,
                                  const std::vector<int64_t>& node_arg_shape) {
  AuxTensorInfo info;
  // Tensor dtype is filled based on node_attrs, not from the tensor dtype
  // auto int_dtype = node_arg_get_element_type(node_arg);
  // info.dtype = get_element_type_str(int_dtype);
  info.dtype = node_arg_dtype;

  auto arg_shape = node_arg_get_shape_i64(node_arg);
  if (!node_arg_shape.empty()) {
    info.shape = node_arg_shape;
  } else if (arg_shape) {
    info.shape = *arg_shape;
  }

  auto elem_size = get_size_of_type(node_arg_dtype);
  info.size_in_bytes = static_cast<int32_t>(
      std::accumulate(info.shape.begin(), info.shape.end(), (int64_t)1,
                      std::multiplies<int64_t>()) *
      elem_size);
  return info;
}

AuxTensorInfo get_tensor_aux_info(const TensorProto& tensor_proto,
                                  const std::string& node_arg_dtype) {
  AuxTensorInfo info;
  // Tensor dtype is filled based on node_attrs, not from the tensor dtype
  // auto int_dtype = VAIP_ORT_API(tensor_proto_data_type)(tensor_proto);
  // info.dtype = get_element_type_str(int_dtype);

  info.dtype = node_arg_dtype;
  auto shape = VAIP_ORT_API(tensor_proto_get_shape_unsafe)(tensor_proto);
  CHECK(shape.get() != nullptr)
      << "tensor_proto_get_shape_unsafe should not return null shape";
  info.shape = *shape;
  // calc size
  auto elem_size = get_size_of_type(node_arg_dtype);
  info.size_in_bytes = static_cast<int32_t>(
      std::accumulate(info.shape.begin(), info.shape.end(), (int64_t)1,
                      std::multiplies<int64_t>()) *
      elem_size);
  return info;
}

int align_to_next(int n, int A) { return ((n + A - 1) / A) * A; }

using TensorInfoVec = std::vector<std::pair<std::string, AuxTensorInfo>>;
using tensor_dtype_map_t = std::unordered_map<std::string, std::string>;
using tensor_shape_map_t =
    std::unordered_map<std::string, std::vector<int64_t>>;

std::pair<int, TensorInfoVec>
pack_tensors(const std::vector<vaip_cxx::NodeArgConstRef>& tensors, int align,
             const tensor_dtype_map_t& tensor_dtype_map,
             const tensor_shape_map_t& tensor_shape_map) {
  int buffer_size = 0;
  TensorInfoVec res;
  for (auto node_arg : tensors) {
    auto node_arg_name = node_arg.name();
    auto node_arg_dtype = tensor_dtype_map.at(node_arg_name);
    auto node_arg_shape =
        tensor_shape_map.find(node_arg_name) != tensor_shape_map.end()
            ? tensor_shape_map.at(node_arg_name)
            : std::vector<int64_t>{};

    auto tensor_info =
        get_tensor_aux_info(node_arg, node_arg_dtype, node_arg_shape);
    tensor_info.offset = buffer_size;
    res.emplace_back(node_arg_name, tensor_info);
    buffer_size += tensor_info.size_in_bytes;
    buffer_size = align_to_next(buffer_size, align);
  }
  return {buffer_size, res};
}

std::pair<int, TensorInfoVec>
pack_tensors(const std::vector<vaip_cxx::NodeArgConstRef>& initializers,
             int align, const tensor_dtype_map_t& tensor_dtype_map) {
  int buffer_size = 0;
  TensorInfoVec res;
  for (auto initializer : initializers) {
    auto node_arg_name = initializer.name();
    auto node_arg_dtype = tensor_dtype_map.at(node_arg_name);
    auto shape = initializer.shape();
    CHECK(shape != nullptr);
    auto tensor_info = get_tensor_aux_info(initializer, node_arg_dtype, *shape);
    tensor_info.offset = buffer_size;
    res.emplace_back(node_arg_name, tensor_info);
    buffer_size += tensor_info.size_in_bytes;
    buffer_size = align_to_next(buffer_size, align);
  }
  return {buffer_size, res};
}

std::set<vaip_cxx::NodeArgConstRef>
graph_get_all_node_args(const vaip_cxx::Subgraph& graph) {
  std::set<vaip_cxx::NodeArgConstRef> ret;
  auto nodes = graph.nodes();
  // may need some checks before insert
  for (auto node : nodes) {
    auto node_inputs = node.inputs();
    for (auto node_input : node_inputs) {
      CHECK(node_input.has_value());
      ret.insert(node_input.value());
    }
    auto node_outputs = node.outputs();
    for (auto node_output : node_outputs) {
      CHECK(node_output.has_value());
      ret.insert(node_output.value());
    }
  }
  return ret;
}

static bool find_name_in_vec(const TensorInfoVec& vec,
                             const std::string& name) {
  for (const auto& item : vec) {
    if (item.first == name) {
      return true;
    }
  }
  return false;
}

std::vector<vaip_cxx::NodeArgConstRef> get_intermediate_node_args(
    const std::set<vaip_cxx::NodeArgConstRef>& all_node_args,
    const TensorInfoVec& inputs, const TensorInfoVec& outputs,
    const TensorInfoVec& initializers) {
  std::vector<vaip_cxx::NodeArgConstRef> ret;
  for (auto node_arg : all_node_args) {
    auto name = node_arg.name();
    if (!find_name_in_vec(inputs, name) && !find_name_in_vec(outputs, name) &&
        !find_name_in_vec(initializers, name)) {
      ret.push_back(node_arg);
    }
  }
  return ret;
}

// names will be refactored
// name, size, map
using AllTensorVec = std::vector<std::tuple<std::string, int, TensorInfoVec>>;

std::pair<NewTensors, NewTensorInfoMap>
prepare_tensor_maps(const AllTensorVec& all_tensors) {
  // we could combine the 2 loops later
  int cnt = 0;
  NewTensors new_tensors;
  for (const auto& [name, size, info] : all_tensors) {
    NewTensorInfo new_info;
    new_info.buffer_size = size;
    new_info.xrt_arg_id = cnt;
    // collect names
    for (const auto& item : info) {
      new_info.packed_tensors.push_back(item.first);
    }
    new_tensors[name] = new_info;
    cnt++;
  }

  NewTensorInfoMap new_tensors_map;
  for (const auto& [name, size, info] : all_tensors) {
    for (const auto& item : info) {
      if (new_tensors_map.count(item.first) == 0) {
        NewTensorMapItem new_item;
        new_item.packed_buffer_label = name;
        new_item.xrt_arg_id = new_tensors[name].xrt_arg_id;
        new_item.aux_info = item.second; // unpack when serialize
        new_item.file_size = 0;
        new_tensors_map[item.first] = new_item;
      } else {
        LOG(FATAL) << "found duplicate key: " << item.first;
      }
    }
  }
  return {new_tensors, new_tensors_map};
}

using ConstInfoMap = std::map<std::string, std::pair<std::string, size_t>>;

std::string translate_tensor_name(const std::string& name) {
  const std::string src_chars = "/: ";
  std::string new_str = name;
  for (auto c : src_chars) {
    std::replace(new_str.begin(), new_str.end(), c, '_');
  }
  return new_str;
}

ConstInfoMap
write_consts(const std::vector<vaip_cxx::NodeArgConstRef>& initializers,
             const std::filesystem::path& dir_path, LeanConstDB& const_db) {
  // ConstInfoMap const_file_info;
  for (const auto& initializer : initializers) {
    auto name = initializer.name();
    auto new_name = translate_tensor_name(name);
    std::string base_name = new_name + ".const";
    std::filesystem::path filename = dir_path / base_name;
    auto abs_path = std::filesystem::absolute(filename);
    // may need to check for unsupported types
    auto raw_data = initializer.const_data_as_raw();
    // if need to save data_type, for simplicity this can be the first 4 bytes
    // currently numpy don't save this
    // std::ofstream outfile(filename, std::ofstream::binary);
    // outfile.write(raw_data.data(), raw_data.size());
    // const_file_info[name] = {base_name, raw_data.size()};
    // LOG(INFO)<< name << " " << (size_t) raw_data.data()<< " " << (size_t)
    // raw_data.size();
    const_db[name] = {const_cast<char*>(raw_data.data()), raw_data.size()};
    // std::vector<char>(raw_data.data(), raw_data.data() + raw_data.size());
  }
  return {}; // const_file_info;
}

std::string to_string_with_precision(float f, int n = 6) {
  std::ostringstream oss;
  oss.precision(n);
  oss << f;
  return oss.str();
}

static tensor_dtype_map_t
extract_tensor_dtypes(const vaip_cxx::Subgraph& graph) {
  tensor_dtype_map_t tensor_dtypes;

  auto nodes = graph.nodes();
  for (const auto& node : nodes) {
    // auto node_op = VAIP_ORT_API(node_op_type)(*node);
    // std::cout<<node_op<<std::endl;
    auto input_args = node.inputs();
    auto output_args = node.outputs();
    auto in_dtypes = node.get_attr_strings("in_dtypes");
    auto out_dtypes = node.get_attr_strings("out_dtypes");
    CHECK(input_args.size() == in_dtypes.size())
        << input_args.size() << " v/s " << in_dtypes.size();
    CHECK(output_args.size() == out_dtypes.size())
        << output_args.size() << " v/s " << out_dtypes.size();

    for (size_t i = 0; i < input_args.size(); ++i) {
      CHECK(input_args.at(i).has_value()) << "input arg is null, i=" << i;
      auto input_arg_name = input_args.at(i).value().name();
      if (tensor_dtypes.find(input_arg_name) == tensor_dtypes.end()) {
        tensor_dtypes[input_arg_name] = in_dtypes[i];
      } else {
        const auto& old_type = tensor_dtypes.at(input_arg_name);
        CHECK(old_type == in_dtypes[i])
            << "Same tensor, but different dtypes for tensor: "
            << input_arg_name << ", " << old_type << " v/s " << in_dtypes[i];
      }
    }

    for (size_t i = 0; i < output_args.size(); ++i) {
      auto output_arg_name = node_arg_get_name(*output_args.at(i));
      if (tensor_dtypes.find(output_arg_name) == tensor_dtypes.end()) {
        tensor_dtypes[output_arg_name] = out_dtypes[i];
      } else {
        const auto& old_type = tensor_dtypes.at(output_arg_name);
        CHECK(old_type == out_dtypes[i])
            << "Same tensor, but different dtypes for tensor: "
            << output_arg_name << ", " << old_type << " v/s " << out_dtypes[i];
      }
    }
  }
  return tensor_dtypes;
}

template <typename DType>
static std::vector<DType> string_to_values(std::string str_values) {
  std::stringstream ss(str_values);
  std::vector<DType> values;
  for (DType dim; ss >> dim;) {
    values.push_back(dim);
    if (ss.peek() == ' ')
      ss.ignore();
  }
  return values;
}

static std::vector<std::vector<int64_t>>
cvt_strvec_to_shapevec(const std::vector<std::string>& strvec) {
  std::vector<std::vector<int64_t>> shapes;
  for (const auto& str : strvec) {
    auto shape = string_to_values<int64_t>(str);
    shapes.push_back(std::move(shape));
  }
  return shapes;
}

static tensor_shape_map_t
extract_tensor_shapes(const vaip_cxx::Subgraph& graph) {
  tensor_shape_map_t tensor_shapes;

  auto nodes = graph.nodes();
  for (const auto& node : nodes) {
    if (!(node.has_attr("dd_op_in_shape") &&
          node.has_attr("dd_op_out_shape"))) {
      continue;
    }
    auto input_args = node.inputs();
    auto output_args = node.outputs();
    auto in_shapes =
        cvt_strvec_to_shapevec(node.get_attr_strings("dd_op_in_shape"));
    auto out_shapes =
        cvt_strvec_to_shapevec(node.get_attr_strings("dd_op_out_shape"));
    // CHECK(input_args.size() == in_shapes.size())
    //     << input_args.size() << " v/s " << in_shapes.size();
    // CHECK(output_args.size() == out_shapes.size())
    //     << output_args.size() << " v/s " << out_shapes.size();

    for (size_t i = 0; i < in_shapes.size() /*input_args.size()*/; ++i) {
      auto input_arg_name = node_arg_get_name(*input_args.at(i));
      if (tensor_shapes.find(input_arg_name) == tensor_shapes.end()) {
        tensor_shapes[input_arg_name] = in_shapes[i];
      } else {
        const auto& old_type = tensor_shapes.at(input_arg_name);
        if (vaip::dd::reduce(old_type) != vaip::dd::reduce(in_shapes[i])) {
          LOG(WARNING)
              << "Same tensor, but different shape for tensor (input): "
              << input_arg_name << ", " << vaip::dd::shape_as_string(old_type)
              << " v/s " << vaip::dd::shape_as_string(in_shapes[i]);
        } else {
          // Overwrite it anyway
          tensor_shapes[input_arg_name] = in_shapes[i];
        }
      }
    }

    for (size_t i = 0; i < out_shapes.size() /*output_args.size()*/; ++i) {
      auto output_arg_name = node_arg_get_name(*output_args.at(i));
      if (tensor_shapes.find(output_arg_name) == tensor_shapes.end()) {
        tensor_shapes[output_arg_name] = out_shapes[i];
      } else {
        const auto& old_type = tensor_shapes.at(output_arg_name);
        if (vaip::dd::reduce(old_type) != vaip::dd::reduce(out_shapes[i])) {
          LOG(WARNING)
              << "Same tensor, but different shape for tensor (output): "
              << output_arg_name << ", " << vaip::dd::shape_as_string(old_type)
              << " v/s " << vaip::dd::shape_as_string(out_shapes[i]);
        } else {
          tensor_shapes[output_arg_name] = out_shapes[i];
        }
      }
    }
  }
  return tensor_shapes;
}

std::tuple<std::vector<OPInfo>, NewTensors, NewTensorInfoMap, LeanConstDB>
graph_prepare_metadata(const vaip_cxx::Subgraph& graph,
                       const std::filesystem::path& dir_path) {
  auto tensor_dtype_map = extract_tensor_dtypes(graph);
  auto tensor_shape_map = extract_tensor_shapes(graph);

  const int TensorPackAlignment = 4;
  // input
  auto& inputs = graph.inputs();
  auto input_pair = pack_tensors(inputs, TensorPackAlignment, tensor_dtype_map,
                                 tensor_shape_map);

  // outputs
  auto& outputs = graph.outputs();
  auto output_pair = pack_tensors(outputs, TensorPackAlignment,
                                  tensor_dtype_map, tensor_shape_map);

  // initializers
  auto& initializers = graph.constant_initializers();

  auto initializers_pair =
      pack_tensors(initializers, TensorPackAlignment, tensor_dtype_map);

  // intermediate tensors
  auto all_node_args = graph_get_all_node_args(graph);
  auto scratchs =
      get_intermediate_node_args(all_node_args, input_pair.second,
                                 output_pair.second, initializers_pair.second);
  auto scratchs_pair = pack_tensors(scratchs, TensorPackAlignment,
                                    tensor_dtype_map, tensor_shape_map);

  AllTensorVec all_tensors;
  all_tensors.emplace_back("in", input_pair.first, input_pair.second);
  all_tensors.emplace_back("out", output_pair.first, output_pair.second);
  all_tensors.emplace_back("scratch", scratchs_pair.first,
                           scratchs_pair.second);
  all_tensors.emplace_back("const", initializers_pair.first,
                           initializers_pair.second);
  all_tensors.emplace_back("super_instr", int(0), TensorInfoVec{});
  auto [new_tensors, new_tensors_map] = prepare_tensor_maps(all_tensors);

  // writeConsts
  // Migration note:
  // const_file_info : old flow, contains tensor name -> filename details
  // const_db : new flow, constains tensor name -> tensor data details
  LeanConstDB const_db;
  auto const_file_info = write_consts(initializers, dir_path, const_db);
  // update map
  // for (auto& [k, v] : const_file_info) {
  //   CHECK(new_tensors_map.count(k)) << "tensor don't exist: " << k;
  //   new_tensors_map[k].file_name = v.first;
  //   new_tensors_map[k].file_size = static_cast<int32_t>(v.second);
  // }

  // get op_list
  auto& nodes = graph.nodes();
  std::vector<OPInfo> op_list;
  // auto topo_node_idx = graph_get_node_in_topoligical_order(graph);
  CHECK(nodes.size() > 0) << "get nodes in topological order failed";
  for (auto& node : nodes) {
    auto node_idx = node.index();
    OPInfo op_info;
    op_info.name = node.name();
    op_info.type = node.op_type();

    // input/output names
    auto node_inputs = node.inputs();
    for (auto node_input : node_inputs) {
      const auto& arg_name = node_input.value().name();
      if (const_db.find(arg_name) == const_db.end()) {
        op_info.in_args.push_back(arg_name);
      } else {
        op_info.const_args.push_back(arg_name);
      }
    }
    auto node_outputs = node.outputs();
    for (auto node_output : node_outputs) {
      op_info.out_args.push_back(node_arg_get_name(node_output.value()));
    }
    // attrs
    auto& attrs = node_get_attributes_ref(node);
    auto attr_keys = VAIP_ORT_API(node_attributes_get_keys)(
        const_cast<NodeAttributes&>(attrs));
    for (auto& attr_key : *attr_keys) {
      auto attr_proto = node_attributes_get(attrs, attr_key);
      switch (VAIP_ORT_API(attr_proto_get_type)(*attr_proto)) {
      case onnx::AttributeProto_AttributeType_FLOAT: {
        auto f_value = VAIP_ORT_API(attr_proto_get_float)(*attr_proto);
        op_info.attrs[attr_key] = {"float",
                                   {to_string_with_precision(f_value)}};
        break;
      }
      case onnx::AttributeProto_AttributeType_INT: {
        auto i_value = VAIP_ORT_API(attr_proto_get_int)(*attr_proto);
        op_info.attrs[attr_key] = {"int", {std::to_string(i_value)}};
        break;
      }
      case onnx::AttributeProto_AttributeType_STRING: {
        auto s_value = VAIP_ORT_API(attr_proto_get_string)(*attr_proto);
        op_info.attrs[attr_key] = {"str", {s_value}};
        break;
      }
      case onnx::AttributeProto_AttributeType_FLOATS: {
        auto floats_value = VAIP_ORT_API(attr_proto_get_floats)(*attr_proto);
        std::vector<std::string> s_vec;
        for (auto f : floats_value) {
          s_vec.push_back(to_string_with_precision(f));
        }
        op_info.attrs[attr_key] = {"float", s_vec};
        break;
      }
      case onnx::AttributeProto_AttributeType_INTS: {
        auto ints_value = VAIP_ORT_API(attr_proto_get_ints)(*attr_proto);
        std::vector<std::string> s_vec;
        for (auto i : ints_value) {
          s_vec.push_back(std::to_string(i));
        }
        op_info.attrs[attr_key] = {"int", s_vec};
        break;
      }
      case onnx::AttributeProto_AttributeType_STRINGS: {
        auto strs_value = VAIP_ORT_API(attr_proto_get_strings)(*attr_proto);
        op_info.attrs[attr_key] = {"str", strs_value};
        break;
      }
      default:
        LOG(WARNING) << "Cannot write attr to json: "
                     << attr_proto_as_string(*attr_proto);
      }
    }
    op_list.push_back(op_info);
  }
  return {op_list, new_tensors, new_tensors_map, const_db};
}

std::string save_tensors_to_json(const std::vector<OPInfo>& op_list,
                                 const NewTensors& new_tensors,
                                 const NewTensorInfoMap& new_tensors_map) {
  DDMetadataProto metadata;
  metadata.set_dd_meta_major_version(1);
  metadata.set_dd_meta_minor_version(1);
  // op_list
  for (const auto& op_info : op_list) {
    auto new_op_info = metadata.add_op_list();
    new_op_info->set_name(op_info.name);
    new_op_info->set_type(op_info.type);
    for (const auto& arg : op_info.in_args) {
      new_op_info->add_in_args(arg);
    }
    for (const auto& arg : op_info.const_args) {
      new_op_info->add_const_args(arg);
    }
    for (const auto& arg : op_info.out_args) {
      new_op_info->add_out_args(arg);
    }
    auto attrs = new_op_info->mutable_attrs();
    for (const auto& [k, val_pair] : op_info.attrs) {
      DDOPAttrProto dd_attr;
      dd_attr.set_type(val_pair.first);
      for (const auto& value : val_pair.second) {
        dd_attr.add_value(value);
      }
      (*attrs)[k] = dd_attr;
    }
  }

  // new_tensors
  for (const auto& [k, v] : new_tensors) {
    auto tensor_info = metadata.mutable_fused_tensors();
    DDFusedTensorProto fused_tensor;
    fused_tensor.set_buffer_size(v.buffer_size);
    fused_tensor.set_xrt_arg_id(v.xrt_arg_id);
    for (const auto& name : v.packed_tensors) {
      fused_tensor.add_packed_tensors(name);
    }
    (*tensor_info)[k] = fused_tensor;
  }

  // new_tensors_map
  for (const auto& [k, v] : new_tensors_map) {
    auto tensor_map = metadata.mutable_tensor_map();
    DDTensorInfoProto tensor_info;
    tensor_info.set_packed_buffer_label(v.packed_buffer_label);
    tensor_info.set_xrt_arg_id(v.xrt_arg_id);
    tensor_info.set_dtype(v.aux_info.dtype);
    for (const auto& s : v.aux_info.shape) {
      tensor_info.add_shape(static_cast<int32_t>(s));
    }
    tensor_info.set_size_in_bytes(v.aux_info.size_in_bytes);
    tensor_info.set_offset(v.aux_info.offset);
    tensor_info.set_file_name(v.file_name);
    tensor_info.set_file_size(v.file_size);
    (*tensor_map)[k] = tensor_info;
  }

  google::protobuf::util::JsonPrintOptions options;
  options.add_whitespace = true;
  options.always_print_primitive_fields = true;
  options.preserve_proto_field_names = true;
  auto json_str = std::string();
  auto status =
      google::protobuf::util::MessageToJsonString(metadata, &json_str, options);
  CHECK(status.ok()) << "cannot write json string:" << metadata.DebugString();
  return json_str;
}

} // namespace dd
