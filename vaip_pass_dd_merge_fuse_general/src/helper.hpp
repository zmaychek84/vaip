/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */
#ifdef __GNUC__
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wpedantic"
#  pragma GCC diagnostic ignored "-Wconversion"
#endif

#include "vaip/dd/coeffs.hpp"
#include "vaip/dd/dd_utils.hpp"

#ifdef __GNUC__
#  pragma GCC diagnostic pop
#endif
#include "vaip/vaip.hpp"
#include "vitis/ai/env_config.hpp"
#include <glog/logging.h>

DEF_ENV_PARAM(DEBUG_DD_PATTERN, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_DD_PATTERN) >= n)
namespace {
using namespace vaip_core;

// Splits a space-separated string into a vector of words
// Handles multiple consecutive spaces and leading/trailing spaces
static std::vector<std::string> splitString(const std::string& input) {
  std::vector<std::string> result;
  std::stringstream ss(input);
  std::string word;

  while (ss >> word) {
    result.push_back(word);
  }

  return result;
}

static void add_attributes(NodeBuilder& node_builder,
                           const AttributeProto* attr_proto,
                           std::string node_name) {
  auto attr_name = VAIP_ORT_API(attr_proto_get_name)(*attr_proto);
  auto attr_type = VAIP_ORT_API(attr_proto_get_type)(*attr_proto);

  if (attr_type == onnx::AttributeProto_AttributeType_INT) {
    auto value = VAIP_ORT_API(attr_proto_get_int)(*attr_proto);
    node_builder.add(attr_name + "_" + node_name, value);
  } else if (attr_type == onnx::AttributeProto_AttributeType_FLOAT) {
    auto value = VAIP_ORT_API(attr_proto_get_float)(*attr_proto);
    node_builder.add(attr_name + "_" + node_name, value);
  } else if (attr_type == onnx::AttributeProto_AttributeType_STRING) {
    auto value = VAIP_ORT_API(attr_proto_get_string)(*attr_proto);
    node_builder.add(attr_name + "_" + node_name, value);
  } else if (attr_type == onnx::AttributeProto_AttributeType_INTS) {
    auto value = VAIP_ORT_API(attr_proto_get_ints)(*attr_proto);
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < value.size(); ++i) {
      oss << value[i];
      if (i < value.size() - 1) {
        oss << ", ";
      }
    }
    oss << "]";
    node_builder.add(attr_name + "_" + node_name, oss.str());
  } else if (attr_type == onnx::AttributeProto_AttributeType_FLOATS) {
    auto value = VAIP_ORT_API(attr_proto_get_floats)(*attr_proto);
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < value.size(); ++i) {
      oss << value[i];
      if (i < value.size() - 1) {
        oss << ", ";
      }
    }
    oss << "]";
    node_builder.add(attr_name + "_" + node_name, oss.str());
  } else if (attr_type == onnx::AttributeProto_AttributeType_STRINGS) {
    auto value = VAIP_ORT_API(attr_proto_get_strings)(*attr_proto);
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < value.size(); ++i) {
      oss << value[i];
      if (i < value.size() - 1) {
        oss << ", ";
      }
    }
    oss << "]";
    node_builder.add(attr_name + "_" + node_name, oss.str());
  } else {
    LOG(FATAL) << "TODO: attr_type: " << attr_type;
  }
}

static void add_initializers(std::vector<const NodeArg*>& input_node_arg,
                             std::string node_binder_name, binder_t& binder,
                             onnxruntime::Graph* graph,
                             std::vector<std::string>& in_dtypes) {
  const NodeArg* node_arg_ = binder[node_binder_name].node_arg;
  if (node_arg_ && node_arg_is_constant(*graph, *node_arg_)) {
    input_node_arg.push_back(node_arg_);
    in_dtypes.push_back(vaip::dd::nodearg_dtype_to_string(*node_arg_));
  }
}

static void set_explicit_attributes(std::string name, std::string value,
                                    std::string dtype,
                                    NodeBuilder& node_builder) {
  if ("string" == dtype) {
    node_builder.add(name, value);
  } else if ("int64" == dtype) {
    int64_t v = std::stoi(value);
    node_builder.add(name, v);
  } else if ("float32" == dtype) {
    float v = std::stof(value);
    node_builder.add(name, v);
  } else if ("string[]" == dtype) {
    std::istringstream iss(value);
    std::vector<std::string> words;
    std::string word;
    while (iss >> word) {
      words.push_back(word);
    }
    node_builder.add(name, words);
  } else if ("int64[]" == dtype) {
    std::istringstream iss(value);
    std::vector<int64_t> words;
    std::string word;
    while (iss >> word) {
      words.push_back(std::stoi(word));
    }
    node_builder.add(name, words);
  } else if ("float32[]" == dtype) {
    std::istringstream iss(value);
    std::vector<float> words;
    std::string word;
    while (iss >> word) {
      words.push_back(std::stof(word));
    }
    node_builder.add(name, words);
  } else {
    throw std::runtime_error("Unsupported dtype : " + dtype +
                             " . Supported dtypes : string, int64, float32, "
                             "string[], int64[] and float32[].");
  }
}

static void set_explicit_attributes(
    google::protobuf::RepeatedPtrField<ExplicitAttributeAccessorProto>
        explicit_attributes,
    NodeBuilder& node_builder) {
  for (const auto& attr : explicit_attributes) {
    set_explicit_attributes(
        attr.attribute_name(), attr.attribute_value(),
        (attr.has_attribute_dtype() ? attr.attribute_dtype() : "string"),
        node_builder);
  }
}

static std::vector<float> get_q_params(
    google::protobuf::RepeatedPtrField<std::string> q_params_extractors,
    onnxruntime::Graph* graph, binder_t& binder) {
  if (q_params_extractors.size() & 1) {
    throw std::runtime_error("q params size should be even. size = " +
                             q_params_extractors.size());
  }

  std::vector<std::string> q_params_node_name(q_params_extractors.begin(),
                                              q_params_extractors.end());

  std::vector<float> q_params;

  for (int i = 0; i < (q_params_node_name.size() >> 1); i++) {
    int idx = i << 1;
    q_params.push_back(node_arg_get_const_data_as_float(
        *graph, *binder[q_params_node_name[idx]].node_arg));      // scale
    q_params.push_back(float(vaip::dd::get_zp_from_node(
        *graph, *binder[q_params_node_name[idx + 1]].node_arg))); // zero point
  }

  return q_params;
}

} // namespace
