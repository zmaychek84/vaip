/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */
#include <fstream>
#include <glog/logging.h>
#include <iostream>

#ifdef __GNUC__
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wpedantic"
#  pragma GCC diagnostic ignored "-Wconversion"
#endif

#include "vaip/dd/coeffs.hpp"
#include "vaip/dd/dd_utils.hpp"
#include "vaip/pattern_zoo.hpp"

#ifdef __GNUC__
#  pragma GCC diagnostic pop
#endif

#include "vaip/vaip.hpp"
#include "vitis/ai/env_config.hpp"

DEF_ENV_PARAM(DEBUG_DD_MERGE_IDENTITY, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_DD_MERGE_IDENTITY) >= n)

template <typename Type> std::string to_str(const Type& t) {
  std::ostringstream os;
  os << t;
  return os.str();
}
/**
 * test case: <???>
 *
 *
 * Replace pattern:
 *
 * From: <???>
 * To  : <???>
 */

// add the following line in your vaip_config.json
/*
    { "name": "vaip_pass_dd_merge_identity",
       "plugin": "vaip-pass_dd_merge_identity",
       "disabled": false
    }
*/
namespace {
using namespace vaip_core;
struct Dd_merge_identity {
  Dd_merge_identity(IPass& self) : self_{self} {}
  std::unique_ptr<Rule> create_rule(IPass* self) {
    auto identity = vaip::pattern_zoo::get_pattern("m_identity");
    CHECK(identity != nullptr) << "Pattern returned is null";
    std::cout << "inside Identity pattern matcher " << std::endl;
    return Rule::create_rule(
        identity, [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
          // auto in_node = binder["input"];
          auto const_node = binder["constant_1"];
          auto out_node = binder["identity"];

          // if (in_node.node == nullptr)
          //   return false;
          // std::cout << "Pattern Matched : "
          //           << node_arg_get_name(*out_node.node_arg) << std::endl;

          auto node_name = node_arg_get_name(*out_node.node_arg);
          // auto const_data =
          //     node_arg_get_const_data_as_float(*graph, *const_node.node_arg);
          auto const_name = node_arg_get_name(*const_node.node_arg);
          auto const_dtype = node_arg_get_element_type(*const_node.node_arg);

          std::vector<std::string> inputs = {};

          // std::vector<std::string> inputs = {
          //     node_arg_get_name(*in_node.node_arg)};

          std::vector<std::string> outputs = {node_name};
          auto constant_initializers = std::vector<std::string>{const_name};
          auto [meta_def, err] =
              self->try_fuse(*graph, "identity_" + node_name, inputs, outputs,
                             constant_initializers, "IDENTITY");
          if (meta_def == nullptr) {
            LOG(FATAL) << "Cannot fuse IDENTITY:  " << err.comments;
          }
          auto& generic_param = *meta_def->mutable_generic_param();

          if (const_dtype == (int)ONNX_NAMESPACE::TensorProto_DataType_UINT8) {
            generic_param["const_dtype"] = "uint8";
            auto weight =
                node_arg_get_const_data_as_u8s(*graph, *const_node.node_arg);

            generic_param["const_value"] = to_str(weight[0]);
          } else if (const_dtype ==
                     (int)ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
            generic_param["const_dtype"] = "float";
            generic_param["const_value"] = to_str(
                node_arg_get_const_data_as_float(*graph, *const_node.node_arg));
          }

          auto log_dir = self->get_log_path();

          std::string name = node_arg_get_name(*const_node.node_arg);
          std::string valid_chars =
              "-_.()"
              "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
          name.erase(std::remove_if(name.begin(), name.end(),
                                    [&valid_chars](char c) {
                                      return valid_chars.find(c) ==
                                             std::string::npos;
                                    }),
                     name.end());
          std::replace(name.begin(), name.end(), ' ', '_');

          std::string wts_bin = name + ".bin";
          auto path = log_dir / wts_bin;
          std::string wts_file{path.u8string()};

          std::ofstream file(wts_file, std::ios::binary);
          if (const_dtype == (int)ONNX_NAMESPACE::TensorProto_DataType_UINT8) {
            auto weight =
                node_arg_get_const_data_as_u8s(*graph, *const_node.node_arg);
            file.write(reinterpret_cast<const char*>(weight.data()),
                       weight.size() * sizeof(int8_t));

            generic_param["const_dtype"] = "uint8";
            generic_param["num_elements"] = to_str(weight.size());
          } else if (const_dtype ==
                     (int)ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
            auto weight =
                node_arg_get_const_data_as_floats(*graph, *const_node.node_arg);
            file.write(reinterpret_cast<const char*>(weight.data()),
                       weight.size() * sizeof(float));
            generic_param["const_dtype"] = "float";
            generic_param["num_elements"] = to_str(weight.size());
          }
          generic_param["const_filename"] = wts_file;
          file.close();

          [[maybe_unused]] auto& fused_node =
              self->fuse(*graph, std::move(*meta_def));

          return true;
        });
  }
  // apply the rule
  void process(IPass& self, Graph& graph) {
    MY_LOG(1) << self_.get_pass_proto().name() << "["
              << self_.get_pass_proto().plugin() << "] start processing graph";
    create_rule(&self)->apply(&graph);
    MY_LOG(1) << self_.get_pass_proto().name() << "["
              << self_.get_pass_proto().plugin() << "] finish processing graph";
  }

  IPass& self_;
};
} // namespace

DEFINE_VAIP_PASS(Dd_merge_identity, vaip_pass_dd_merge_identity)
