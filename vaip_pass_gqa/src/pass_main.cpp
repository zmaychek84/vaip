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
#include <fstream>
#include <functional>
#include <glog/logging.h>
#include <numeric>

DEF_ENV_PARAM(DEBUG_GQA, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_GQA) >= n)

using namespace vaip_core;
[[maybe_unused]] static std::string
write_const_to_file(std::string const_name, onnxruntime::Graph& graph,
                    const NodeArg& node_arg,
                    const std::filesystem::path& log_dir) {
  std::string valid_chars =
      "-_.()"
      "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
  const_name.erase(std::remove_if(const_name.begin(), const_name.end(),
                                  [&valid_chars](char c) {
                                    return valid_chars.find(c) ==
                                           std::string::npos;
                                  }),
                   const_name.end());
  std::replace(const_name.begin(), const_name.end(), ' ', '_');
  // File path/name
  std::string _bin = const_name + ".bin";
  auto path = log_dir / _bin;
  std::string _bin_file{path.u8string()};
  // Open file
  std::ofstream file(_bin_file, std::ios::binary);
  // TODO: Read const data from graph and dump into file inside
  // cache dir
  auto dtype = node_arg_get_element_type(node_arg);
  if (dtype == (int)ONNX_NAMESPACE::TensorProto_DataType_UINT8) {
    gsl::span<const uint8_t> const_tensor =
        node_arg_get_const_data_as_u8s(graph, node_arg);
    auto tensor_size = const_tensor.size() * sizeof(uint8_t);
    // Write to file
    file.write(reinterpret_cast<const char*>(const_tensor.data()), tensor_size);
  } else if (dtype == (int)ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
    gsl::span<const float> const_tensor =
        node_arg_get_const_data_as_floats(graph, node_arg);
    auto tensor_size = const_tensor.size() * sizeof(float);
    // Write to file
    file.write(reinterpret_cast<const char*>(const_tensor.data()), tensor_size);
  } else {
    CHECK(dtype == (int)ONNX_NAMESPACE::TensorProto_DataType_UINT8 ||
          dtype == (int)ONNX_NAMESPACE::TensorProto_DataType_FLOAT)
        << "Currently only uint8 and float are supported";
  }
  // Close file
  file.close();
  return _bin_file;
}

int gqa_cnt = 0;
struct GQA {

  GQA(IPass& self) : self_{self}, log_dir_{self.get_log_path()} {}

  std::unique_ptr<Rule> create_rule(IPass* self) {
    auto builder = PatternBuilder();

    auto input_0 = builder.wildcard();
    auto cast_0 = builder.node2("Cast", {input_0});

    auto key = builder.wildcard();
    auto value = builder.wildcard();

    auto past_key = builder.wildcard();
    auto past_value = builder.wildcard();
    auto past_seq_len = builder.wildcard();

    auto total_seq_len = builder.wildcard();
    auto cos_cache = builder.constant();
    auto sin_cache = builder.constant();

    auto GQA_ = builder.node3(
        "com.microsoft:GroupQueryAttention",
        {cast_0, key, value, past_key, past_value, past_seq_len, total_seq_len,
         cos_cache, sin_cache},
        {false, true, true, false, false, false, false, false, false});

    auto cast_GQA_out = builder.node2("Cast", {GQA_});

    return Rule::create_rule(
        cast_GQA_out, [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
          auto input_0_node = binder[input_0->get_id()];
          auto key_node = binder[key->get_id()];
          auto value_node = binder[value->get_id()];
          auto past_key_node = binder[past_key->get_id()];
          auto past_value_node = binder[past_value->get_id()];
          auto past_seq_len_node = binder[past_seq_len->get_id()];
          auto total_seq_len_node = binder[total_seq_len->get_id()];
          auto cos_cache_node = binder[cos_cache->get_id()];
          auto sin_cache_node = binder[sin_cache->get_id()];
          auto GQA_node = binder[GQA_->get_id()];
          auto out_node_0 = binder[cast_GQA_out->get_id()];
          auto node_outputs = node_get_output_node_args(*GQA_node.node);
          const NodeArg& present_key = *(node_outputs[1]);
          const NodeArg& present_value = *(node_outputs[2]);
          auto GQA_name = node_arg_get_name(*GQA_node.node_arg);

          int64_t do_rotary = node_get_attr_int(*GQA_node.node, "do_rotary");
          int64_t kv_num_heads =
              node_get_attr_int(*GQA_node.node, "kv_num_heads");
          int64_t num_heads = node_get_attr_int(*GQA_node.node, "num_heads");

          std::string cos_cache_file, sin_cache_file;
          auto cos_cache_name = node_arg_get_name(*cos_cache_node.node_arg);
          auto sin_cache_name = node_arg_get_name(*sin_cache_node.node_arg);
          cos_cache_file = write_const_to_file(
              cos_cache_name, *graph, *cos_cache_node.node_arg, log_dir_);
          sin_cache_file = write_const_to_file(
              sin_cache_name, *graph, *sin_cache_node.node_arg, log_dir_);

          auto cos_cache_shape = vaip::dd::shape_as_string(
              *(node_arg_get_shape_i64(*cos_cache_node.node_arg)));
          auto sin_cache_shape = vaip::dd::shape_as_string(
              *(node_arg_get_shape_i64(*sin_cache_node.node_arg)));
          // std::cout<<"cos cache shape: "<<cos_cache_shape<<std::endl;

          std::vector<std::string> inputs, outputs;
          bool is_key_present = (key_node.node_arg != nullptr);
          bool is_value_present = (value_node.node_arg != nullptr);

          inputs.push_back(node_arg_get_name(*input_0_node.node_arg));
          if (is_key_present)
            inputs.push_back(node_arg_get_name(*key_node.node_arg));
          if (is_value_present)
            inputs.push_back(node_arg_get_name(*value_node.node_arg));
          inputs.push_back(node_arg_get_name(*past_key_node.node_arg));
          inputs.push_back(node_arg_get_name(*past_value_node.node_arg));
          inputs.push_back(node_arg_get_name(*past_seq_len_node.node_arg));
          inputs.push_back(node_arg_get_name(*total_seq_len_node.node_arg));
          inputs.push_back(cos_cache_name);
          inputs.push_back(sin_cache_name);

          outputs.push_back(node_arg_get_name(*out_node_0.node_arg));
          outputs.push_back(node_arg_get_name(present_key));
          outputs.push_back(node_arg_get_name(present_value));

          auto constant_initializers = std::vector<std::string>{};
          auto [meta_def, err] =
              self->try_fuse(*graph, "vaip_" + GQA_name, inputs, outputs,
                             constant_initializers, "GQA");
          if (meta_def == nullptr) {
            LOG(FATAL) << "Cannot fuse GQA:  " << err.comments;
          }
          auto& generic_param = *meta_def->mutable_generic_param();

          // metadef
          generic_param["cnt"] = std::to_string(gqa_cnt);
          generic_param["node_name"] = "vaip_" + GQA_name;
          generic_param["cos_cache_file"] = cos_cache_file;
          generic_param["sin_cache_file"] = sin_cache_file;
          generic_param["cos_cache_shape"] = cos_cache_shape;
          generic_param["sin_cache_shape"] = sin_cache_shape;
          generic_param["do_rotary"] = std::to_string(do_rotary);
          generic_param["kv_num_heads"] = std::to_string(kv_num_heads);
          generic_param["num_heads"] = std::to_string(num_heads);
          generic_param["is_packed_qkv"] = std::to_string(!is_key_present);

          [[maybe_unused]] auto& fused_node =
              self->fuse(*graph, std::move(*meta_def));
          gqa_cnt++;
          return true;
        });
  }
  // apply the rule
  void process(IPass& self, Graph& graph) { create_rule(&self)->apply(&graph); }

public:
  IPass& self_;
  const std::filesystem::path& log_dir_;
};

DEFINE_VAIP_PASS(GQA, vaip_pass_gqa)
