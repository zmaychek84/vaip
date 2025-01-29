/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */
#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "vaip/dd/coeffs.hpp"
#include "vaip/dd/dd_utils.hpp"
#include "vaip/node.hpp"
#include "vaip/vaip.hpp"
#include "vitis/ai/env_config.hpp"
#include <functional>
#include <glog/logging.h>
#include <numeric>

DEF_ENV_PARAM(DEBUG_CPP_PATTERN, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_CPP_PATTERN) >= n)
using namespace vaip_core;
[[maybe_unused]] static std::string
vector_floats_to_string(const std::vector<float>& values) {
  std::ostringstream str;
  auto sep = std::string("");
  for (auto value : values) {
    str << sep << value;
    sep = " ";
  }
  return str.str();
}

[[maybe_unused]] static std::string
vector_int_to_string(const std::vector<int>& values) {
  std::ostringstream str;
  auto sep = std::string("");
  for (auto value : values) {
    str << sep << value;
    sep = " ";
  }
  return str.str();
}

[[maybe_unused]] static std::string
write_const_to_file(std::string const_name, onnxruntime::Graph& graph,
                    const NodeArg& node_arg, std::filesystem::path& log_dir) {
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

int mlp_ = 0;

struct MladfMlp {
  MladfMlp(IPass& self) : self_{self}, log_dir_{self.get_log_path()} {}
  static std::unique_ptr<Rule> create_rule(IPass* self, int8_t dry_run_) {
    auto builder = PatternBuilder();
    // Primary Input = o/p from SkipSimplifiedLayerNormalization
    auto p_input = builder.wildcard();
    // Cast primary
    auto cast_gp = builder.node2("Cast", {p_input});
    auto cast_up = builder.node2("Cast", {p_input});
    // Gate projection constants
    auto gate_wts = builder.wildcard();
    auto gate_scl = builder.wildcard();
    auto gate_zps = builder.wildcard();
    auto zp = builder.constant();
    // Up projection constants
    auto up_wts = builder.wildcard();
    auto up_scl = builder.wildcard();
    auto up_zps = builder.wildcard();
    // Down projection constants
    auto down_wts = builder.wildcard();
    auto down_scl = builder.wildcard();
    auto down_zps = builder.wildcard();
    // Gate projection MatMul
    auto gate_proj = builder.node3("com.microsoft:MatMulNBits",
                                   {cast_gp, gate_wts, gate_scl, gate_zps},
                                   {false, false, false, true});
    // Up projection MatMul
    auto up_proj = builder.node3("com.microsoft:MatMulNBits",
                                 {cast_up, up_wts, up_scl, up_zps},
                                 {false, false, false, true});
    // Cast gp output
    auto cast_gp_out = builder.node2("Cast", {gate_proj});
    // Cast up output
    auto cast_up_out = builder.node2("Cast", {up_proj});

    //// GP down
    auto cast_gp_left = builder.node2("Cast", {cast_gp_out});
    auto cast_gp_right = builder.node2("Cast", {cast_gp_out});

    // Sigmoid
    auto sigmoid = builder.node2("Sigmoid", {cast_gp_right});
    auto sigmoid_bf16 = builder.node2("Cast", {sigmoid});
    auto sigmoid_fp32 = builder.node2("Cast", {sigmoid_bf16});
    // Mul
    auto mul_1 = builder.node2("Mul", {cast_gp_left, sigmoid_fp32});
    auto mul_1_bf16 = builder.node2("Cast", {mul_1});
    auto mul_1_fp32 = builder.node2("Cast", {mul_1_bf16});
    // Up out cast fp32
    auto cast_up_fp32 = builder.node2("Cast", {cast_up_out});
    // Mul
    auto mul_2 = builder.node2("Mul", {mul_1_fp32, cast_up_fp32});
    // DP in cast
    auto mul_2_bf16 = builder.node2("Cast", {mul_2});
    auto mul_2_fp32 = builder.node2("Cast", {mul_2_bf16});
    // Down projection MatMul
    auto down_proj = builder.node3("com.microsoft:MatMulNBits",
                                   {mul_2_fp32, down_wts, down_scl, down_zps},
                                   {false, false, false, true});
    // DP out cast
    auto down_proj_out = builder.node2("Cast", {down_proj});

    return Rule::create_rule(
        down_proj_out,
        [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
          std::vector<std::string> attr_nodes;
          for (auto& ni : binder) {
            if (!(*node_arg_is_constant)(*graph, *ni.second.node_arg)) {
              attr_nodes.push_back(node_arg_get_name(*ni.second.node_arg));
            }
          }

          // std::cout << "- MLP Pattern matches ... " << std::endl;

          auto p_input_node = binder[p_input->get_id()];
          // Gate projection constants
          auto gate_wts_node = binder[gate_wts->get_id()];
          auto gate_scl_node = binder[gate_scl->get_id()];
          auto gate_zps_node = binder[gate_zps->get_id()];
          // Up projection constants
          auto up_wts_node = binder[up_wts->get_id()];
          auto up_scl_node = binder[up_scl->get_id()];
          auto up_zps_node = binder[up_zps->get_id()];
          // Down projection constants
          auto down_wts_node = binder[down_wts->get_id()];
          auto down_scl_node = binder[down_scl->get_id()];
          auto down_zps_node = binder[down_zps->get_id()];
          // MatMuls
          auto gate_proj_node = binder[gate_proj->get_id()];
          auto up_proj_node = binder[up_proj->get_id()];
          auto down_proj_node = binder[down_proj->get_id()];
          auto down_proj_cast_node = binder[down_proj_out->get_id()];

          // Get cache dir
          auto log_dir = self->get_log_path();

          // Inputs
          std::vector<std::string> inputs;
          // primary input
          inputs.push_back(node_arg_get_name(*p_input_node.node_arg));

          // Save const data in bin files
          //////////////////////////////////////////////////////////////////////////////

          ////// Gate Projection
          int64_t gp_k = node_get_attr_int(*gate_proj_node.node, "K");
          int64_t gp_n = node_get_attr_int(*gate_proj_node.node, "N");
          int64_t gp_bits = node_get_attr_int(*gate_proj_node.node, "bits");
          int64_t gp_bsize =
              node_get_attr_int(*gate_proj_node.node, "block_size");

          std::string gate_wts_name =
              node_arg_get_name(*gate_wts_node.node_arg);
          std::string gate_scl_name =
              node_arg_get_name(*gate_scl_node.node_arg);

          inputs.push_back(gate_wts_name);
          inputs.push_back(gate_scl_name);

          auto gp_wts_f = write_const_to_file(gate_wts_name, *graph,
                                              *gate_wts_node.node_arg, log_dir);
          auto gp_scl_f = write_const_to_file(gate_scl_name, *graph,
                                              *gate_scl_node.node_arg, log_dir);

          // Check if zp is present
          std::string gp_zps_f;
          if (gate_zps_node.node_arg != nullptr) {
            std::string gate_zps_name =
                node_arg_get_name(*gate_zps_node.node_arg);
            inputs.push_back(gate_zps_name);
            gp_zps_f = write_const_to_file(gate_zps_name, *graph,
                                           *gate_zps_node.node_arg, log_dir);
          }

          ////// Up Projection
          int64_t up_k = node_get_attr_int(*up_proj_node.node, "K");
          int64_t up_n = node_get_attr_int(*up_proj_node.node, "N");
          int64_t up_bits = node_get_attr_int(*up_proj_node.node, "bits");
          int64_t up_bsize =
              node_get_attr_int(*up_proj_node.node, "block_size");

          std::string up_wts_name = node_arg_get_name(*up_wts_node.node_arg);
          std::string up_scl_name = node_arg_get_name(*up_scl_node.node_arg);

          inputs.push_back(up_wts_name);
          inputs.push_back(up_scl_name);

          auto up_wts_f = write_const_to_file(up_wts_name, *graph,
                                              *up_wts_node.node_arg, log_dir);
          auto up_scl_f = write_const_to_file(up_scl_name, *graph,
                                              *up_scl_node.node_arg, log_dir);

          // Check if zp is present
          std::string up_zps_f;
          if (up_zps_node.node_arg != nullptr) {
            std::string up_zps_name = node_arg_get_name(*up_zps_node.node_arg);
            inputs.push_back(up_zps_name);
            up_zps_f = write_const_to_file(up_zps_name, *graph,
                                           *up_zps_node.node_arg, log_dir);
          }

          ////// Down Projection
          int64_t dp_k = node_get_attr_int(*down_proj_node.node, "K");
          int64_t dp_n = node_get_attr_int(*down_proj_node.node, "N");
          int64_t dp_bits = node_get_attr_int(*down_proj_node.node, "bits");
          int64_t dp_bsize =
              node_get_attr_int(*down_proj_node.node, "block_size");

          std::string down_wts_name =
              node_arg_get_name(*down_wts_node.node_arg);
          std::string down_scl_name =
              node_arg_get_name(*down_scl_node.node_arg);

          inputs.push_back(down_wts_name);
          inputs.push_back(down_scl_name);

          auto dp_wts_f = write_const_to_file(down_wts_name, *graph,
                                              *down_wts_node.node_arg, log_dir);
          auto dp_scl_f = write_const_to_file(down_scl_name, *graph,
                                              *down_scl_node.node_arg, log_dir);

          // Check if zp is present
          std::string dp_zps_f;
          if (down_zps_node.node_arg != nullptr) {
            std::string down_zps_name =
                node_arg_get_name(*down_zps_node.node_arg);
            inputs.push_back(down_zps_name);
            dp_zps_f = write_const_to_file(down_zps_name, *graph,
                                           *down_zps_node.node_arg, log_dir);
          }

          // Primary output name
          std::string p_output_name =
              node_arg_get_name(*down_proj_cast_node.node_arg);
          std::vector<std::string> outputs = {p_output_name};
          auto constant_initializers = std::vector<std::string>{};
          auto [meta_def, err] =
              self->try_fuse(*graph, "mladf_mlp_" + inputs[0], inputs, outputs,
                             constant_initializers, "MLP");
          if (meta_def == nullptr) {
            LOG(FATAL) << "- Cannot fuse MLADF MLP:  " << err.comments;
          }

          // Add generic params in meta-def
          auto& generic_param = *meta_def->mutable_generic_param();
          // Add a name
          generic_param["node_name"] = "mladf_mlp_" + inputs[0];
          generic_param["cnt"] = std::to_string(mlp_);
          generic_param["dry_run"] = std::to_string(dry_run_);

          // Add attributes
          generic_param["gp_wts_file"] = gp_wts_f;
          generic_param["gp_scl_file"] = gp_scl_f;
          if (!gp_zps_f.empty()) {
            generic_param["gp_zps_file"] = gp_zps_f;
          }
          generic_param["gp_K"] = std::to_string(gp_k);
          generic_param["gp_N"] = std::to_string(gp_n);
          generic_param["gp_bits"] = std::to_string(gp_bits);
          generic_param["gp_block_size"] = std::to_string(gp_bsize);

          generic_param["up_wts_file"] = up_wts_f;
          generic_param["up_scl_file"] = up_scl_f;
          if (!up_zps_f.empty()) {
            generic_param["up_zps_file"] = up_zps_f;
          }
          generic_param["up_K"] = std::to_string(up_k);
          generic_param["up_N"] = std::to_string(up_n);
          generic_param["up_bits"] = std::to_string(up_bits);
          generic_param["up_block_size"] = std::to_string(up_bsize);

          generic_param["dp_wts_file"] = dp_wts_f;
          generic_param["dp_scl_file"] = dp_scl_f;
          if (!dp_zps_f.empty()) {
            generic_param["dp_zps_file"] = dp_zps_f;
          }
          generic_param["dp_K"] = std::to_string(dp_k);
          generic_param["dp_N"] = std::to_string(dp_n);
          generic_param["dp_bits"] = std::to_string(dp_bits);
          generic_param["dp_block_size"] = std::to_string(dp_bsize);

          // Fuse
          [[maybe_unused]] auto& fused_node =
              self->fuse(*graph, std::move(*meta_def));
          mlp_++;
          return true;
        });
  }
  void process(IPass& self, Graph& graph) {
    const auto& session_option = self.get_config_proto().provider_options();
    int8_t dry_run_en_ = 0;
    if (session_option.find("dry_run") != session_option.end()) {
      const auto& dry_run = session_option.find("dry_run")->second;
      // TODO: correct
      if (dry_run == "true") {
        dry_run_en_ = 1;
      }
    }
    create_rule(&self, dry_run_en_)->apply(&graph);
  }

public:
  IPass& self_;
  const std::filesystem::path& log_dir_;
};
DEFINE_VAIP_PASS(MladfMlp, vaip_pass_mladf_mlp)
