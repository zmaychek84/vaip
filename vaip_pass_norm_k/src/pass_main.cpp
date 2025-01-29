/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */
#include <fstream>
#include <glog/logging.h>
#include <iostream>

#include "vaip/vaip.hpp"
#include "vitis/ai/env_config.hpp"

DEF_ENV_PARAM(DEBUG_NORM_K, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_NORM_K) >= n)

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
    { "name": "vaip_pass_norm_k",
       "plugin": "vaip-pass_norm_k",
       "disabled": false
    }
*/
namespace {
using namespace vaip_core;
int cnt = 0;

template <typename T>
void save_vec_span_2_bin(const gsl::span<const T>& span,
                         const std::string& filename) {
  std::ofstream outFile(filename, std::ios::binary);
  if (!outFile) {
    std::cerr << "Error opening file for writing: " << filename << std::endl;
    return;
  }

  for (const auto& data : span) {
    outFile.write(reinterpret_cast<const char*>(&data), sizeof(T));
  }

  outFile.close();
}

template <typename T>
void save_vec_span_2_bin(const std::vector<T>& vec,
                         const std::string& filename) {
  save_vec_span_2_bin(gsl::span<const T>(vec.data(), vec.size()), filename);
}

struct Norm_k {
  Norm_k(IPass& self) : self_{self} {}
  std::unique_ptr<Rule> create_rule(IPass* self) {
    auto ret = std::shared_ptr<vaip_core::Pattern>();
#include "./norm_k_pattern.h.inc"
    auto pattern_ = ret;
    return Rule::create_rule(
        pattern_, [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
          auto input_0_node = binder[input_0->get_id()];

          auto constant_0_node = binder
              [constant_0
                   ->get_id()]; // /norm_k/Constant_1_output_0_QuantizeLinear_Output,
                                // add input
          auto constant_1_node =
              binder[constant_1->get_id()]; // /norm_k/Constant_1_output_0_scale
          auto constant_2_node =
              binder[constant_2
                         ->get_id()]; // /norm_k/Constant_1_output_0_zero_point
          auto constant_3_node = binder
              [constant_3
                   ->get_id()]; // /norm_k/Constant_output_0_QuantizeLinear_Output,
                                // pow second input
          auto constant_4_node =
              binder[constant_4->get_id()]; // /norm_k/Constant_output_0_scale,
                                            // scale
          auto constant_5_node =
              binder[constant_5
                         ->get_id()]; // /norm_k/Constant_output_0_zero_point,
                                      // zp
          auto constant_6_node =
              binder[constant_6->get_id()]; // /pos_emb/Constant_output_0,
                                            // indices for gather
          auto constant_7_node =
              binder[constant_7
                         ->get_id()]; // encoder.pos_emb.pe_k.weight_scale,
                                      // scale for input_0
          auto constant_8_node =
              binder[constant_8
                         ->get_id()]; // encoder.pos_emb.pe_k.weight_zero_point,
                                      // zp for input_0
          auto constant_9_node =
              binder[constant_9->get_id()]; // 7379, ReduceMean axes
          auto constant_10_node =
              binder[constant_10
                         ->get_id()]; // /norm_k/ReduceMean_output_0_scale,
                                      // scale
          auto constant_11_node =
              binder[constant_11
                         ->get_id()]; // /norm_k/ReduceMean_output_0_zero_point,
                                      // zp
          auto constant_12_node =
              binder[constant_12->get_id()]; // /norm_k/Sub_output_0_scale,
                                             // scale
          auto constant_13_node =
              binder[constant_13->get_id()]; // /norm_k/Sub_output_0_zero_point,
                                             // zp
          auto constant_14_node =
              binder[constant_14->get_id()]; // /norm_k/Pow_output_0_scale,
                                             // scale
          auto constant_15_node =
              binder[constant_15->get_id()]; // /norm_k/Pow_output_0_zero_point,
                                             // zp
          auto constant_16_node =
              binder[constant_16
                         ->get_id()]; // /norm_k/ReduceMean_1_output_0_scale,
                                      // scale
          auto constant_17_node = binder
              [constant_17
                   ->get_id()]; // /norm_k/ReduceMean_1_output_0_zero_point,
                                // zp
          auto constant_18_node =
              binder[constant_18->get_id()]; // /norm_k/Add_output_0_scale,
                                             // scale
          auto constant_19_node =
              binder[constant_19->get_id()]; // /norm_k/Add_output_0_zero_point,
                                             // zp
          auto constant_20_node =
              binder[constant_20->get_id()]; // /norm_k/Sqrt_output_0_scale,
                                             // scale
          auto constant_21_node =
              binder[constant_21
                         ->get_id()]; // /norm_k/Sqrt_output_0_zero_point,
                                      // zp
          auto constant_22_node =
              binder[constant_22->get_id()]; // /norm_k/Div_output_0_scale,
                                             // scale
          auto constant_23_node =
              binder[constant_23->get_id()]; // /norm_k/Div_output_0_zero_point,
                                             // zp

          auto Gather_0_node =
              binder[Gather_0->get_id()]; // /pos_emb/pe_k/Gather_output_0,
                                          // Gather

          auto com_microsoft_QuantizeLinear_8_node =
              binder[com_microsoft_QuantizeLinear_8->get_id()];
          std::string input_0_name = node_arg_get_name(*input_0_node.node_arg);
          std::string constant_0_name =
              node_arg_get_name(*constant_0_node.node_arg);
          std::string Gather_0_name =
              node_arg_get_name(*constant_0_node.node_arg);
          std::vector<std::string> inputs{input_0_name, constant_0_name};

          std::string com_microsoft_QuantizeLinear_8_name =
              node_arg_get_name(*com_microsoft_QuantizeLinear_8_node.node_arg);
          std::vector<std::string> outputs = {
              com_microsoft_QuantizeLinear_8_name};
          std::vector<float> const_scale;
          auto constant_initializers = std::vector<std::string>{};
          auto const_0_data =
              node_arg_get_const_data_as_u16(*graph, *constant_0_node.node_arg);
          auto const_1_data = node_arg_get_const_data_as_float(
              *graph, *constant_1_node.node_arg);
          auto const_2_data =
              node_arg_get_const_data_as_u16(*graph, *constant_2_node.node_arg);
          auto const_3_data =
              node_arg_get_const_data_as_u16(*graph, *constant_3_node.node_arg);
          auto const_4_data = node_arg_get_const_data_as_float(
              *graph, *constant_4_node.node_arg);
          auto const_5_data =
              node_arg_get_const_data_as_u16(*graph, *constant_5_node.node_arg);
          auto const_6_data = node_arg_get_const_data_as_i64s(
              *graph, *constant_6_node.node_arg);
          int64_t gather_axis = node_get_attr_int(*Gather_0_node.node, "axis");
          auto const_7_data = node_arg_get_const_data_as_float(
              *graph, *constant_7_node.node_arg);
          auto const_8_data =
              node_arg_get_const_data_as_u16(*graph, *constant_8_node.node_arg);
          auto const_10_data = node_arg_get_const_data_as_float(
              *graph, *constant_10_node.node_arg);
          auto const_11_data = node_arg_get_const_data_as_u16(
              *graph, *constant_11_node.node_arg);
          auto const_12_data = node_arg_get_const_data_as_float(
              *graph, *constant_12_node.node_arg);
          auto const_13_data = node_arg_get_const_data_as_u16(
              *graph, *constant_13_node.node_arg);
          auto const_14_data = node_arg_get_const_data_as_float(
              *graph, *constant_14_node.node_arg);
          auto const_15_data = node_arg_get_const_data_as_u16(
              *graph, *constant_15_node.node_arg);
          auto const_16_data = node_arg_get_const_data_as_float(
              *graph, *constant_16_node.node_arg);
          auto const_17_data = node_arg_get_const_data_as_u16(
              *graph, *constant_17_node.node_arg);
          auto const_18_data = node_arg_get_const_data_as_float(
              *graph, *constant_18_node.node_arg);
          auto const_19_data = node_arg_get_const_data_as_u16(
              *graph, *constant_19_node.node_arg);
          auto const_20_data = node_arg_get_const_data_as_float(
              *graph, *constant_20_node.node_arg);
          auto const_21_data = node_arg_get_const_data_as_u16(
              *graph, *constant_21_node.node_arg);
          auto const_22_data = node_arg_get_const_data_as_float(
              *graph, *constant_22_node.node_arg);
          auto const_23_data = node_arg_get_const_data_as_u16(
              *graph, *constant_23_node.node_arg);
          auto const_9_data = node_arg_get_const_data_as_i64s(
              *graph, *constant_9_node.node_arg);

          auto log_dir = self->get_log_path();
          std::string name = node_arg_get_name(*input_0_node.node_arg);
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
          std::string input_0_file{path.u8string()};
          auto input_0_data =
              node_arg_get_const_data_as_u16s(*graph, *input_0_node.node_arg);
          save_vec_span_2_bin(input_0_data, input_0_file);

          path = log_dir / "gather_indices.bin";
          std::string gather_indices_file{path.u8string()};
          save_vec_span_2_bin(const_6_data, gather_indices_file);

          path = log_dir / "scale.bin";
          std::string scale_file{path.u8string()};
          std::vector<float> scale_const = {
              const_1_data,  const_4_data,  const_7_data,  const_10_data,
              const_12_data, const_14_data, const_16_data, const_18_data,
              const_20_data, const_22_data};
          save_vec_span_2_bin(scale_const, scale_file);

          auto [meta_def, err] =
              self->try_fuse(*graph, input_0_name + "_normk_custom", inputs,
                             outputs, constant_initializers, "NORMK");
          if (meta_def == nullptr) {
            LOG(FATAL) << "Cannot fuse norm_k pattern in gt:  " << err.comments;
          }
          auto& generic_param = *meta_def->mutable_generic_param();
          generic_param["cnt"] = std::to_string(cnt);
          generic_param["input_0_file"] = input_0_file;
          generic_param["scale_file"] = scale_file;
          generic_param["gather_indices_file"] = gather_indices_file;
          generic_param["gather_axis"] = std::to_string(gather_axis);
          generic_param["const_0"] = std::to_string(const_0_data);
          generic_param["const_2"] = std::to_string(const_2_data);
          generic_param["const_3"] = std::to_string(const_3_data);
          generic_param["const_5"] = std::to_string(const_5_data);
          generic_param["const_8"] = std::to_string(const_8_data);
          // here for normk in gt, only have one elment in axes array
          generic_param["const_9"] = std::to_string(const_9_data.front());
          generic_param["const_11"] = std::to_string(const_11_data);
          generic_param["const_13"] = std::to_string(const_13_data);
          generic_param["const_15"] = std::to_string(const_15_data);
          generic_param["const_17"] = std::to_string(const_17_data);
          generic_param["const_19"] = std::to_string(const_19_data);
          generic_param["const_21"] = std::to_string(const_21_data);
          generic_param["const_23"] = std::to_string(const_23_data);

          MY_LOG(1) << "Sample log message.";
          [[maybe_unused]] auto& fused_node =
              self->fuse(*graph, std::move(*meta_def));

          cnt++;
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

DEFINE_VAIP_PASS(Norm_k, vaip_pass_norm_k)
