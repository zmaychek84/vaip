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
// DEF_ENV_PARAM(NUM_GEMM, "1")

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

std::tuple<bool, std::string, std::string>
check_bias(onnxruntime::Graph* graph, const NodeInput& out_node) {
  bool if_bias_matmul = false;
  std::string last_cast_name;
  std::string bias_name;
  std::vector<const Node*> level_1_next_nodes =
      graph_get_consumer_nodes(*graph, node_arg_get_name(*out_node.node_arg));
  if (level_1_next_nodes.size() == 1 &&
      VAIP_ORT_API(node_op_type)(*level_1_next_nodes[0]) ==
          "Cast") { // level-1 :  cast node
    std::vector<const Node*> level_2_next_nodes = graph_get_consumer_nodes(
        *graph, node_get_first_output_name(*level_1_next_nodes[0]));
    if (level_2_next_nodes.size() == 1 &&
        VAIP_ORT_API(node_op_type)(*level_2_next_nodes[0]) ==
            "Add") { // level-2 : Add node
      std::vector<const Node*> level_3_next_nodes = graph_get_consumer_nodes(
          *graph, node_get_first_output_name(*level_2_next_nodes[0]));
      if (level_3_next_nodes.size() == 1 &&
          VAIP_ORT_API(node_op_type)(*level_3_next_nodes[0]) ==
              "Cast") { // level-3 :CastNode
        if_bias_matmul = true;
        last_cast_name = node_get_first_output_name(*level_3_next_nodes[0]);
        auto add_inputs = node_get_input_node_args(*level_2_next_nodes[0]);
        bias_name = node_arg_get_name(*add_inputs[1]);
      }
    }
  }
  return std::make_tuple(if_bias_matmul, last_cast_name, bias_name);
}

int cnt = 0;
static std::vector<std::string> present_shapes;
struct MatMulNBits {
  MatMulNBits(IPass& self) : self_{self}, log_dir_{self.get_log_path()} {}
  static std::unique_ptr<Rule> create_rule(IPass* self) {

    auto log_dir = self->get_log_path();
    std::string shape_bin = "matmulnbits_shapes.bin";
    auto path_shape_bin = log_dir / shape_bin;
    std::string shape_bin_file{path_shape_bin.u8string()};
    std::ofstream shape_file(shape_bin_file, std::ios::binary | std::ios::out);
    shape_file.close();
    auto builder = PatternBuilder();
    auto p_input = builder.wildcard();
    auto input_w = builder.wildcard();
    auto scales = builder.wildcard();
    auto zp = builder.constant();

    auto cast_in = builder.node2("Cast", {p_input});

    auto matmul = builder.node3("com.microsoft:MatMulNBits",
                                {cast_in, input_w, scales, zp},
                                {false, false, false, true});

    auto cast_out = builder.node2("Cast", {matmul});

    return Rule::create_rule(
        cast_out, [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
          std::vector<std::string> attr_nodes;
          for (auto& ni : binder) {
            if (!(*node_arg_is_constant)(*graph, *ni.second.node_arg)) {
              attr_nodes.push_back(node_arg_get_name(*ni.second.node_arg));
            }
          }

          auto p_input_node = binder[p_input->get_id()];
          auto input_w_node = binder[input_w->get_id()];
          auto scales_node = binder[scales->get_id()];
          auto zp_node = binder[zp->get_id()];
          auto matmul_node = binder[matmul->get_id()];
          auto cast_out_node = binder[cast_out->get_id()];

          // Check if bias add is after the pattern that got matched here, if
          // yes, set the flag, get the bias name and output name
          bool is_bais_matmul = false;
          auto bias_tuple = check_bias(graph, cast_out_node);
          std::string output_node_name, bias_name;
          if (std::get<0>(bias_tuple)) {
            is_bais_matmul = true;
            output_node_name = std::get<1>(bias_tuple);
            bias_name = std::get<2>(bias_tuple);
          }

          bool is_zp_present = false;
          if (zp_node.node_arg != nullptr) {
            is_zp_present = true;
          }

          int64_t m_k = node_get_attr_int(*matmul_node.node, "K");
          int64_t m_n = node_get_attr_int(*matmul_node.node, "N");
          int64_t bits = node_get_attr_int(*matmul_node.node, "bits");
          int64_t block_size =
              node_get_attr_int(*matmul_node.node, "block_size");

          auto log_dir = self->get_log_path();
          std::vector<std::string> ns = vaip::dd::get_node_names(graph, binder);

          // push_back node_names

          std::string name = node_arg_get_name(*input_w_node.node_arg);
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

          // Creating shape_list file commaon for all matmulnbits node to be
          // consumed in cutom op
          std::string shape_bin = "matmulnbits_shapes.bin";
          auto path_shape_bin = log_dir / shape_bin;
          std::string shape_bin_file{path_shape_bin.u8string()};

          std::string shape = std::to_string(m_k) + "_" + std::to_string(m_n);
          if (std::find(present_shapes.begin(), present_shapes.end(), shape) ==
              present_shapes.end()) {
            std::ofstream shape_file(shape_bin_file, std::ios::binary |
                                                         std::ios::app |
                                                         std::ios::out);
            // Appending K,N,Block_Size
            shape_file.write(reinterpret_cast<const char*>(&m_k),
                             sizeof(int64_t));
            shape_file.write(reinterpret_cast<const char*>(&m_n),
                             sizeof(int64_t));
            shape_file.write(reinterpret_cast<const char*>(&block_size),
                             sizeof(int64_t));
            shape_file.close();
            present_shapes.push_back(shape);
          }

          std::string wts_bin = name + ".bin";
          auto path = log_dir / wts_bin;
          std::string wts_file{path.u8string()};

          std::ofstream file(wts_file, std::ios::binary);
          auto weight =
              node_arg_get_const_data_as_u8s(*graph, *input_w_node.node_arg);
          file.write(reinterpret_cast<const char*>(weight.data()),
                     weight.size() * sizeof(int8_t));
          file.close();

          std::string name1 = node_arg_get_name(*scales_node.node_arg);
          name1.erase(std::remove_if(name1.begin(), name1.end(),
                                     [&valid_chars](char c) {
                                       return valid_chars.find(c) ==
                                              std::string::npos;
                                     }),
                      name1.end());
          std::replace(name1.begin(), name1.end(), ' ', '_');

          std::string scl_bin = name1 + ".bin";
          auto path_1 = log_dir / scl_bin;
          std::string scl_file{path_1.u8string()};

          std::ofstream file1(scl_file, std::ios::binary);
          // Get scale tensor
          auto scale =
              node_arg_get_const_data_as_floats(*graph, *scales_node.node_arg);
          file1.write(reinterpret_cast<const char*>(scale.data()),
                      scale.size() * sizeof(float));
          file1.close();

          std::string zp_file;
          if (is_zp_present) {
            std::string zp_name = node_arg_get_name(*zp_node.node_arg);
            zp_name.erase(std::remove_if(zp_name.begin(), zp_name.end(),
                                         [&valid_chars](char c) {
                                           return valid_chars.find(c) ==
                                                  std::string::npos;
                                         }),
                          zp_name.end());
            std::replace(zp_name.begin(), zp_name.end(), ' ', '_');
            std::string zp_bin = zp_name + ".bin";
            auto path_2 = log_dir / zp_bin;
            zp_file = path_2.generic_string();
            std::ofstream file2(zp_file, std::ios::binary);
            // Get zp tensor
            auto zp = node_arg_get_const_data_as_u8s(*graph, *zp_node.node_arg);
            file2.write(reinterpret_cast<const char*>(zp.data()),
                        zp.size() * sizeof(int8_t));
            file2.close();
          }

          // if bias Add is present after MatMulNbits, get the bias, store it in
          // a file and keep the filename in attributes
          std::string bias_file;
          if (is_bais_matmul) {

            auto bias_node_arg =
                VAIP_ORT_API(graph_get_node_arg)(*graph, bias_name);
            bias_name.erase(std::remove_if(bias_name.begin(), bias_name.end(),
                                           [&valid_chars](char c) {
                                             return valid_chars.find(c) ==
                                                    std::string::npos;
                                           }),
                            bias_name.end());
            std::replace(bias_name.begin(), bias_name.end(), ' ', '_');
            std::string bias_bin = bias_name + ".bin";
            auto bpath = log_dir / bias_bin;
            bias_file = bpath.u8string();

            std::ofstream bfile(bias_file, std::ios::binary);
            auto biasdata =
                node_arg_get_const_data_as_floats(*graph, *bias_node_arg);
            bfile.write(reinterpret_cast<const char*>(biasdata.data()),
                        biasdata.size() * sizeof(float));
            bfile.close();
          }
          std::string ip_name = node_arg_get_name(*p_input_node.node_arg);
          std::string wts_name = node_arg_get_name(*input_w_node.node_arg);
          std::string scl_name = node_arg_get_name(*scales_node.node_arg);
          std::string zpoint_name;
          if (is_zp_present)
            zpoint_name = node_arg_get_name(*zp_node.node_arg);
          std::vector<std::string> inputs, outputs;

          if (!is_bais_matmul) {
            if (is_zp_present) {
              inputs = {ip_name, wts_name, scl_name, zpoint_name};
            } else {
              inputs = {ip_name, wts_name, scl_name};
            }
            std::string op_name = node_arg_get_name(*cast_out_node.node_arg);
            outputs = {op_name};
          } else {
            if (is_zp_present) {
              inputs = {ip_name, wts_name, scl_name, zpoint_name, bias_name};
            } else {
              inputs = {ip_name, wts_name, scl_name, bias_name};
            }
            // std::string op_name = node_arg_get_name(*cast_out_node.node_arg);
            outputs = {output_node_name};
          }

          auto constant_initializers = std::vector<std::string>{};
          auto [meta_def, err] =
              self->try_fuse(*graph, "matmul_nbits_" + name, inputs, outputs,
                             constant_initializers, "MATMULNBITS");
          if (meta_def == nullptr) {
            LOG(FATAL) << "Cannot fuse matmul_nbits:  " << err.comments;
          }
          auto& generic_param = *meta_def->mutable_generic_param();

          generic_param["cnt"] = std::to_string(cnt);
          generic_param["node_name"] = "MATMUL_NBITS_" + name;
          generic_param["wts_file"] = wts_file;
          generic_param["scl_file"] = scl_file;
          generic_param["shape_list_file"] = shape_bin_file;
          if (is_zp_present) {
            generic_param["zp_file"] = zp_file;
          }
          if (is_bais_matmul) {
            generic_param["bias_file"] = bias_file;
          }
          generic_param["K"] = std::to_string(m_k);
          generic_param["N"] = std::to_string(m_n);
          generic_param["bits"] = std::to_string(bits);
          generic_param["block_size"] = std::to_string(block_size);

          [[maybe_unused]] auto& fused_node =
              self->fuse(*graph, std::move(*meta_def));
          cnt++;
          return true;
        });
  }
  void process(IPass& self, Graph& graph) { create_rule(&self)->apply(&graph); }

public:
  IPass& self_;
  const std::filesystem::path& log_dir_;
};
DEFINE_VAIP_PASS(MatMulNBits, vaip_pass_matmul_nbits)
