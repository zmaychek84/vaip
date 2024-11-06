/*
 *     The Xilinx Vitis AI Vaip in this distribution are provided under the
 * following free and permissive binary-only license, but are not provided in
 * source code form.  While the following free and permissive license is similar
 * to the BSD open source license, it is NOT the BSD open source license nor
 * other OSI-approved open source license.
 *
 *      Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc.
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
#include "dtype_util.h"
#include "vaip/vaip.hpp"
#include "vitis/ai/env_config.hpp"
#include <algorithm>
#include <glog/logging.h>
#include <utility>
DEF_ENV_PARAM(DEBUG_DD_PATTERN, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_DD_PATTERN) >= n)

namespace {
using namespace vaip_core;

struct Dd_merge_dtype {
  Dd_merge_dtype(IPass& self) : self_{self} {}
  std::vector<std::string>
  get_new_in_dtype_attributes(vaip::dtype_util::NodeAttrContext& ctx) {
    auto node = ctx.node;
    auto precision = ctx.precision;
    auto node_op = VAIP_ORT_API(node_op_type)(*node);
    std::string attr_name = "in_dtypes";
    auto& attrs = node_get_attributes_ref(*node);
    std::vector<std::string> ret;
    if (node_op == "QMatMulAddGelu" || node_op == "QMatMulAdd" ||
        node_op == "QMatMul") {
      auto attr_proto = node_attributes_get(attrs, attr_name);
      auto strs_value = VAIP_ORT_API(attr_proto_get_strings)(*attr_proto);

      if (precision == "a16w8")
        strs_value.at(0) = "uint16";
      else
        strs_value.at(0) = "uint8";
      return strs_value;
    } else if (node_op == "QEltWiseAdd") {
      auto attr_proto = node_attributes_get(attrs, attr_name);
      auto strs_value = VAIP_ORT_API(attr_proto_get_strings)(*attr_proto);

      if (precision == "a16w8") {
        // Only override if the precision is set to uint8
        if (!(strs_value.at(0) == "bfloat16"))
          strs_value.at(0) = "uint16";
        if (!(strs_value.at(1) == "bfloat16"))
          strs_value.at(1) = "uint16";
        return strs_value;
      }

    } else if (node_op == "QMHAGRPB") {
      auto attr_proto = node_attributes_get(attrs, attr_name);
      auto strs_value = VAIP_ORT_API(attr_proto_get_strings)(*attr_proto);

      if (precision == "a16w8") {
        strs_value.at(0) = "uint16";
        strs_value.at(1) = "uint16";
        strs_value.at(2) = "uint16";

      } else {
        strs_value.at(0) = "uint8";
        strs_value.at(1) = "uint8";
        strs_value.at(2) = "uint8";
      }
      return strs_value;
    }
    return ret;
  }

  std::vector<std::string>
  get_new_out_dtype_attributes(vaip::dtype_util::NodeAttrContext& ctx) {
    auto node = ctx.node;
    auto precision = ctx.precision;
    auto node_op = VAIP_ORT_API(node_op_type)(*node);
    std::string attr_name = "out_dtypes";
    std::vector<std::string> ret;
    if (node_op == "QMHAGRPB" || node_op == "QLayerNorm" ||
        node_op == "QMatMulAddGelu" || node_op == "QMatMulAdd" ||
        node_op == "QMatMul") {
      auto& attrs = node_get_attributes_ref(*node);
      auto attr_proto = node_attributes_get(attrs, attr_name);
      auto strs_value = VAIP_ORT_API(attr_proto_get_strings)(*attr_proto);
      if (precision == "a16w8")
        strs_value.at(0) = "uint16";
      else
        strs_value.at(0) = "uint8";
      return strs_value;
    }
    return ret;
  }
  bool update_node_attributes(vaip::dtype_util::NodeAttrContext& ctx) {
    auto node = ctx.node;
    std::map<std::string, std::vector<std::string>> m;
    m["in_dtypes"] = get_new_in_dtype_attributes(ctx);
    m["out_dtypes"] = get_new_out_dtype_attributes(ctx);
    bool replaced = false;
    for (const auto& kv : m) {
      if (kv.second.size()) {
        auto nab = NodeAttributesBuilder();
        nab.add(kv.first, kv.second);
        auto x = const_cast<Node*>(node);
        nab.merge_into(*x);
        replaced = true;
      }
    }
    return replaced;
  }

  // apply the rule
  void process(IPass& self, Graph& graph) {
    MY_LOG(1) << self_.get_pass_proto().name() << "["
              << self_.get_pass_proto().plugin() << "] start processing graph";
    // create_rule(&self)->apply(&graph);
    MY_LOG(1) << self.get_context()
                     ->xclbin_path_to_cache_files(std::filesystem::path(
                         self_.get_pass_proto().pass_dd_param().xclbin()))
                     .string();
    std::string precision = "a16w8"; // TODO Remove this Hardcoding
    for (const auto node_idx : graph_get_node_in_topoligical_order(graph)) {
      auto node = VAIP_ORT_API(graph_get_node)(graph, node_idx);
      auto node_ctx = vaip::dtype_util::build_context(graph, node, precision);

      if (update_node_attributes(node_ctx)) {
        MY_LOG(1) << "Changed in_dtype attribute";
      }
    }
    MY_LOG(1) << self_.get_pass_proto().name() << "["
              << self_.get_pass_proto().plugin() << "] finish processing graph";
  }

  IPass& self_;
};
} // namespace

DEFINE_VAIP_PASS(Dd_merge_dtype, vaip_pass_dd_merge_dtype)
