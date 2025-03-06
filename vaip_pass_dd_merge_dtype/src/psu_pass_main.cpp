/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */
#include "dtype_util.h"
#include "vaip/vaip.hpp"
#include "vitis/ai/env_config.hpp"
#include <algorithm>
#include <glog/logging.h>
#include <utility>

#include "vaip/dd/coeffs.hpp"
#include "vaip/dd/dd_utils.hpp"
DEF_ENV_PARAM(DEBUG_DD_PATTERN, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_DD_PATTERN) >= n)

namespace {
using namespace vaip_core;

struct EndPass_psu {
  EndPass_psu(IPass& self) : self_{self} {}

  void update_node_attributes(vaip::dtype_util::NodeAttrContext& ctx,
                              std::string design_param) {
    auto node = ctx.node;
    auto node_op = VAIP_ORT_API(node_op_type)(*node);
    auto nab = NodeAttributesBuilder();
    if (design_param == "4x4PSU")
      nab.add("design_param", "4x4PSU");
    else
      nab.add("design_param", "8x4PSU");
    auto x = const_cast<Node*>(node);
    nab.merge_into(*x);
  }

  void process(IPass& self, Graph& graph) {
    std::string precision = "a168"; // not used
    const auto& session_option = self.get_config_proto().provider_options();
    std::string design_param = "4x4PSU";

    /*
    - By default the design param for psu would be 4x4
    - If provider option : design_param is mentioned and the value is 8x4
    override the design_param value
    - Alternatively if "8x4PSU" arg is mentioned in the psu pass then we overrie
    the design_param value here
    */
    auto args = self.get_pass_proto().args();
    if (!args.empty()) {
      std::vector<std::string> arg;
      for (const auto& str : args) {
        arg.push_back(str);
      }
      if (arg[0] == "8x4PSU")
        design_param = arg[0];
    }
    if ((session_option.find("design_param") != session_option.end() &&
         session_option.find("design_param")->second == "8x4")) {
      design_param = "8x4PSU";
    }
    for (const auto node_idx : graph_get_node_in_topoligical_order(graph)) {
      auto node = VAIP_ORT_API(graph_get_node)(graph, node_idx);
      auto node_ctx = vaip::dtype_util::build_context(graph, node, precision);
      auto node_op = VAIP_ORT_API(node_op_type)(*node);
      update_node_attributes(node_ctx, design_param);
    }
  }

  IPass& self_;
};

} // namespace

DEFINE_VAIP_PASS(EndPass_psu, vaip_pass_dd_merge_EndPass_psu)
