/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */
// #include "./get_merged_attributes.hpp"

// #include "qmha/qmha_processor.hpp"
#include "vaip/dd/coeffs.hpp"
#include "vaip/dd/dd_utils.hpp"
// #include "vaip/pattern_zoo.hpp"
#include "vaip/pattern_zoo.hpp"
#include "vaip/vaip.hpp"
#include "vitis/ai/env_config.hpp"
#include <glog/logging.h>

DEF_ENV_PARAM(DEBUG_DD_MERGE_QL2NORM, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_DD_MERGE_QL2NORM) >= n)

namespace {
using namespace vaip_core;

struct MergeQLPnorm {
  MergeQLPnorm(IPass& self) : self_{self} {}
  std::unique_ptr<Rule> create_rule(IPass* self) {

    auto dq2 = vaip::pattern_zoo::get_pattern("m_lpnorm");
    CHECK(dq2 != nullptr) << "Pattern returned is null";

    return Rule::create_rule(
        dq2, [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
          auto attr_nodes = vaip::dd::get_node_names(graph, binder);
          auto a_node = binder["a"];
          auto as_node = binder["a_s"];
          auto az_node = binder["a_z"];
          auto a_sc =
              node_arg_get_const_data_as_float(*graph, *as_node.node_arg);
          auto a_zp = vaip::dd::get_zp_from_node(*graph, *az_node.node_arg);

          auto dq2_node = binder["dq2"];
          auto dq2_s_node = binder["dq2_s"];
          auto dq2_z_node = binder["dq2_z"];
          auto dq2_sc =
              node_arg_get_const_data_as_float(*graph, *dq2_s_node.node_arg);
          auto dq2_zp =
              vaip::dd::get_zp_from_node(*graph, *dq2_z_node.node_arg);

          std::vector<int32_t> qdq_coeffs(16, 0);

          // 0,1,2, - input_dequant -enable
          // 3,4,5, - output_quant -disable
          qdq_coeffs[0] = a_zp;
          qdq_coeffs[1] = vaip::dd::qmatmulcalc::float_to_bfloat16(a_sc);
          qdq_coeffs[2] = 1; // in_dequant_enable
          qdq_coeffs[3] = dq2_zp;
          qdq_coeffs[4] =
              vaip::dd::qmatmulcalc::float_to_bfloat16(1.0f / dq2_sc);
          qdq_coeffs[5] = 0; // out_quant_enable
          auto node_name = node_arg_get_name(*dq2_node.node_arg);

          std::string qdq_name = std::string(node_name + "_qdq_");
          auto& qdq_arg = vaip::dd::insert_named_tensor_in_graph<int32_t>(
              graph, qdq_name, qdq_coeffs,
              std::vector({(int64_t)qdq_coeffs.size()}));

          std::vector<std::string> in_dtypes = {
              vaip::dd::nodearg_dtype_to_string(*a_node.node_arg), "int32"};
          std::vector<std::string> out_dtypes = {
              "bfloat16"}; //{vaip::dd::nodearg_dtype_to_string(*ni_output.node_arg)};

          NodeBuilder(*graph, self_)
              .set_op_type("QL2norm", "com.xilinx")
              .set_input_node_args({a_node.node_arg, &qdq_arg})
              .clone_attrs(*a_node.node)
              .clone_data_type(*dq2_node.node_arg)
              .clone_shape(*dq2_node.node_arg)
              .set_anchor_point1(*dq2_node.node_arg)
              .add("in_dtypes", in_dtypes)
              .add("out_dtypes", out_dtypes)
              .build();

          return true;
        });
  }
  void process(IPass& self, Graph& graph) { create_rule(&self)->apply(&graph); }

public:
  IPass& self_;
};
} // namespace

DEFINE_VAIP_PASS(MergeQLPnorm, vaip_pass_dd_merge_qlpnorm)
