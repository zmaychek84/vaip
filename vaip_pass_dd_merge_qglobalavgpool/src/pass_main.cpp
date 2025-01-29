/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */
#include <glog/logging.h>

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

DEF_ENV_PARAM(DEBUG_DD_MERGE_QGLOBALAVGPOOL, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_DD_MERGE_QGLOBALAVGPOOL) >= n)

namespace {
using namespace vaip_core;
struct Dd_merge_qglobalavgpool {
  Dd_merge_qglobalavgpool(IPass& self) : self_{self} {}
  std::unique_ptr<Rule> create_rule(IPass* self) {
    auto com_microsoft_QuantizeLinear_3 =
        vaip::pattern_zoo::get_pattern("m_qglobalavgpool");
    CHECK(com_microsoft_QuantizeLinear_3 != nullptr)
        << "Pattern returned is null";
    return Rule::create_rule(
        com_microsoft_QuantizeLinear_3,
        [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
          auto in_node = binder["input_0"];
          auto pool_in_sc_node = binder["constant_0"];
          auto pool_in_zp_node = binder["constant_1"];
          auto pool_out_sc_node = binder["constant_5"];
          auto pool_out_zp_node = binder["constant_6"];
          auto pool_in_node = binder["com_microsoft_DequantizeLinear_2"];
          auto transpose_node = binder["Transpose_0"];
          auto flatten_node = binder["Flatten_0"];
          auto pool_node = binder["GlobalAveragePool_0"];
          auto out_node = binder["com_microsoft_QuantizeLinear_3"];

          std::vector<std::string> ns = vaip::dd::get_node_names(graph, binder);
          MY_LOG(1) << "found match at " << ns.front();

          auto in_shape = node_arg_get_shape_i64(*pool_in_node.node_arg);
          auto out_shape = node_arg_get_shape_i64(*pool_node.node_arg);

          auto pool_sc = node_arg_get_const_data_as_float(
              *graph, *pool_in_sc_node.node_arg);
          auto pool_zp =
              vaip::dd::get_zp_from_node(*graph, *pool_in_zp_node.node_arg);
          auto pool_out_sc = node_arg_get_const_data_as_float(
              *graph, *pool_out_sc_node.node_arg);
          auto pool_out_zp =
              vaip::dd::get_zp_from_node(*graph, *pool_out_zp_node.node_arg);
          auto zero_point =
              node_arg_get_const_data_as_u16(*graph, *pool_in_zp_node.node_arg);

          double pool_sc_re = 1.0 / (double)pool_sc;
          double pool_out_sc_re = 1.0 / (double)pool_out_sc;
          auto [offset, divFactor, divShift] =
              vaip::dd::qmatmulcalc::global_avg_pool_qdq(
                  pool_sc_re, (uint16_t)pool_zp, pool_out_sc_re,
                  (uint16_t)pool_out_zp);

          uint64_t mask = 0xFFFFFFFF;
          int32_t lsb = (int32_t)(mask & offset);
          mask <<= 32;
          int32_t msb = (int32_t)((mask & offset) >> 32);

          auto transpose_perm_span =
              node_get_attr_ints(*transpose_node.node, "perm");
          std::vector<int64_t> transpose_perm(transpose_perm_span.begin(),
                                              transpose_perm_span.end());

          auto flatten_axis = node_get_attr_int(*flatten_node.node, "axis");

          // hard code for m3uec, may need to change
          std::vector<std::string> input_types{"uint16", "int32"};
          std::vector<std::string> output_types{"uint16"};

          std::vector<int32_t> coeffs(32, 0);
          auto node_name = node_arg_get_name(*out_node.node_arg);
          std::string coeff_name = std::string(node_name + "_qdq_");
          auto& coeff_arg = vaip::dd::insert_named_tensor_in_graph<int32_t>(
              graph, coeff_name, coeffs, std::vector({(int64_t)coeffs.size()}));

          NodeBuilder(*graph, *self)
              .set_input_node_args({in_node.node_arg, &coeff_arg})
              .set_op_type("QGlobalAvgPool", "com.xilinx")
              .clone_attrs(*out_node.node)
              .add("nodes", ns)
              .add("input_shape", *in_shape)
              .add("output_shape", *out_shape)
              .add("zero_point", int64_t(zero_point))
              .add("in_dtypes", input_types)
              .add("out_dtypes", output_types)
              .add("transpose_perm", transpose_perm)
              .add("flatten_axis", flatten_axis)
              .add("offset_lsb", int64_t(lsb))
              .add("offset_msb", int64_t(msb))
              .add("div_factor", int64_t(divFactor))
              .add("div_shift", int64_t(divShift))
              .set_anchor_point1(*out_node.node)
              .build();
          return true;
        });
  }
  // apply the rule
  void process(IPass& self, Graph& graph) {
    MY_LOG(1) << "running merge_qglobalavgpool";
    create_rule(&self)->apply(&graph);
  }

  IPass& self_;
};
} // namespace

DEFINE_VAIP_PASS(Dd_merge_qglobalavgpool, vaip_pass_dd_merge_qglobalavgpool)
