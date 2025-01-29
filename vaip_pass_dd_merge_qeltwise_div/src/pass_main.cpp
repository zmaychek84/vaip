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
#include "vaip/pattern_zoo.hpp"
#include "vaip/vaip.hpp"
#include "vitis/ai/env_config.hpp"
#include <glog/logging.h>

DEF_ENV_PARAM(DEBUG_DD_PATTERN, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_DD_PATTERN) >= n)
using namespace vaip_core;

struct MergeQEltWiseDiv {
  MergeQEltWiseDiv(IPass& self) : self_{self} {}

  std::unique_ptr<Rule> create_rule(IPass* self) {

    auto com_microsoft_QuantizeLinear_0 =
        vaip::pattern_zoo::get_pattern("m_qelwediv");
    CHECK(com_microsoft_QuantizeLinear_0 != nullptr)
        << "Pattern returned is null";

    return Rule::create_rule(
        com_microsoft_QuantizeLinear_0,
        [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
          auto attr_nodes = vaip::dd::get_node_names(graph, binder);

          auto a_node = binder["input_0"];
          auto as_node = binder["constant_0"];
          auto az_node = binder["constant_1"];
          auto a_sc =
              node_arg_get_const_data_as_float(*graph, *as_node.node_arg);
          auto a_zp = vaip::dd::get_zp_from_node(*graph, *az_node.node_arg);
          auto a_shape = node_arg_get_shape_i64(*a_node.node_arg);

          auto b_node = binder["input_1"];
          auto bs_node = binder["constant_2"];
          auto bz_node = binder["constant_3"];
          auto b_shape = node_arg_get_shape_i64(*b_node.node_arg);
          auto b_sc =
              node_arg_get_const_data_as_float(*graph, *bs_node.node_arg);
          auto b_zp = vaip::dd::get_zp_from_node(*graph, *bz_node.node_arg);

          auto out_node = binder["com_microsoft_QuantizeLinear_0"];
          auto outs_node = binder["constant_4"];
          auto outz_node = binder["constant_5"];
          auto out_sc =
              node_arg_get_const_data_as_float(*graph, *outs_node.node_arg);
          auto out_zp = vaip::dd::get_zp_from_node(*graph, *outz_node.node_arg);
          auto out_shape = node_arg_get_shape_i64(*out_node.node_arg);
          auto node_name = node_arg_get_name(*out_node.node_arg);

          // CHECK Data Type
          // auto out_dtype = node_arg_get_element_type(*out_node.node_arg);
          std::vector<float> input_q_params = {a_sc, float(a_zp), b_sc,
                                               float(b_zp)};
          std::vector<float> output_q_params = {out_sc, float(out_zp)};
          std::vector<std::string> input_types{"bfloat16", "bfloat16", "int32"};
          std::vector<std::string> output_types{"uint16"};
          auto [final_out_scale, final_out_zp] =
              vaip::dd::qmatmulcalc::calc_lrn_coeff(1 / out_sc, out_zp);

          std::vector<int32_t> qdq_params(16, 0);

          int32_t is_matC_uint16 = 1;

          qdq_params[0] = final_out_scale;
          qdq_params[1] = final_out_zp;
          qdq_params[2] = is_matC_uint16;

          std::string qdq_name = std::string(node_name + "_qdq_");
          auto& qdq_arg = vaip::dd::insert_named_tensor_in_graph<int32_t>(
              graph, qdq_name, qdq_params,
              std::vector({(int64_t)qdq_params.size()}));

          auto qeltwise_div = NodeBuilder(*graph, *self);
          qeltwise_div.set_input_node_args(
              {a_node.node_arg, b_node.node_arg, &qdq_arg});
          qeltwise_div.set_op_type("QEltWiseDiv", "com.xilinx");
          qeltwise_div.set_anchor_point1(*out_node.node);
          qeltwise_div.add("nodes", attr_nodes);
          qeltwise_div.add("input1_shape", *(a_shape.get()));
          qeltwise_div.add("input2_shape", *(b_shape.get()));
          qeltwise_div.add("input_q_params", input_q_params);
          qeltwise_div.add("output_q_params", output_q_params);
          qeltwise_div.add("in_dtypes", input_types);
          qeltwise_div.add("out_dtypes", output_types);
          qeltwise_div.add("orig_output_shape", *(out_shape.get()));
          qeltwise_div.build();
          return true;
        });
  }

  void process(IPass& self, Graph& graph) { create_rule(&self)->apply(&graph); }

public:
  IPass& self_;
};

DEFINE_VAIP_PASS(MergeQEltWiseDiv, vaip_pass_dd_merge_qeltwise_div)
