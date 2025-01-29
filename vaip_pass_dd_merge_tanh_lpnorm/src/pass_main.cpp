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
#include <glog/logging.h>

#include "vaip/pattern_zoo.hpp"

DEF_ENV_PARAM(DEBUG_DD_PATTERN, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_DD_PATTERN) >= n)
using namespace vaip_core;

struct Merge_tanh_lpnorm {
  Merge_tanh_lpnorm(IPass& self) : self_{self} {}

  static std::vector<std::string> change_inputs(const NodeArg& a,
                                                const NodeArg& b) {
    std::vector<std::string> dtypes;
    dtypes.emplace_back("uint16");
    dtypes.emplace_back("uint16");
    dtypes.emplace_back("int32");

    return dtypes;
  }

  static std::vector<std::string> change_outputs(const NodeArg& a) {
    std::vector<std::string> dtypes;
    dtypes.emplace_back("bfloat16");
    return dtypes;
  }

  std::unique_ptr<Rule> create_rule(IPass* self) {

    auto com_microsoft_DequantizeLinear_5 =
        vaip::pattern_zoo::get_pattern("m_tanh_lpnorm");
    CHECK(com_microsoft_DequantizeLinear_5 != nullptr)
        << "Pattern returned is null";

    return Rule::create_rule(
        com_microsoft_DequantizeLinear_5,
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

          auto out_node = binder["com_microsoft_DequantizeLinear_5"];

          auto out_shape = node_arg_get_shape_i64(*out_node.node_arg);
          auto output_shape = vaip::dd::shape_as_string(*(out_shape.get()));
          auto node_name = node_arg_get_name(*out_node.node_arg);

          std::vector<int32_t> qdq_params(16, 0);

          qdq_params[0] = a_zp;
          qdq_params[1] = vaip::dd::qmatmulcalc::float_to_bfloat16(a_sc);
          qdq_params[2] = b_zp;
          qdq_params[3] = vaip::dd::qmatmulcalc::float_to_bfloat16(b_sc);

          std::string qdq_name = std::string(node_name + "_qdq_");
          auto& qdq_arg = vaip::dd::insert_named_tensor_in_graph<int32_t>(
              graph, qdq_name, qdq_params,
              std::vector({(int64_t)qdq_params.size()}));

          auto tanh_lpnorm = NodeBuilder(*graph, *self);
          tanh_lpnorm.set_input_node_args(
              {a_node.node_arg, b_node.node_arg, &qdq_arg});
          tanh_lpnorm.set_op_type("Qtanh_lpnorm", "com.xilinx");
          tanh_lpnorm.set_anchor_point1(*out_node.node);
          tanh_lpnorm.add("nodes", attr_nodes);
          tanh_lpnorm.add("in_dtypes",
                          change_inputs(*a_node.node_arg, *b_node.node_arg));
          tanh_lpnorm.add("out_dtypes", change_outputs(*out_node.node_arg));
          tanh_lpnorm.add("input1_shape", *(a_shape.get()));
          tanh_lpnorm.add("input2_shape", *(b_shape.get()));
          tanh_lpnorm.add("orig_output_shape", output_shape);
          tanh_lpnorm.build();
          return true;
        });
  }

  void process(IPass& self, Graph& graph) { create_rule(&self)->apply(&graph); }

public:
  IPass& self_;
};

DEFINE_VAIP_PASS(Merge_tanh_lpnorm, vaip_pass_dd_merge_tanh_lpnorm)
