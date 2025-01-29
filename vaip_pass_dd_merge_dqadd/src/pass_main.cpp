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
#include "vaip/pattern_zoo.hpp"

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

struct MergeDQAdd {
  MergeDQAdd(IPass& self) : self_{self} {}

  static std::vector<std::string> change_inputs(const NodeArg& a,
                                                const NodeArg& b) {
    std::vector<std::string> dtypes;
    dtypes.emplace_back(vaip::dd::nodearg_dtype_to_string(a));
    dtypes.emplace_back(vaip::dd::nodearg_dtype_to_string(b));
    dtypes.emplace_back("int32");
    return dtypes;
  }

  static std::vector<std::string> change_outputs(const NodeArg& a) {
    std::vector<std::string> dtypes;
    // Add conditional code here (Below may only work for mdsqr)
    dtypes.emplace_back("bfloat16");
    return dtypes;
  }

  std::unique_ptr<Rule> create_rule(IPass* self) {

    auto Add_0 = vaip::pattern_zoo::get_pattern("m_dqadd");
    CHECK(Add_0 != nullptr) << "Pattern returned is null";

    return Rule::create_rule(
        Add_0, [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
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

          auto add_node = binder["Add_0"];
          auto add_shape = node_arg_get_shape_i64(*add_node.node_arg);
          auto node_name = node_arg_get_name(*add_node.node_arg);
          // CHECK Data Type
          // auto out_dtype = node_arg_get_element_type(*add_node.node_arg);
          std::vector<float> input_q_params = {a_sc, float(a_zp), b_sc,
                                               float(b_zp)};

          std::vector<int32_t> dq_add_params(16, 0);
          int32_t cmat_uint16 = 0;
          int32_t amat_uint16 = 1;
          dq_add_params[0] = (int32_t)vaip::dd::qmatmulcalc::float_to_bfloat16(
              input_q_params[0]);
          dq_add_params[1] = (int32_t)(input_q_params[1]);
          dq_add_params[2] = (int32_t)vaip::dd::qmatmulcalc::float_to_bfloat16(
              input_q_params[2]);
          dq_add_params[3] = (int32_t)(input_q_params[3]);
          dq_add_params[6] = amat_uint16;
          dq_add_params[7] = cmat_uint16;

          std::string qdq_name = std::string(node_name + "_qdq_");
          auto& qdq_arg = vaip::dd::insert_named_tensor_in_graph<int32_t>(
              graph, qdq_name, dq_add_params,
              std::vector({(int64_t)dq_add_params.size()}));

          auto dqadd = NodeBuilder(*graph, *self);
          dqadd.set_input_node_args(
              {a_node.node_arg, b_node.node_arg, &qdq_arg});
          dqadd.set_op_type("DQAdd", "com.xilinx");
          dqadd.set_anchor_point1(*add_node.node);
          dqadd.add("nodes", attr_nodes);
          dqadd.add("in_dtypes",
                    change_inputs(*a_node.node_arg, *b_node.node_arg));
          dqadd.add("out_dtypes", change_outputs(*add_node.node_arg));
          dqadd.add("input1_shape", *(a_shape.get()));
          dqadd.add("input2_shape", *(b_shape.get()));
          dqadd.add("input_q_params", input_q_params);
          dqadd.add("orig_output_shape", *(add_shape.get()));
          dqadd.build();
          return true;
        });
  }

  void process(IPass& self, Graph& graph) { create_rule(&self)->apply(&graph); }

public:
  IPass& self_;
};

DEFINE_VAIP_PASS(MergeDQAdd, vaip_pass_dd_merge_dqadd)
