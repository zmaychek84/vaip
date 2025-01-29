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
#include "vaip/vaip.hpp"
#include "vitis/ai/env_config.hpp"
#include <functional>
#include <glog/logging.h>
#include <numeric>

DEF_ENV_PARAM(DEBUG_DD_MERGE_QSILU, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_DD_MERGE_QSILU) >= n)

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
    { "name": "vaip_pass_dd_merge_qsilu",
       "plugin": "vaip-pass_dd_merge_qsilu",
       "disabled": false
    }
*/
namespace {
using namespace vaip_core;
struct QSilu {
  QSilu(IPass& self) : self_{self} {}
  static std::vector<std::string> change_inputs(const NodeArg& a,
                                                const NodeArg& b) {
    //     std::cout<<"CHANGE INPUTS\n";
    std::vector<std::string> dtypes;
    // Add conditional code here :TODO
    //     dtypes.emplace_back(vaip::dd::nodearg_dtype_to_string(a));
    dtypes.emplace_back("bfloat16");
    dtypes.emplace_back(vaip::dd::nodearg_dtype_to_string(b));
    return dtypes;
  }

  static std::vector<std::string> change_outputs(const NodeArg& a) {
    // std::cout<<"CHANGE OUTPUTS\n";
    std::vector<std::string> dtypes;
    // Add conditional code here (Below may only work for mdsqr)
    dtypes.emplace_back(vaip::dd::nodearg_dtype_to_string(a));
    return dtypes;
  }

  std::unique_ptr<Rule> create_rule(IPass* self) {

    auto q2 = vaip::pattern_zoo::get_pattern("m_qsilu_0");
    return Rule::create_rule(
        q2, [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
          // auto input = binder[input_->get_id()];
          std::vector<std::string> ns = vaip::dd::get_node_names(graph, binder);
          auto a_node = binder["a"];
          auto as_node = binder["a_s"];
          auto az_node = binder["a_z"];
          auto a_sc =
              node_arg_get_const_data_as_float(*graph, *as_node.node_arg);
          auto a_zp = vaip::dd::get_zp_from_node(*graph, *az_node.node_arg);
          auto a_shape = node_arg_get_shape_i64(*a_node.node_arg);

          //     auto b_node = binder[b->get_id()];
          //     auto bs_node = binder[b_s->get_id()];
          //     auto bz_node = binder[b_z->get_id()];
          //     auto b_sc =
          //          node_arg_get_const_data_as_float(*graph,
          //          *bs_node.node_arg);
          //     auto b_zp = vaip::dd::get_zp_from_node(*graph,
          //     *bz_node.node_arg); auto b_shape =
          //     node_arg_get_shape_i64(*b_node.node_arg);

          // auto dq2_node = binder[dq2->get_id()];
          auto dq2s_node = binder["dq2_s"];
          auto dq2z_node = binder["dq2_z"];
          auto dq2_sc =
              node_arg_get_const_data_as_float(*graph, *dq2s_node.node_arg);
          auto dq2_zp = vaip::dd::get_zp_from_node(*graph, *dq2z_node.node_arg);

          // OUTPUT
          auto q2_node = binder["q2"];
          auto q2s_node = binder["q2_s"];
          auto q2z_node = binder["q2_z"];
          auto q2_sc =
              node_arg_get_const_data_as_float(*graph, *q2s_node.node_arg);
          auto q2_zp = vaip::dd::get_zp_from_node(*graph, *q2z_node.node_arg);
          auto q2_shape = node_arg_get_shape_i64(*q2_node.node_arg);
          // CHECK Data Type
          auto out_dtype = node_arg_get_element_type(*q2_node.node_arg);
          auto node_name = node_arg_get_name(*q2_node.node_arg);
          MY_LOG(1) << "QSilu" << out_dtype;

          // PARAMS
          std::vector<float> input_q_params;
          input_q_params.push_back(a_sc);
          input_q_params.push_back(float(a_zp));
          input_q_params.push_back(dq2_sc);
          input_q_params.push_back(float(dq2_zp));

          std::vector<float> output_q_params;
          output_q_params.push_back(q2_sc);
          output_q_params.push_back(float(q2_zp));

          std::vector<int16_t> lrn_qdq_tensor(16, 0);
          lrn_qdq_tensor[0] = (int16_t)q2_zp;
          lrn_qdq_tensor[1] =
              (int16_t)vaip::dd::qmatmulcalc::float_to_bfloat16(1 / q2_sc);
          lrn_qdq_tensor[2] = 1;

          std::string qdq_name = std::string(node_name + "_qdq_");
          auto& lrn_qdq_arg = vaip::dd::insert_named_tensor_in_graph<int16_t>(
              graph, qdq_name, lrn_qdq_tensor,
              std::vector({(int64_t)lrn_qdq_tensor.size()}));

          NodeBuilder(*graph, *self)
              .set_input_node_args({a_node.node_arg, &lrn_qdq_arg})
              .set_op_type("QSilu", "com.xilinx")
              .clone_attrs(*q2_node.node)
              .add("nodes", ns)
              .set_anchor_point1(*q2_node.node)
              .add("in_dtypes", change_inputs(*a_node.node_arg, lrn_qdq_arg))
              .add("out_dtypes", change_outputs(*q2_node.node_arg))
              .add("input_q_params", input_q_params)
              .add("output_q_params", output_q_params)
              .build();
          return true; // return true if graph is modified.
        });
  }
  // apply the rule
  void process(IPass& self, Graph& graph) {
    // MY_LOG(1) << self_.get_pass_proto().name() << "[" <<
    // self_.get_pass_proto().plugin() << "] start processing graph";
    create_rule(&self)->apply(&graph);
    // MY_LOG(1) << self_.get_pass_proto().name() << "[" <<
    // self_.get_pass_proto().plugin() << "] finish processing graph";
  }

  IPass& self_;
};
} // namespace

DEFINE_VAIP_PASS(QSilu, vaip_pass_dd_merge_qsilu)
