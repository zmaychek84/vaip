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

#ifdef __GNUC__
#  pragma GCC diagnostic pop
#endif

#include "vaip/vaip.hpp"
#include "vitis/ai/env_config.hpp"

DEF_ENV_PARAM(DEBUG_DD_MERGE_MLADFELWMUL, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_DD_MERGE_MLADFELWMUL) >= n)

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
    { "name": "vaip_pass_dd_merge_mladfelwmul",
       "plugin": "vaip-pass_dd_merge_mladfelwmul",
       "disabled": false
    }
*/
namespace {
using namespace vaip_core;
struct Dd_merge_mladfelwmul {
  Dd_merge_mladfelwmul(IPass& self) : self_{self} {}
  std::unique_ptr<Rule> create_rule(IPass* self) {
    auto builder = PatternBuilder();
    auto input_0 = builder.wildcard();
    auto scale_0 = builder.constant();
    auto zp_0 = builder.constant();
    auto dq_0 = builder.node2("com.microsoft:DequantizeLinear",
                              {input_0, scale_0, zp_0});
    auto input_1 = builder.wildcard();
    auto scale_1 = builder.constant();
    auto zp_1 = builder.constant();
    auto dq_1 = builder.node2("com.microsoft:DequantizeLinear",
                              {input_1, scale_1, zp_1});
    auto mul = builder.node2("Mul", {dq_0, dq_1});
    auto scale_2 = builder.constant();
    auto zp_2 = builder.constant();
    auto q =
        builder.node2("com.microsoft:QuantizeLinear", {mul, scale_2, zp_2});

    return Rule::create_rule(
        q, [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
          auto in0_node = binder[input_0->get_id()];
          auto scale0_node = binder[scale_0->get_id()];
          auto zp0_node = binder[zp_0->get_id()];
          auto in1_node = binder[input_1->get_id()];
          auto scale1_node = binder[scale_1->get_id()];
          auto zp1_node = binder[zp_1->get_id()];
          auto scale2_node = binder[scale_2->get_id()];
          auto zp2_node = binder[zp_2->get_id()];
          auto out_node = binder[q->get_id()];

          std::vector<std::string> ns = vaip::dd::get_node_names(graph, binder);
          MY_LOG(1) << "found match at " << ns.front();

          NodeInput in_ifm, in_const, ifm_scale_n, ifm_zp_n, const_scale_n,
              const_zp_n;
          if (node_arg_is_constant(*graph, *in0_node.node_arg)) {
            in_ifm = in1_node;
            ifm_scale_n = scale1_node;
            ifm_zp_n = zp1_node;
            in_const = in0_node;
            const_scale_n = scale0_node;
            const_zp_n = zp0_node;
          } else {
            in_ifm = in0_node;
            ifm_scale_n = scale0_node;
            ifm_zp_n = zp0_node;
            in_const = in1_node;
            const_scale_n = scale1_node;
            const_zp_n = zp1_node;
          }
          auto ifm_scale =
              node_arg_get_const_data_as_float(*graph, *ifm_scale_n.node_arg);
          auto ifm_zp =
              node_arg_get_const_data_as_u16(*graph, *ifm_zp_n.node_arg);
          auto const_scale =
              node_arg_get_const_data_as_float(*graph, *const_scale_n.node_arg);
          auto const_zp =
              node_arg_get_const_data_as_u16(*graph, *const_zp_n.node_arg);
          auto ofm_scale =
              node_arg_get_const_data_as_float(*graph, *scale2_node.node_arg);
          auto ofm_zp =
              node_arg_get_const_data_as_u16(*graph, *zp2_node.node_arg);
          auto ifm_shape = node_arg_get_shape_i64(*in_ifm.node_arg);
          auto tensor_sz =
              std::accumulate(ifm_shape->begin(), ifm_shape->end(), int64_t(1),
                              std::multiplies<int64_t>());

          // get const data, if not scalar need to change
          auto const_data =
              node_arg_get_const_data_as_u16(*graph, *in_const.node_arg);
          auto const_nodearg_name = node_arg_get_name(*in_const.node_arg);
          std::vector<uint16_t> const_as_vec{const_data};
          auto& input_const_arg = vaip::dd::insert_named_tensor_in_graph(
              graph, const_nodearg_name + "_to_arr_", const_as_vec,
              std::vector({(int64_t)const_as_vec.size()}));

          auto qdq_param = vaip::dd::qmatmulcalc::mladfelwmul_qdq_param_gen(
              ifm_scale, const_scale, ofm_scale, ifm_zp, const_zp, ofm_zp,
              tensor_sz);
          auto node_name = node_arg_get_name(*out_node.node_arg);
          auto& input_qdq_arg = vaip::dd::insert_named_tensor_in_graph(
              graph, node_name + "_qdq_", qdq_param,
              std::vector({(int64_t)qdq_param.size()}));

          std::vector<std::string> input_types{"uint16", "uint16", "uint8"};
          std::vector<std::string> output_types{"uint16"};

          NodeBuilder(*graph, *self)
              .set_input_node_args(
                  {in_ifm.node_arg, &input_const_arg, &input_qdq_arg})
              .set_op_type("Mladfelwmul", "com.xilinx")
              .add("nodes", ns)
              .add("in_dtypes", input_types)
              .add("out_dtypes", output_types)
              .set_anchor_point1(*out_node.node)
              .build();
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

DEFINE_VAIP_PASS(Dd_merge_mladfelwmul, vaip_pass_dd_merge_mladfelwmul)
