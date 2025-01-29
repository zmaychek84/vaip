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
DEF_ENV_PARAM(DEBUG_DD_MERGE_QGELU, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_DD_MERGE_QGELU) >= n)

// add the following line in your vaip_config.json
/*
    { "name": "vaip_pass_dd_merge_qgelu",
       "plugin": "vaip-pass_dd_merge_qgelu",
       "disabled": false
    }
*/
namespace {
using namespace vaip_core;
struct QGelu {
  QGelu(IPass& self) : self_{self} {}

  static std::vector<std::string> change_inputs(const NodeArg& a,
                                                const NodeArg& b) {

    std::vector<std::string> dtypes;
    // Add conditional code here :TODO
    dtypes.emplace_back(vaip::dd::nodearg_dtype_to_string(a));
    // dtypes.emplace_back("bfloat16");
    dtypes.emplace_back(vaip::dd::nodearg_dtype_to_string(b));
    return dtypes;
  }
  static std::vector<std::string> change_nodetypes(const NodeArg& a) {
    std::vector<std::string> dtypes;
    // Add conditional code here (Below may only work for mdsqr)
    dtypes.emplace_back(vaip::dd::nodearg_dtype_to_string(a));
    return dtypes;
  }
  static std::vector<std::string> change_outputs(const NodeArg& a) {
    std::vector<std::string> dtypes;
    // Add conditional code here (Below may only work for mdsqr)
    dtypes.emplace_back("bfloat16");
    return dtypes;
  }
  static std::pair<float, uint16_t> get_scale_zp_with_ancestor_check(
      onnxruntime::Graph* graph, binder_t& binder, vaip_core::NodeInput& a,
      vaip_core::NodeInput& as_node, vaip_core::NodeInput& az_node) {
    auto a_sc = node_arg_get_const_data_as_float(*graph, *as_node.node_arg);
    auto a_zp = vaip::dd::get_zp_from_node(*graph, *az_node.node_arg);
    auto parent_op_type = VAIP_ORT_API(node_op_type)(*a.node);
    MY_LOG(1) << " " << a_sc << " " << a_zp;
    if (parent_op_type == "QSlice") {
      // Pick Q params from here
      MY_LOG(1) << "Have to get the q_params from here instead";
      auto& attrs = node_get_attributes_ref(*a.node);
      auto attr_proto = node_attributes_get(attrs, "q_scale");
      a_sc = VAIP_ORT_API(attr_proto_get_float)(*attr_proto);
      attr_proto = node_attributes_get(attrs, "q_zp");
      a_zp = (uint16_t)(VAIP_ORT_API(attr_proto_get_int)(*attr_proto));
      MY_LOG(1) << " " << a_sc << " " << a_zp;
    }
    MY_LOG(1) << "DONE";
    auto ret = std::make_pair(a_sc, a_zp);
    return ret;
  }
  std::unique_ptr<Rule> create_rule(IPass* self) {
    //    std::cout<<"Called create_rule\n";

    auto q = vaip::pattern_zoo::get_pattern("m_qgelu_0");
    return Rule::create_rule(
        q, [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
          std::vector<std::string> ns = vaip::dd::get_node_names(graph, binder);

          auto a_node = binder["a"];
          auto as_node = binder["a_s"];
          auto az_node = binder["a_z"];
          auto r = get_scale_zp_with_ancestor_check(graph, binder, a_node,
                                                    as_node, az_node);
          auto a_sc = r.first;
          auto a_zp = r.second;
          auto q_node = binder["q"];
          auto qs_node = binder["q_s"];
          auto qz_node = binder["q_z"];
          auto q_sc =
              node_arg_get_const_data_as_float(*graph, *qs_node.node_arg);
          auto q_zp = vaip::dd::get_zp_from_node(*graph, *qz_node.node_arg);
          auto q_shape = node_arg_get_shape_i64(*q_node.node_arg);
          auto node_dtype = node_arg_get_element_type(*q_node.node_arg);
          auto node_name = node_arg_get_name(*q_node.node_arg);
          //   std::cout<<"Got nodes from binder\n";
          MY_LOG(1) << "QGelu" << node_dtype;

          std::vector<float> input_q_params;
          input_q_params.push_back(a_sc);
          input_q_params.push_back(float(a_zp));

          std::vector<float> output_q_params;
          output_q_params.push_back(q_sc);
          output_q_params.push_back(float(q_zp));

          // Weight tensor gelu_qdq
          auto lrn_c1 = a_zp;
          auto lrn_c0 = vaip::dd::qmatmulcalc::float_to_bfloat16(a_sc);
          std::vector<uint16_t> gelu_qdq_tensor(16, 0);

          gelu_qdq_tensor[3] = lrn_c1;
          gelu_qdq_tensor[4] = lrn_c0;
          gelu_qdq_tensor[5] = 1; // Enable dequant at input

          std::string qdq_name = std::string(node_name + "gelu_qdq_");
          auto& gelu_qdq_arg = vaip::dd::insert_named_tensor_in_graph<uint16_t>(
              graph, qdq_name, gelu_qdq_tensor,
              std::vector({(int64_t)gelu_qdq_tensor.size()}));

          NodeBuilder(*graph, *self)
              .set_input_node_args({a_node.node_arg, &gelu_qdq_arg})
              .set_op_type("QGelu", "com.xilinx")
              .clone_attrs(*q_node.node)
              .add("nodes", ns)
              .set_anchor_point1(*q_node.node)
              .add("in_dtypes", change_inputs(*a_node.node_arg, gelu_qdq_arg))
              .add("Node_dtype", change_nodetypes(*q_node.node_arg))
              .add("out_dtypes", change_outputs(*q_node.node_arg))
              .add("input_q_params", input_q_params)
              .add("output_q_params", output_q_params)
              .add("orig_output_shape", *(q_shape.get()))
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

DEFINE_VAIP_PASS(QGelu, vaip_pass_dd_merge_qgelu)
