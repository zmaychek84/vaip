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

DEF_ENV_PARAM(DEBUG_DD_MERGE_QSKIPADD, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_DD_MERGE_QSKIPADD) >= n)

namespace {
using namespace vaip_core;

std::tuple<bool, bool> check_add_in_parent(const Node* in_0, const Node* in_1) {
  if (in_0 && node_is_op(*in_0, "QEltWiseAdd", "com.xilinx")) {
    return {true, true};
  }
  if (in_1 && node_is_op(*in_1, "QEltWiseAdd", "com.xilinx")) {
    return {true, false};
  }
  return {false, false};
}

bool lrn_in_consumer(const Graph& g, const NodeArg* out) {
  auto consumers = graph_get_consumer_nodes(g, node_arg_get_name(*out));
  for (auto node : consumers) {
    if (node_is_op(*node, "QLayerNorm", "com.xilinx")) {
      return true;
    }
  }
  return false;
}

bool check_lrn_in_sibling(const Graph& g, const NodeArg* in_0,
                          const NodeArg* in_1) {
  return lrn_in_consumer(g, in_0) || lrn_in_consumer(g, in_1);
}

std::vector<std::string> change_inputs(const NodeArg& a, const NodeArg& b) {
  std::vector<std::string> dtypes;
  // Add conditional code here (Below may only work for mdsqr)
  auto a_dtype = node_arg_get_element_type(a);
  if (a_dtype == (int)ONNX_NAMESPACE::TensorProto_DataType_UINT16)
    dtypes.emplace_back("uint16");
  else
    dtypes.emplace_back("uint8");

  auto b_dtype = node_arg_get_element_type(b);
  if (b_dtype == (int)ONNX_NAMESPACE::TensorProto_DataType_UINT16)
    dtypes.emplace_back("uint16");
  else
    dtypes.emplace_back("uint8");
  dtypes.emplace_back("int32");
  return dtypes;
}

std::vector<std::string> change_outputs(const NodeArg& a) {
  std::vector<std::string> dtypes;
  // Add conditional code here (Below may only work for mdsqr)
  dtypes.emplace_back("bfloat16");
  return dtypes;
}

struct Dd_merge_qskipadd {
  Dd_merge_qskipadd(IPass& self) : self_{self} {}
  std::unique_ptr<Rule> create_rule(IPass* self) {

    auto com_microsoft_QuantizeLinear_11 =
        vaip::pattern_zoo::get_pattern("m_qskipadd_0");
    CHECK(com_microsoft_QuantizeLinear_11 != nullptr)
        << "Pattern returned is null";
    return Rule::create_rule(
        com_microsoft_QuantizeLinear_11,
        [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
          auto in_node_0 = binder["input_4"];
          auto in_0_scale_node = binder["constant_5"];
          auto in_0_zp_node = binder["constant_6"];
          auto in_0_scale = node_arg_get_const_data_as_float(
              *graph, *in_0_scale_node.node_arg);
          auto in_0_zp =
              node_arg_get_const_data_as_u16(*graph, *in_0_zp_node.node_arg);

          auto in_node_1 = binder["input_0"];
          auto in_1_scale_node = binder["constant_1"];
          auto in_1_zp_node = binder["constant_2"];
          auto in_1_scale = node_arg_get_const_data_as_float(
              *graph, *in_1_scale_node.node_arg);
          auto in_1_zp =
              node_arg_get_const_data_as_u16(*graph, *in_1_zp_node.node_arg);

          auto out_scale_node = binder["constant_9"];
          auto out_zp_node = binder["constant_10"];
          auto out_node = binder["com_microsoft_QuantizeLinear_11"];
          auto out_scale = node_arg_get_const_data_as_float(
              *graph, *out_scale_node.node_arg);
          auto out_zp =
              node_arg_get_const_data_as_u16(*graph, *out_zp_node.node_arg);
          // TODO: check_if_wts_add, for m3uec all matched add is used:
          // fuse = True, fuse_badd = False

          std::vector<std::string> ns = vaip::dd::get_node_names(graph, binder);
          MY_LOG(1) << "found match at " << ns.front();

          auto [add_in_parent, first_parent_is_add] =
              check_add_in_parent(in_node_0.node, in_node_1.node);
          auto lrn_in_sibling = check_lrn_in_sibling(*graph, in_node_0.node_arg,
                                                     in_node_1.node_arg);
          auto lrn_in_opt = lrn_in_consumer(*graph, out_node.node_arg);

          int32_t amat_uint16 = lrn_in_sibling ? 0 : 1;
          int32_t cmat_uint16 = lrn_in_opt ? 0 : 1;

          std::vector<float> input_q_params{in_0_scale, float(in_0_zp),
                                            in_1_scale, float(in_1_zp)};
          if (add_in_parent && !first_parent_is_add) {
            std::swap(input_q_params[0], input_q_params[2]);
            std::swap(input_q_params[1], input_q_params[3]);
          }
          std::vector<float> output_q_params{out_scale, float(out_zp)};

          std::vector<int32_t> elt_coeffs(16, 0);
          elt_coeffs[0] =
              vaip::dd::qmatmulcalc::float_to_bfloat16(input_q_params[0]);
          elt_coeffs[1] = int32_t(input_q_params[1]);
          elt_coeffs[2] =
              vaip::dd::qmatmulcalc::float_to_bfloat16(input_q_params[2]);
          elt_coeffs[3] = int32_t(input_q_params[3]);
          elt_coeffs[4] =
              vaip::dd::qmatmulcalc::float_to_bfloat16(1.0f / out_scale);
          elt_coeffs[5] = out_zp;
          elt_coeffs[6] = amat_uint16;
          elt_coeffs[7] = cmat_uint16;
          auto node_name = node_arg_get_name(*out_node.node_arg);
          std::string elt_coeff_name = std::string(node_name + "_qdq_");
          auto& elt_arg = vaip::dd::insert_named_tensor_in_graph<int32_t>(
              graph, elt_coeff_name, elt_coeffs,
              std::vector({(int64_t)elt_coeffs.size()}));

          std::vector<const NodeArg*> new_inputs{in_node_0.node_arg,
                                                 in_node_1.node_arg, &elt_arg};
          if (add_in_parent && !first_parent_is_add) {
            std::swap(new_inputs[0], new_inputs[1]);
          }

          NodeBuilder(*graph, *self)
              .set_input_node_args(new_inputs)
              .set_op_type("QEltWiseAdd", "com.xilinx")
              .add("nodes", ns)
              .add("amat_uint16", (int64_t)amat_uint16)
              .add("cmat_uint16", (int64_t)cmat_uint16)
              .add("input_q_params", input_q_params)
              .add("output_q_params", output_q_params)
              .add("in_dtypes",
                   change_inputs(*in_node_0.node_arg, *in_node_1.node_arg))
              .add("out_dtypes", change_outputs(*out_node.node_arg))
              .set_anchor_point1(*out_node.node)
              .build();
          return true;
        });
  }
  // apply the rule
  void process(IPass& self, Graph& graph) {
    MY_LOG(1) << "running merge_qskipadd";
    create_rule(&self)->apply(&graph);
  }

  IPass& self_;
};
} // namespace

DEFINE_VAIP_PASS(Dd_merge_qskipadd, vaip_pass_dd_merge_qskipadd)
