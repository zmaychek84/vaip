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
#include <cmath>

#include <fstream>
#include <glog/logging.h>

DEF_ENV_PARAM(DEBUG_DD_PATTERN, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_DD_PATTERN) >= n)
using namespace vaip_core;

std::tuple<float, uint16_t, std::string>
get_concat_input_qparams_add_sibling(onnxruntime::Graph* graph,
                                     const NodeInput& in_node, float inp_sc,
                                     uint16_t inp_zp) {

  auto in_node_found = in_node.node;
  std::string concat_in_sibling = "false";
  if (in_node_found != nullptr) {
    auto op_type = VAIP_ORT_API(node_op_type)(*in_node_found);

    if (op_type == "IConv" || op_type == "QEltWiseAdd") {
      auto concat_attr = node_has_attr(*in_node_found, "concat_in_child");
      if (concat_attr &&
          node_get_attr_string(*in_node_found, "concat_in_child") == "true") {
        inp_sc = node_get_attr_float(*in_node_found, "output_scale");
        inp_zp = (uint16_t)(node_get_attr_float(*in_node_found, "output_zp"));
        concat_in_sibling = "true";
        MY_LOG(1) << "QEltWiseAdd  a has concat in silbling";
      }
    }
  }
  return std::make_tuple(inp_sc, inp_zp, concat_in_sibling);
}

std::tuple<float, uint16_t, std::string>
get_concat_output_qparams_add(onnxruntime::Graph* graph,
                              const NodeInput& out_node, float out_sc,
                              uint16_t out_zp) {
  // Concat pass would be completed before QEltWiseAdd pass, so we read the
  // attributes of Concat (consumer of eltwise add) and update the output
  // qparams
  std::string concat_in_child = "false";
  std::vector<const Node*> out_node_nextnodes =
      graph_get_consumer_nodes(*graph, node_arg_get_name(*out_node.node_arg));
  // multiple consumers
  for (auto consumer : out_node_nextnodes) {
    std::string concat_node_name = node_get_first_output_name(*consumer);
    auto concat_node_op_type = VAIP_ORT_API(node_op_type)(*consumer);
    if (concat_node_op_type == "QConcat" &&
        node_has_attr(*consumer, "QDQConcat")) {

      out_sc = node_get_attr_float(*consumer, "output_scale");
      out_zp = (uint16_t)(node_get_attr_float(*consumer, "output_zp"));
      concat_in_child = "true";

      MY_LOG(1) << "Concat is a child of QEltWiseAdd node";
    }
  }
  return std::make_tuple(out_sc, out_zp, concat_in_child);
}

static std::vector<const Node*> get_all_parent_nodes(const Node* cnode) {
  auto node_inputs = node_get_inputs(*cnode);
  std::vector<const Node*> ret;
  for (const auto& ni : node_inputs) {
    if (ni.node != nullptr) {
      ret.emplace_back(ni.node);
    }
  }
  return ret;
}

static bool check_div_parent(Graph& g, const Node* a, float& div_scale,
                             uint16_t& div_zp, NodeArg*& div_node_arg) {
  auto inputs = get_all_parent_nodes(a);
  if (inputs.size() == 0)
    return false;
  auto x = inputs[0];
  auto parent_op_type = VAIP_ORT_API(node_op_type)(*x);
  if (parent_op_type != "Div")
    return false;
  else {
    inputs = get_all_parent_nodes(x);
    if (inputs.size() == 0)
      return false;
    x = inputs[0];
    parent_op_type = VAIP_ORT_API(node_op_type)(*x);
    // if (parent_op_type != "DequantizeLinear")
    //   return false;

    while (parent_op_type != "GatherElements") {
      inputs = get_all_parent_nodes(x);
      if (inputs.size() == 0)
        return false;
      x = inputs[0];
      parent_op_type = VAIP_ORT_API(node_op_type)(*x);
    }
    inputs = get_all_parent_nodes(x);
    if (inputs.size() == 0)
      return false;
    x = inputs[0];
    parent_op_type = VAIP_ORT_API(node_op_type)(*x);
    if (parent_op_type != "DequantizeLinear")
      return false;
  }
  auto input_node_args = node_get_input_node_args(*x);
  for (auto ni : input_node_args) {
    if (!node_arg_is_constant(g, *ni)) {
      div_node_arg = const_cast<NodeArg*>(ni);
      continue;
    }
    auto ni_dtype = node_arg_get_element_type(*ni);
    if (ni_dtype == (int)ONNX_NAMESPACE::TensorProto_DataType_FLOAT)
      div_scale = node_arg_get_const_data_as_float(g, *ni) / std::sqrt(192.0);
    else if (ni_dtype == (int)ONNX_NAMESPACE::TensorProto_DataType_UINT8 ||
             ni_dtype == (int)ONNX_NAMESPACE::TensorProto_DataType_UINT16)
      div_zp = vaip::dd::get_zp_from_node(g, *ni);
    else
      MY_LOG(1) << "Div Zero point not valid" << std::endl;
  }
  return true;
}
struct MergeQEltWiseAdd {
  MergeQEltWiseAdd(IPass& self) {}
  static std::tuple<bool, bool> check_add_in_parent(const Node* in_0,
                                                    const Node* in_1) {
    if (in_0 && node_is_op(*in_0, "QEltWiseAdd", "com.xilinx")) {
      return {true, true};
    }
    if (in_1 && node_is_op(*in_1, "QEltWiseAdd", "com.xilinx")) {
      return {true, false};
    }
    return {false, false};
  }

  static bool lrn_in_consumer(const Graph& g, const NodeArg* out) {
    auto consumers = graph_get_consumer_nodes(g, node_arg_get_name(*out));
    for (auto node : consumers) {
      if (node_is_op(*node, "QLayerNorm", "com.xilinx")) {
        return true;
      }
    }
    return false;
  }

  static bool check_lrn_in_sibling(const Graph& g, const NodeArg* in_0,
                                   const NodeArg* in_1) {
    return lrn_in_consumer(g, in_0) || lrn_in_consumer(g, in_1);
  }
  static std::vector<std::string> change_inputs(const NodeArg& a,
                                                const NodeArg& b,
                                                const int32_t& amat_uint16) {
    std::vector<std::string> dtypes;
    // Add conditional code here (Below may only work for mdsqr)
    auto a_dtype = node_arg_get_element_type(a);
    if (a_dtype == (int)ONNX_NAMESPACE::TensorProto_DataType_UINT16) {
      if (!amat_uint16)
        dtypes.emplace_back("bfloat16");
      else
        dtypes.emplace_back("uint16");
    }

    else
      dtypes.emplace_back("uint8");

    auto b_dtype = node_arg_get_element_type(b);
    if (b_dtype == (int)ONNX_NAMESPACE::TensorProto_DataType_UINT16) {
      dtypes.emplace_back("uint16");
    } else
      dtypes.emplace_back("uint8");
    dtypes.emplace_back("int32");
    return dtypes;
  }
  static std::vector<std::string> change_outputs(const NodeArg& a,
                                                 const int32_t& amat_uint16) {
    std::vector<std::string> dtypes;
    // Add conditional code here (Below may only work for mdsqr)
    if (!amat_uint16)
      dtypes.emplace_back("bfloat16");
    else
      dtypes.emplace_back("uint16");
    return dtypes;
  }
  static std::unique_ptr<Rule> create_rule(IPass* self) {

    auto q = vaip::pattern_zoo::get_pattern("m_qeltwise_add_0");
    CHECK(q != nullptr) << "Pattern returned is null";
    return Rule::create_rule(
        q, [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
          // Get nodes
          auto attr_nodes = vaip::dd::get_node_names(graph, binder);
          // inputs/outputs
          auto a_node = binder["a"];
          auto b_node = binder["b"];

          auto a_shape = node_arg_get_shape_i64(*a_node.node_arg);
          auto b_shape = node_arg_get_shape_i64(*b_node.node_arg);
          if (*(a_shape.get()) != *(b_shape.get())) {
            MY_LOG(1) << "Shape mismatch, not fusing eltwiseadd";
            return false;
          }
          std::string design_param_ = "4x2";

          auto args = self->get_pass_proto().args();
          if (!args.empty()) {
            std::vector<std::string> arg;
            for (const auto& str : args) {
              arg.push_back(str);
            }
            if (arg[0] == "4x4PSU")
              design_param_ = arg[0];
          }

          int is_biasadd = false;
          if ((*node_arg_is_constant)(*graph, *a_node.node_arg) ||
              (*node_arg_is_constant)(*graph, *b_node.node_arg))
            is_biasadd = true;
          float div_scale_0, div_scale_1;
          uint16_t div_zp_0, div_zp_1;
          NodeArg* div_node_arg_0 = nullptr;
          NodeArg* div_node_arg_1 = nullptr;
          bool is_div_add = check_div_parent(*graph, a_node.node, div_scale_0,
                                             div_zp_0, div_node_arg_0) &&
                            check_div_parent(*graph, b_node.node, div_scale_1,
                                             div_zp_1, div_node_arg_1);
          auto q_node = binder["q"];

          // get qparams
          auto as_node = binder["a_s"];
          auto az_node = binder["a_z"];
          auto bs_node = binder["b_s"];
          auto bz_node = binder["b_z"];
          auto ys_node = binder["y_s"];
          auto yz_node = binder["y_z"];
          // auto dq1_node = binder["dq1"];

          auto a_sc =
              node_arg_get_const_data_as_float(*graph, *as_node.node_arg);
          auto a_zp = vaip::dd::get_zp_from_node(*graph, *az_node.node_arg);
          auto b_sc =
              node_arg_get_const_data_as_float(*graph, *bs_node.node_arg);
          auto b_zp = vaip::dd::get_zp_from_node(*graph, *bz_node.node_arg);

          // TODO: Update input q params withe concat's output q params , if
          // Concat is sibling to the current op
          bool a_params_updated = false;
          bool b_params_updated = false;
          std::string concat_in_sibling = "false";
          auto sibling_concat_qparams_a =
              get_concat_input_qparams_add_sibling(graph, a_node, a_sc, a_zp);
          if (std::get<2>(sibling_concat_qparams_a) == "true") {
            a_sc = std::get<0>(sibling_concat_qparams_a);
            a_zp = std::get<1>(sibling_concat_qparams_a);
            concat_in_sibling = std::get<2>(sibling_concat_qparams_a);
            a_params_updated = true;
          }
          auto sibling_concat_qparams_b =
              get_concat_input_qparams_add_sibling(graph, b_node, b_sc, b_zp);
          if (std::get<2>(sibling_concat_qparams_b) == "true") {
            b_sc = std::get<0>(sibling_concat_qparams_b);
            b_zp = std::get<1>(sibling_concat_qparams_b);
            concat_in_sibling = std::get<2>(sibling_concat_qparams_b);
            b_params_updated = true;
          }

          float input_scale = 0;
          uint16_t input_zp = 0;
          if (a_params_updated) {
            input_scale = a_sc;
            input_zp = a_zp;
          } else if (b_params_updated) {
            input_scale = b_sc;
            input_zp = b_zp;
          }

          auto y_sc =
              node_arg_get_const_data_as_float(*graph, *ys_node.node_arg);
          auto y_zp = vaip::dd::get_zp_from_node(*graph, *yz_node.node_arg);
          std::string concat_in_child = "false";
          auto child_concat_qparams =
              get_concat_output_qparams_add(graph, q_node, y_sc, y_zp);
          if (std::get<2>(child_concat_qparams) == "true") {
            y_sc = std::get<0>(child_concat_qparams);
            y_zp = std::get<1>(child_concat_qparams);
            concat_in_sibling = std::get<2>(child_concat_qparams);
            concat_in_child = "true";
          }

          auto node_name = node_arg_get_name(*q_node.node_arg);
          // Add conditions based on different models
          auto [add_in_parent, first_parent_is_add] =
              check_add_in_parent(a_node.node, b_node.node);
          auto lrn_in_sibling =
              check_lrn_in_sibling(*graph, a_node.node_arg, b_node.node_arg);
          auto lrn_in_opt = lrn_in_consumer(*graph, q_node.node_arg);

          int32_t amat_uint16 = lrn_in_sibling ? 0 : 1;
          int32_t cmat_uint16 = lrn_in_opt ? 0 : 1;

          std::vector<float> input_q_params{a_sc, float(a_zp), b_sc,
                                            float(b_zp)};
          if (add_in_parent && !first_parent_is_add) {
            std::swap(input_q_params[0], input_q_params[2]);
            std::swap(input_q_params[1], input_q_params[3]);
          }
          std::vector<float> output_q_params{y_sc, float(y_zp)};
          MY_LOG(1) << "QEltWiseAdd: Matched " << attr_nodes.size()
                    << std::endl;

          if (is_biasadd) {
            auto mul_const_in =
                node_arg_get_const_data_as_u16s(*graph, *b_node.node_arg);
            std::vector<uint16_t> mul_const_vec(mul_const_in.begin(),
                                                mul_const_in.end());
            auto mul_const_scale =
                node_arg_get_const_data_as_float(*graph, *bs_node.node_arg);
            auto mul_const_zp =
                node_arg_get_const_data_as_u16(*graph, *bz_node.node_arg);
            auto gamma = vaip::dd::qmatmulcalc::dq_vec_to_bf16(
                mul_const_vec, mul_const_scale, mul_const_zp);
            auto gamma_shape = node_arg_get_shape_i64(*b_node.node_arg);
            auto& input_gamma_arg = vaip::dd::insert_named_tensor_in_graph(
                graph, node_name + "_gamma_", gamma, *gamma_shape);

            std::vector<std::string> inp_dtype = {"bfloat16", "bfloat16"};
            std::vector<std::string> out_dtype = {"bfloat16"};

            NodeBuilder(*graph, *self)
                .set_input_node_args({a_node.node_arg, &input_gamma_arg})
                .set_op_type("Qbias_add", "com.xilinx")
                .set_anchor_point1(*q_node.node)
                .add("nodes", attr_nodes)
                .add("input_q_params", input_q_params)
                .add("output_q_params", output_q_params)
                .add("in_dtypes", inp_dtype)
                .add("out_dtypes", out_dtype)
                .build();
          } else if (is_div_add) {
            input_q_params = {div_scale_0, (float)div_zp_0, div_scale_1,
                              (float)div_zp_1};
            std::vector<int32_t> elt_coeffs =
                vaip::dd::qmatmulcalc::calculate_add_qdq_params(
                    div_scale_0, div_zp_0, div_scale_1, div_zp_1, y_sc, y_zp);
            std::string elt_coeff_name = std::string(node_name + "_qdq_");
            auto& elt_arg = vaip::dd::insert_named_tensor_in_graph<int32_t>(
                graph, elt_coeff_name, elt_coeffs,
                std::vector({(int64_t)elt_coeffs.size()}));
            std::vector<const NodeArg*> new_inputs{div_node_arg_0,
                                                   div_node_arg_1, &elt_arg};
            auto node_builder = NodeBuilder(*graph, *self);
            node_builder.set_input_node_args(new_inputs);
            node_builder.set_op_type("QGatherDivAdd", "com.xilinx");
            node_builder.set_anchor_point1(*q_node.node);
            node_builder.add("nodes", attr_nodes);
            node_builder.add("input_q_params", input_q_params);
            node_builder.add("output_q_params", output_q_params);
            node_builder.add(
                "in_dtypes",
                change_inputs(*div_node_arg_0, *div_node_arg_1, amat_uint16));
            node_builder.add("out_dtypes",
                             change_outputs(*q_node.node_arg, cmat_uint16));
            node_builder.add("amat_uint16", (int64_t)amat_uint16);
            node_builder.add("cmat_uint16", (int64_t)cmat_uint16);
            node_builder.add(
                "concat_in_child",
                concat_in_child); // used by ops that are consumers of
                                  // EltWiseAdd, if concat is one of the
                                  // consumers of EltwiseAdd
            node_builder.add("output_scale", y_sc);
            node_builder.add("output_zp", (float)y_zp);
            node_builder.build();
          } else {
            // Accessing the args value from config proto, if it is intadd then
            // select QIntEltwiseAdd
            std::string op_name = "QEltWiseAdd";
            bool intadd = false;
            auto args = self->get_pass_proto().args();
            if (!args.empty()) {
              std::vector<std::string> arg;
              for (const auto& str : args) {
                arg.push_back(str);
              }
              if (arg[0] == "intadd")
                intadd = true;
            }
            std::vector<int32_t> elt_coeffs(16, 0);
            if (intadd == false) {
              elt_coeffs[0] =
                  vaip::dd::qmatmulcalc::float_to_bfloat16(input_q_params[0]);
              elt_coeffs[1] = int32_t(input_q_params[1]);
              elt_coeffs[2] =
                  vaip::dd::qmatmulcalc::float_to_bfloat16(input_q_params[2]);
              elt_coeffs[3] = int32_t(input_q_params[3]);
              elt_coeffs[4] =
                  vaip::dd::qmatmulcalc::float_to_bfloat16(1.0f / y_sc);
              elt_coeffs[5] = y_zp;
              elt_coeffs[6] = (int32_t)amat_uint16;
              elt_coeffs[7] = (int32_t)cmat_uint16;
              op_name = "QEltWiseAdd";

            } else {

              elt_coeffs = vaip::dd::qmatmulcalc::calculate_add_qdq_params(
                  a_sc, a_zp, b_sc, b_zp, y_sc, y_zp);
              op_name = "QIntEltwiseAdd";
            }
            std::string elt_coeff_name = std::string(node_name + "_qdq_");
            auto& elt_arg = vaip::dd::insert_named_tensor_in_graph<int32_t>(
                graph, elt_coeff_name, elt_coeffs,
                std::vector({(int64_t)elt_coeffs.size()}));
            std::vector<const NodeArg*> new_inputs{a_node.node_arg,
                                                   b_node.node_arg, &elt_arg};
            if (add_in_parent && !first_parent_is_add) {
              std::swap(new_inputs[0], new_inputs[1]);
            }
            NodeBuilder(*graph, *self)
                .set_input_node_args(new_inputs)
                .set_op_type(op_name, "com.xilinx")
                .set_anchor_point1(*q_node.node)
                .add("nodes", attr_nodes)
                .add("input_q_params", input_q_params)
                .add("output_q_params", output_q_params)
                .add("in_dtypes", change_inputs(*a_node.node_arg,
                                                *b_node.node_arg, amat_uint16))
                .add("out_dtypes",
                     change_outputs(*q_node.node_arg, cmat_uint16))
                .add("amat_uint16", (int64_t)amat_uint16)
                .add("design_param", design_param_)
                .add("cmat_uint16", (int64_t)cmat_uint16)
                .add("concat_in_child",
                     concat_in_child) // used by ops that are consumers of
                                      // EltWiseAdd, if concat is one of the
                                      // consumers of EltwiseAdd
                .add("output_scale", y_sc)
                .add("output_zp", (float)y_zp)
                .add("concat_in_sibling",
                     concat_in_sibling) // used by ops that are consumers of
                                        // IConv, if concat is one of the
                                        // consumers of IConv
                .add("input_scale", input_scale)
                .add("input_zp", (float)input_zp)
                .build();
          }

          return true;
        });
  }

  void process(IPass& self, Graph& graph) { create_rule(&self)->apply(&graph); }
};

DEFINE_VAIP_PASS(MergeQEltWiseAdd, vaip_pass_dd_merge_qeltwise_add)
