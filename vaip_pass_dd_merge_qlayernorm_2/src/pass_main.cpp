/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

#include <glog/logging.h>

#include "vaip/pattern_zoo.hpp"
#include "vaip/vaip.hpp"
#include "vitis/ai/env_config.hpp"

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

#include <utility> // for std::pair
DEF_ENV_PARAM(DEBUG_DD_PATTERN, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_DD_PATTERN) >= n)
// #include "../../vaip_pass_dd_merge_qlayernorm\src\calc_coeffs.hpp"
using namespace vaip_core;

struct MergeQLayerNorm2 {
  MergeQLayerNorm2(IPass& self) : self_{self} {}
  static bool check_conv_matmuladd_or_pool_in_parent(const Node* in_0) {
    if (in_0 && (node_is_op(*in_0, "IConv", "com.xilinx") ||
                 node_is_op(*in_0, "QGlobalAvgPool", "com.xilinx") ||
                 node_is_op(*in_0, "QMatMulAdd", "com.xilinx"))) {
      return true;
    }
    return false;
  }
  std::unique_ptr<Rule> create_rule(IPass* self) {
    auto q2 = vaip::pattern_zoo::get_pattern("m_qlayernorm_2");
    CHECK(q2 != nullptr) << "Pattern returned is null";

    return Rule::create_rule(
        q2, [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
          /*
          TODO: Remove following leakage check(for mxgan); should be generalised
          - identify the 6th node and 8th node
          - replace consumers of 8th node (from) with consumers of 6th node (to)
          [fusnction used : graph_replace_node_arg]
              - Checked conditions
                  - q1 should have > 1 consumers
                  - q2 should have 1 consumer
                  - then 8th node will be the consumer of q2 node


          */
          // TODO This is explicitly excluded in LayerNorm2 pattern to prevent a
          // crash in mxgan network
          //  This is the last one to match in the network
          std::vector<std::string> exclusion_pattern_list = {
              "987_convert_QuantizeLinear_Output"};
          auto output_name = node_arg_get_name(*binder["q2"].node_arg);
          if (std::any_of(exclusion_pattern_list.cbegin(),
                          exclusion_pattern_list.cend(),
                          [&](std::string v) { return v == output_name; }))
            return false;

          //  if("987_convert_QuantizeLinear_Output" ==
          //  node_arg_get_name(*binder["q2"].node_arg))
          //     return false;

          std::string q1_name = node_arg_get_name(*binder["q1"].node_arg);
          bool leakage_case = false;
          if (graph_get_consumer_nodes(*graph, q1_name).size() > 1) {
            leakage_case = true;
            std::string dq4_name = node_arg_get_name(*binder["dq4"].node_arg);
            std::string q2_name = node_arg_get_name(*binder["q2"].node_arg);
            auto dequant_4_node_arg =
                VAIP_ORT_API(graph_get_node_arg)(*graph, dq4_name);

            if (graph_get_consumer_nodes(*graph, q2_name).size() == 1) {
              std::vector<const Node*> q2_nextnodes =
                  graph_get_consumer_nodes(*graph, q2_name);
              std::string dq_before_add_node_name =
                  node_get_first_output_name(*q2_nextnodes[0]);
              auto dq_before_add_node_arg = VAIP_ORT_API(graph_get_node_arg)(
                  *graph, dq_before_add_node_name);
              graph_replace_node_arg(
                  *graph, self_, *dq_before_add_node_arg,
                  *dequant_4_node_arg); // from 8th node to 6th node, 8t
            }
          }

          // Get nodes
          std::vector<std::string> attr_nodes;
          for (auto& ni : binder) {
            if (!(*node_arg_is_constant)(*graph, *ni.second.node_arg)) {
              attr_nodes.push_back(node_arg_get_name(*ni.second.node_arg));
            }
          }
          auto a_node = binder["a"];
          auto b_node = binder["b"];
          auto c_node = binder["c"];
          auto q1_node = binder["q1"];
          auto q2_node = binder["q2"];
          auto a_s_node = binder["a_s"];
          auto a_z_node = binder["a_z"];
          auto b_s_node = binder["b_s"];
          auto b_z_node = binder["b_z"];
          auto c_s_node = binder["c_s"];
          auto c_z_node = binder["c_z"];

          auto y_s_node = binder["y_s"];
          auto y_z_node = binder["y_z"];

          auto e_s_node = binder["e_s"];
          auto e_z_node = binder["e_z"];

          float e_sc =
              node_arg_get_const_data_as_float(*graph, *e_s_node.node_arg);
          uint16_t e_zp =
              vaip::dd::get_zp_from_node(*graph, *e_z_node.node_arg);

          if (leakage_case == true) {
            /*
                - If leakage case is triggered,
                    - Change the output scale and zp to q1's (5th node's) scale
               and zp (this means we are fusing only 5 nodes, and qdq
               computation uses the 5th node's scale and zp)
                    - Add the 8th node's output tensor to the attr_nodes (so
               that orphaned nodes data is appened to dod in context.json)
                        - 1-7 node's outputs/nodearg names are already present,
               just need to add 9th node's output/nodearg name
            */
            MY_LOG(1) << "LEAKAGE IN PATTERN -  adjusting accordingly";
            e_sc = node_arg_get_const_data_as_float(*graph, *y_s_node.node_arg);
            e_zp = vaip::dd::get_zp_from_node(*graph, *y_z_node.node_arg);
            std::string q2_name = node_arg_get_name(*binder["q2"].node_arg);
            if (graph_get_consumer_nodes(*graph, q2_name).size() == 1) {
              std::vector<const Node*> q2_nextnodes =
                  graph_get_consumer_nodes(*graph, q2_name);
              std::string dq_before_add_node_name =
                  node_get_first_output_name(*q2_nextnodes[0]);
              // auto dq_before_add_node_arg =
              // node_get_first_output_node_arg(*q2_nextnodes[0]))
              auto dq_before_add_node_arg = VAIP_ORT_API(graph_get_node_arg)(
                  *graph, dq_before_add_node_name);
              attr_nodes.push_back(node_arg_get_name(*dq_before_add_node_arg));
            }
          }

          //////////////////////////////////////////

          // qdq1
          auto alpha_shape = node_arg_get_shape_i64(*b_node.node_arg);
          auto alpha_data =
              node_arg_get_const_data_as_u8s(*graph, *b_node.node_arg);

          std::vector<uint8_t> alpha;
          for (auto e : alpha_data)
            alpha.push_back(
                e); // node_arg_get_const_data_as_u8s(*graph, *b_node.node_arg);
          auto alpha_sc =
              node_arg_get_const_data_as_floats(*graph, *b_s_node.node_arg)[0];
          auto alpha_zp =
              node_arg_get_const_data_as_u8(*graph, *b_z_node.node_arg);
          auto qdq1 =
              vaip::dd::qmatmulcalc::dq_vec_to_bf16(alpha, alpha_sc, alpha_zp);

          std::string qdq1_initializer_name =
              node_arg_get_name(*e_z_node.node_arg) +
              "_qdq1_"; // TODO: this name is based on last q node after lrn
                        // because in mxgan lrn2 and lrn pattern are matching to
                        // same lrn node of original graph due to replication
                        // done while fusion.
          // qdq1_initializer_name.erase(str.length() - 6);
          const std::vector<int64_t> qdq1_initializer_shape = {
              (int64_t)qdq1.size()};

          auto& qdq1_arg = vaip::dd::insert_named_tensor_in_graph<uint16_t>(
              graph, qdq1_initializer_name, qdq1, qdq1_initializer_shape);

          // qdq2
          // params extraction
          auto beta_shape = node_arg_get_shape_i64(*c_node.node_arg);
          auto beta_data =
              node_arg_get_const_data_as_i32s(*graph, *c_node.node_arg);
          std::vector<int32_t> beta;
          for (auto e : beta_data)
            beta.push_back(
                e); // node_arg_get_const_data_as_u8s(*graph, *b_node.node_arg);
          auto beta_sc =
              node_arg_get_const_data_as_floats(*graph, *c_s_node.node_arg)[0];
          auto beta_zp =
              node_arg_get_const_data_as_i32(*graph, *c_z_node.node_arg);

          // calculation
          auto qdq2 =
              vaip::dd::qmatmulcalc::dq_vec_to_bf16(beta, beta_sc, beta_zp);

          // initialization
          std::string qdq2_initializer_name =
              node_arg_get_name(*e_z_node.node_arg) + "_qdq2_";
          const std::vector<int64_t> qdq2_initializer_shape = {
              (int64_t)qdq1.size()};
          auto& qdq2_arg = vaip::dd::insert_named_tensor_in_graph<uint16_t>(
              graph, qdq2_initializer_name, qdq2, qdq2_initializer_shape);

          // qdq_3
          auto act_sc =
              node_arg_get_const_data_as_floats(*graph, *a_s_node.node_arg)[0];
          auto act_zp = vaip::dd::get_zp_from_node(*graph, *a_z_node.node_arg);

          auto out_dtype = node_arg_get_element_type(*q2_node.node_arg);
          if (leakage_case == 1)
            out_dtype = node_arg_get_element_type(*q1_node.node_arg);

          auto lrn_is_uint16 =
              check_conv_matmuladd_or_pool_in_parent(a_node.node);
          bool act_dtype =
              (out_dtype == (int)ONNX_NAMESPACE::TensorProto_DataType_UINT8)
                  ? true
                  : false;

          std::vector<int32_t> qdq3(16, 0);

          auto coeff_dtype = act_dtype ? 0 : 1;

          qdq3[0] = vaip::dd::qmatmulcalc::float_to_bfloat16(1 / e_sc);
          qdq3[1] = static_cast<uint16_t>(e_zp);
          qdq3[2] = coeff_dtype;
          qdq3[3] = vaip::dd::qmatmulcalc::float_to_bfloat16(act_sc);
          qdq3[4] = lrn_is_uint16 ? (int32_t)act_zp
                                  : 0; // TODO this might fail other models
          qdq3[5] = lrn_is_uint16;

          std::string qdq3_initializer_name =
              node_arg_get_name(*e_z_node.node_arg) + "_qdq3_";
          const std::vector<int64_t> qdq3_initializer_shape = {
              (int64_t)qdq3.size()};
          auto& qdq3_arg = vaip::dd::insert_named_tensor_in_graph<int32_t>(
              graph, qdq3_initializer_name, qdq3, qdq3_initializer_shape);

          //////////////////////////////////////////
          std::vector<float> input_q_params;
          input_q_params.push_back(
              node_arg_get_const_data_as_float(*graph, *a_s_node.node_arg));

          input_q_params.push_back(
              float(vaip::dd::get_zp_from_node(*graph, *a_z_node.node_arg)));

          std::vector<float> output_q_params;
          output_q_params.push_back(e_sc);

          output_q_params.push_back(float(e_zp));
          MY_LOG(1) << "- QLayerNorm2: Matched " << attr_nodes.size()
                    << std::endl;
          // FIXME Hacky way other conditions will arise here
          std::string input_dtype =
              check_conv_matmuladd_or_pool_in_parent(a_node.node) ? "uint16"
                                                                  : "bfloat16";
          std::vector<std::string> in_dtypes = {input_dtype, "uint16", "uint16",
                                                "int32"};

          std::vector<std::string> out_dtypes = {
              vaip::dd::nodearg_dtype_to_string(*q2_node.node_arg)};

          /*
          if leakage case is triggered,
              - out_dtypes should be picked up from q1 node
              - q1 node should be the anchor point
          */

          if (leakage_case == true) {

            out_dtypes = {vaip::dd::nodearg_dtype_to_string(*q1_node.node_arg)};
          }

          if (leakage_case == false) {
            NodeBuilder(*graph, *self)
                .set_input_node_args(
                    {a_node.node_arg, &qdq1_arg, &qdq2_arg, &qdq3_arg})
                .set_op_type("QLayerNorm", "com.xilinx")
                .set_anchor_point1(*q2_node.node)
                .add("input_q_params", input_q_params)
                .add("output_q_params", output_q_params)
                .add("nodes", attr_nodes)
                .add("in_dtypes", in_dtypes)
                .add("out_dtypes", out_dtypes)
                .build();
          } else {
            NodeBuilder(*graph, *self)
                .set_input_node_args(
                    {a_node.node_arg, &qdq1_arg, &qdq2_arg, &qdq3_arg})
                .set_op_type("QLayerNorm", "com.xilinx")
                .set_anchor_point1(*q1_node.node)
                .add("input_q_params", input_q_params)
                .add("output_q_params", output_q_params)
                .add("nodes", attr_nodes)
                .add("in_dtypes", in_dtypes)
                .add("out_dtypes", out_dtypes)
                .build();
          }

          return true;
        });
  }

  void process(IPass& self, Graph& graph) { create_rule(&self)->apply(&graph); }

public:
  IPass& self_;
};

DEFINE_VAIP_PASS(MergeQLayerNorm2, vaip_pass_dd_merge_qlayernorm_2)
