/*
 *     The Xilinx Vitis AI Vaip in this distribution are provided under the
 * following free and permissive binary-only license, but are not provided in
 * source code form.  While the following free and permissive license is similar
 * to the BSD open source license, it is NOT the BSD open source license nor
 * other OSI-approved open source license.
 *
 *      Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights
 * reserved.
 *
 *      Redistribution and use in binary form only, without modification, is
 * permitted provided that the following conditions are met:
 *
 *      1. Redistributions must reproduce the above copyright notice, this list
 * of conditions and the following disclaimer in the documentation and/or other
 * materials provided with the distribution.
 *
 *      2. The name of Xilinx, Inc. may not be used to endorse or promote
 * products redistributed with this software without specific prior written
 * permission.
 *
 *      THIS SOFTWARE IS PROVIDED BY XILINX, INC. "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL XILINX, INC. BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 *      PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE
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
DEF_ENV_PARAM(DEBUG_DD_PATTERN, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_DD_PATTERN) >= n)
using namespace vaip_core;

// refer to get_concat_qparams() function in vaip_pass_dd_merge_mzdk5mha
static std::tuple<float, uint16_t, std::string>
get_concat_qparams_conv(onnxruntime::Graph* graph, const NodeInput& out_node,
                        float conv_outq_scale, uint16_t conv_outq_zero_point,
                        std::string concat_in_child) {
  // if (graph_get_consumer_nodes(*graph,node_arg_get_name(*out_node.node_arg
  // )).size() == 1) {
  std::vector<const Node*> final_quant_node_nextnodes =
      graph_get_consumer_nodes(*graph, node_arg_get_name(*out_node.node_arg));
  for (auto consumer : final_quant_node_nextnodes) {
    std::string dq_before_concat_node_name =
        node_get_first_output_name(*consumer);
    auto dq_before_concat_node_arg =
        VAIP_ORT_API(graph_get_node_arg)(*graph, dq_before_concat_node_name);

    if (graph_get_consumer_nodes(*graph,
                                 node_arg_get_name(*dq_before_concat_node_arg))
            .size() == 1) {
      std::vector<const Node*> dq_before_concat_next_nodes =
          graph_get_consumer_nodes(
              *graph, node_arg_get_name(*dq_before_concat_node_arg));
      std::string concat_node_name =
          node_get_first_output_name(*dq_before_concat_next_nodes[0]);
      auto concat_node_arg =
          VAIP_ORT_API(graph_get_node_arg)(*graph, concat_node_name);

      auto concat_node_op_type =
          VAIP_ORT_API(node_op_type)(*dq_before_concat_next_nodes[0]);

      if (concat_node_op_type == "Concat") {
        if (graph_get_consumer_nodes(*graph,
                                     node_arg_get_name(*concat_node_arg))
                .size() == 1) {
          std::vector<const Node*> concat_node_nextnodes =
              graph_get_consumer_nodes(*graph,
                                       node_arg_get_name(*concat_node_arg));
          std::string Q_node_after_concat_name =
              node_get_first_output_name(*concat_node_nextnodes[0]);
          auto Q_node_input_node_args =
              node_get_input_node_args(*concat_node_nextnodes[0]);

          auto Q_node_output_arg =
              node_get_output_node_args(*concat_node_nextnodes[0]);
          std::string q_node_arg_name =
              node_arg_get_name(*Q_node_output_arg[0]);

          if (q_node_arg_name !=
              "/up_blocks.2/cats.2/Concat_output_0_QuantizeLinear_Output") {

            concat_in_child = "true";
            conv_outq_scale = node_arg_get_const_data_as_float(
                *graph, *Q_node_input_node_args[1]);
            conv_outq_zero_point =
                vaip::dd::get_zp_from_node(*graph, *Q_node_input_node_args[2]);
          } else {
            MY_LOG(1) << "Found the staturation point of concat update, so not "
                         "updating the parent ICONV";
          }
        }
      }
    }
  }
  return std::make_tuple(conv_outq_scale, conv_outq_zero_point,
                         concat_in_child);
}

std::tuple<float, uint16_t, std::string>
get_sibling_concat_qparams(onnxruntime::Graph* graph, const NodeInput& in_node,
                           float in_scale, uint16_t in_zero_point) {
  auto node_found = in_node.node;
  std::string concat_in_sibling = "false";
  if (node_found != nullptr) {
    auto is_Iconv = VAIP_ORT_API(node_op_type)(*node_found);
    if (VAIP_ORT_API(node_op_type)(*node_found) == "IConv") {
      auto concat_attr = node_has_attr(*node_found, "concat_in_child");

      if (concat_attr &&
          node_get_attr_string(*node_found, "concat_in_child") == "true") {
        in_scale = node_get_attr_float(*node_found, "output_scale");
        in_zero_point =
            (uint16_t)(node_get_attr_float(*node_found, "output_zp"));
        concat_in_sibling = "true";
        MY_LOG(1) << "Conv has concat in silbling";
      }
    } else if (VAIP_ORT_API(node_op_type)(*node_found) ==
               "QuantizeLinear") { // if producer node is QuantizeLinear, as
                                   // IConv is the first PASS
      auto quant_node_name = node_get_first_output_name(*node_found);
      auto quant_consumers = graph_get_consumer_nodes(*graph, quant_node_name);
      for (auto consumer :
           quant_consumers) {     // for each consumer of QuantizeLinear
        if (VAIP_ORT_API(node_op_type)(*consumer) ==
            "DequantizeLinear") { // we dont want to go into another ICONV, so
                                  // check for Dequant here
          auto dequant_node_name = node_get_first_output_name(*consumer);
          auto dequant_consumers =
              graph_get_consumer_nodes(*graph, dequant_node_name);
          if ((VAIP_ORT_API(node_op_type)(*dequant_consumers[0]) ==
               "Concat")) { // Check if Concat is in the consumer of
                            // DequantLinear
            auto concat_node_name =
                node_get_first_output_name(*dequant_consumers[0]);
            auto concat_consumers =
                graph_get_consumer_nodes(*graph, concat_node_name);
            auto Q_node_input_node_args =
                node_get_input_node_args(*concat_consumers[0]);
            auto Q_node_output_arg =
                node_get_output_node_args(*concat_consumers[0]);
            std::string q_node_arg_name =
                node_arg_get_name(*Q_node_output_arg[0]);

            if (q_node_arg_name !=
                "/up_blocks.3/cats.0/Concat_output_0_QuantizeLinear_Output") {
              in_scale = node_arg_get_const_data_as_float(
                  *graph, *Q_node_input_node_args[1]);
              in_zero_point = vaip::dd::get_zp_from_node(
                  *graph, *Q_node_input_node_args[2]);
              concat_in_sibling = "true";

              MY_LOG(1) << "Conv node with concat sibling and Add as parent "
                           "updated here ";
            } else {
              MY_LOG(1) << "Found the saturation point node, so not updating "
                           "the sibling ICONV";
            }
          }
        }
      }
    }
  }
  return std::make_tuple(in_scale, in_zero_point, concat_in_sibling);
}

// Checking for quant, trans and dequant
std::string get_q_trans_dq_node(onnxruntime::Graph* graph,
                                const NodeInput& in_node) {

  std::string curr_node_name = node_arg_get_name(*in_node.node_arg);
  std::vector<const Node*> in_modified1 = {in_node.node};
  auto node_found = in_node.node;
  if (node_found != nullptr) {
    auto op_type = VAIP_ORT_API(node_op_type)(*node_found);

    if (VAIP_ORT_API(node_op_type)(*node_found) == "QuantizeLinear") {
      std::vector<const NodeArg*> quant_inputs =
          node_get_input_node_args(*node_found);
      auto quant_input_name = node_arg_get_name(*quant_inputs[0]);
      // std::string quant_input_node_name = node_inputs_as_string(*node_found);
      auto tranpose_node =
          VAIP_ORT_API(graph_producer_node)(*graph, quant_input_name);
      if (tranpose_node != nullptr) {
        if (VAIP_ORT_API(node_op_type)(*tranpose_node) == "Transpose") {
          std::vector<const NodeArg*> trans_inputs =
              node_get_input_node_args(*tranpose_node);
          auto trans_input_name = node_arg_get_name(*trans_inputs[0]);
          auto dequant_node =
              VAIP_ORT_API(graph_producer_node)(*graph, trans_input_name);
          if (dequant_node != nullptr) {
            if (VAIP_ORT_API(node_op_type)(*dequant_node) ==
                "DequantizeLinear") {
              auto dquant_node_input_name =
                  node_get_input_node_args(*dequant_node);
              std::vector<const Node*> in_modified1 = {dequant_node};
              return node_arg_get_name(*dquant_node_input_name[0]);
              // return
            }
          }
        }
      }
    }
  }

  return "";
}

// Checking for Dequant, trans and quant
std::tuple<std::vector<const Node*>, std::string, std::vector<std::string>>
get_dq_trans_q_node(onnxruntime::Graph* graph, const NodeInput& out_node) {

  auto quant_node_name = node_get_first_output_name(*out_node.node);
  auto quant_consumers = graph_get_consumer_nodes(*graph, quant_node_name);
  std::vector<const Node*> out_modified1 = {out_node.node};
  std::vector<std::string> dq_trans_q_node_names;
  if (quant_consumers.size() == 1) {

    if (VAIP_ORT_API(node_op_type)(*quant_consumers[0]) == "DequantizeLinear") {
      auto dequant_node_name = node_get_first_output_name(*quant_consumers[0]);
      dq_trans_q_node_names.push_back(dequant_node_name);
      auto dequant_consumers =
          graph_get_consumer_nodes(*graph, dequant_node_name);
      if (dequant_consumers.size() == 1) {
        if ((VAIP_ORT_API(node_op_type)(*dequant_consumers[0]) ==
             "Transpose")) { // Check if Transpose is in the consumer of
                             // QuantLinear
          auto transpose_node_name =
              node_get_first_output_name(*dequant_consumers[0]);
          dq_trans_q_node_names.push_back(transpose_node_name);
          std::vector<const Node*> transpose_consumers =
              graph_get_consumer_nodes(*graph, transpose_node_name);
          if (transpose_consumers.size() == 1) {
            if ((VAIP_ORT_API(node_op_type)(*transpose_consumers[0]) ==
                 "QuantizeLinear")) {
              auto quant_node_name =
                  node_get_first_output_name(*transpose_consumers[0]);
              dq_trans_q_node_names.push_back(quant_node_name);
              return std::make_tuple(
                  transpose_consumers,
                  node_get_first_output_name(*transpose_consumers[0]),
                  dq_trans_q_node_names);
            }
          }
        }
      }
    }
  }
  return std::make_tuple(out_modified1, "", dq_trans_q_node_names);
}

struct MergeIConv_3 {
  MergeIConv_3(IPass& self) : self_{self} {}

  static std::unique_ptr<Rule> create_rule_3(IPass* self) {
    auto ms_QuantizeLinear_15 = vaip::pattern_zoo::get_pattern("m_iconv_3");
    CHECK(ms_QuantizeLinear_15 != nullptr) << "Pattern returned is null";

    return Rule::create_rule(
        ms_QuantizeLinear_15,
        [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
          auto in_node = binder["input_0"];
          auto in_scale_node = binder["constant_1"];
          auto in_zp_node = binder["constant_2"];
          auto act_node = binder["ms_DequantizeLinear_3"];
          auto kernel_node = binder["ms_DequantizeLinear_7"];
          // weights
          auto wt_node = binder["constant_4"];
          auto wt_scale_node = binder["constant_5"];
          auto wt_zp_node = binder["constant_6"];
          // bias
          auto bias_node = binder["constant_8"];
          auto bias_scale_node = binder["constant_9"];
          auto bias_zp_node = binder["constant_10"];
          // conv out q
          auto conv_outq_scale_node = binder["constant_13"];
          auto conv_outq_zp_node = binder["constant_14"];

          auto conv_node = binder["Conv_12"];
          auto out_node = binder["ms_QuantizeLinear_15"];

          auto qd_trns_q = get_dq_trans_q_node(graph, out_node);
          auto q_trns_dq_name = get_q_trans_dq_node(graph, in_node);
          auto tenp = std::get<0>(qd_trns_q);
          std::vector<const Node*> out_modified = {tenp[0]};
          // std::vector<const Node*> in_modified = {q_trns_dq[0]};
          std::string in_name_quant, out_name_quant;

          auto in_name = node_arg_get_name(*in_node.node_arg);
          auto out_name = node_arg_get_name(*out_node.node_arg);

          in_name_quant = q_trns_dq_name != "" ? q_trns_dq_name : in_name;
          out_name_quant =
              std::get<1>(qd_trns_q) != "" ? std::get<1>(qd_trns_q) : out_name;
          auto in_quant_node_arg =
              VAIP_ORT_API(graph_get_node_arg)(*graph, in_name_quant);

          auto dq_trans_q_node_names = std::get<2>(qd_trns_q);
          std::vector<std::string> ns = vaip::dd::get_node_names(graph, binder);
          if (dq_trans_q_node_names.size() == 3) {
            for (auto& new_node_name : dq_trans_q_node_names)
              ns.push_back(new_node_name);
          }
          MY_LOG(1) << "found match at " << ns.front();

          auto out_quant_node_arg =
              VAIP_ORT_API(graph_get_node_arg)(*graph, out_name_quant);
          auto in_scale =
              node_arg_get_const_data_as_float(*graph, *in_scale_node.node_arg);
          auto in_zero_point =
              node_arg_get_const_data_as_u16(*graph, *in_zp_node.node_arg);

          // Change input qparams of IConv if sibling is concat
          std::string concat_in_sibling = "false";
          auto sibling_concat_params = get_sibling_concat_qparams(
              graph, in_node, in_scale, in_zero_point);

          if (std::get<2>(sibling_concat_params) == "true") {
            in_scale = std::get<0>(sibling_concat_params);
            in_zero_point = std::get<1>(sibling_concat_params);
            concat_in_sibling = std::get<2>(sibling_concat_params);
          }

          auto act_in_shape = node_arg_get_shape_i64(*act_node.node_arg);
          auto kernel_shape = node_arg_get_shape_i64(*kernel_node.node_arg);
          auto out_shape = node_arg_get_shape_i64(*conv_node.node_arg);
          // auto orig_output_shape =
          // node_arg_get_shape_i64(*out_node.node_arg);
          auto orig_output_shape = node_arg_get_shape_i64(*out_quant_node_arg);
          auto wt_name = node_arg_get_name(*wt_node.node_arg);

          // 10 parameters needed for qdq calc
          // in_scale, in_zero_point
          // weights/weights scale/weights zp
          auto weights =
              node_arg_get_const_data_as_u8s(*graph, *wt_node.node_arg);
          auto weights_scale =
              node_arg_get_const_data_as_float(*graph, *wt_scale_node.node_arg);
          auto weights_zero_point =
              node_arg_get_const_data_as_u8(*graph, *wt_zp_node.node_arg);
          // mzdk5 specific change
          std::string op_type = "IConv";
          auto wt_shape = *node_arg_get_shape_i64(*wt_node.node_arg).get();
          if (wt_shape.size() >= 2 && wt_shape[wt_shape.size() - 1] == 1 &&
              wt_shape[wt_shape.size() - 2] == 1) {
            op_type = "QConv2MatMul";
          }
          // bias/bias scale/bias zp
          auto bias =
              node_arg_get_const_data_as_i32s(*graph, *bias_node.node_arg);
          // bias scale is array[1]
          auto bias_scale = node_arg_get_const_data_as_floats(
              *graph, *bias_scale_node.node_arg);
          auto bias_zero_point =
              node_arg_get_const_data_as_i32(*graph, *bias_zp_node.node_arg);
          // conv output q scale/zp
          auto conv_outq_scale = node_arg_get_const_data_as_float(
              *graph, *conv_outq_scale_node.node_arg);
          auto conv_outq_zero_point = node_arg_get_const_data_as_u16(
              *graph, *conv_outq_zp_node.node_arg);

          // Snippet to update output qparams of IConv with concat's output
          // qparams if concat is a consumer
          std::string concat_in_child = "false";
          auto concat_params =
              get_concat_qparams_conv(graph, out_node, conv_outq_scale,
                                      conv_outq_zero_point, concat_in_child);

          if (std::get<2>(concat_params) == "true") {
            conv_outq_scale = std::get<0>(concat_params);
            conv_outq_zero_point = std::get<1>(concat_params);
            concat_in_child = std::get<2>(concat_params);
          }
          // just to check if params are updated in this op
          if (std::get<0>(concat_params) != conv_outq_scale) {
            MY_LOG(1) << "IConv output qparams are updated with its consumer "
                         "concat's output q params ";
          }

          auto [C0, C1, C2, conv_shift, shft_c2] =
              vaip::dd::qmatmulcalc::dq_uint16A_uint8W_conv_q_param_gen(
                  in_scale, in_zero_point, weights, weights_scale,
                  weights_zero_point, *kernel_shape, bias, bias_scale[0],
                  bias_zero_point, conv_outq_scale, conv_outq_zero_point);

          auto node_name = node_arg_get_name(*out_node.node_arg);
          auto& input_c0_arg = vaip::dd::insert_named_tensor_in_graph<int64_t>(
              graph, node_name + "_c0_", C0, std::vector({(int64_t)C0.size()}));
          std::vector<int32_t> input_qdq(16, 0);
          input_qdq[2] = static_cast<int32_t>(C1);
          input_qdq[3] = static_cast<int32_t>(C2);
          input_qdq[8] = static_cast<int32_t>(shft_c2);
          input_qdq[9] = static_cast<int32_t>(conv_shift);
          input_qdq[10] = weights_zero_point;
          input_qdq[11] = in_zero_point;
          auto& input_qdq_arg = vaip::dd::insert_named_tensor_in_graph<int32_t>(
              graph, node_name + "_qdq_", input_qdq,
              std::vector({(int64_t)input_qdq.size()}));
          // hard code for m3uec, may need to change
          std::vector<std::string> input_types{"uint16", "uint8", "int64",
                                               "int32"};
          std::vector<std::string> output_types{"uint16"};

          // TODO: hardcode for mzdk5 attrs: from_iconv
          NodeBuilder(*graph, *self)
              .set_input_node_args({in_quant_node_arg, wt_node.node_arg,
                                    &input_c0_arg, &input_qdq_arg})
              .set_op_type(op_type, "com.xilinx")
              .clone_attrs(*conv_node.node)
              .add("nodes", ns)
              .add("input_shape", *act_in_shape)
              .add("weight_shape", *kernel_shape)
              .add("output_shape", *out_shape)
              .add("zero_point", int64_t(in_zero_point))
              .add("wt_name", wt_name)
              .add("in_dtypes", input_types)
              .add("out_dtypes", output_types)
              .add("input_format", "NHWC")
              .add("from_iconv", "true")
              .add("C1", std::to_string(C1))
              .add("C2", std::to_string(C2))
              .add("shift_conv", std::to_string(conv_shift))
              .add("shift_final", std::to_string(shft_c2))
              .add(
                  "concat_in_child",
                  concat_in_child) // used by ops that are consumers of IConv,
                                   // if concat is one of the consumers of IConv
              .add("output_scale", conv_outq_scale)
              .add("output_zp", (float)conv_outq_zero_point)
              .add(
                  "concat_in_sibling",
                  concat_in_child) // used by ops that are consumers of IConv,
                                   // if concat is one of the consumers of IConv
              .add("input_scale", in_scale)
              .add("input_zp", (float)in_zero_point)
              .set_anchor_point1(*out_modified[0])
              .build();
          return true;
        });
  }

  void process(IPass& self, Graph& graph) {
    MY_LOG(1) << "try matching IConv pattern 3";
    create_rule_3(&self)->apply(&graph);
  }

public:
  IPass& self_;
};

DEFINE_VAIP_PASS(MergeIConv_3, vaip_pass_dd_merge_iconv_3)
