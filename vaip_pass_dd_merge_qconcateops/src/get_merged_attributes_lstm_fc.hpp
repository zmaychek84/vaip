/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */
#pragma once
#include "nlohmann/json.hpp"
#include "vaip/dd/coeffs.hpp"
#include "vaip/dd/dd_utils.hpp"
#include "vaip/vaip.hpp"
#include <string>
#include <vector>

#ifdef __GNUC__
#  pragma GCC diagnostic pop
#endif
#include "vaip/vaip.hpp"
#include "vitis/ai/env_config.hpp"
#include <functional>
#include <glog/logging.h>
#include <numeric>

#include <cstdint>
#include <fstream>

namespace {
using namespace vaip_core;

bool same_shapes(std::vector<int64_t> in_shape,
                 std::vector<int64_t> out_shape) {
  for (int i = 0; i < 4; i++) {
    if (in_shape[i] != out_shape[i])
      return false;
  }
  return true;
}
static void add_attr_to_json(nlohmann::json& json) {
  json["opType"] = "conv";
  json["opIfmDtype"] = "uint16";
  json["opWtsDtype"] = "uint16";
  json["opOfmDtype"] = "uint16";
}

static std::vector<const Node*>
get_all_child_nodes(const onnxruntime::Graph& graph, const Node* node) {
  std::vector<const Node*> ret;
  for (const auto& output_arg : node_get_output_node_args(*node)) {
    std::string output_name = node_arg_get_name(*output_arg);
    std::vector<const onnxruntime::Node*> consumers =
        graph_get_consumer_nodes(graph, output_name);

    for (const auto consumer_node : consumers) {
      ret.push_back(consumer_node);
    }
  }
  return ret;
}
const NodeArg& concat_wgts_mswbjvw(onnxruntime::Graph* graph,
                                   const NodeArg& weight) {
  std::vector<int64_t> shape = {16, 8, 3, 3};
  int64_t total = 1;
  for (auto s : shape) {
    total *= s;
  }
  auto value = std::vector<uint16_t>(total, 0);
  auto actual_tensor = node_arg_get_const_data_as_u16s(*graph, weight);
  for (size_t i = 0u; i < actual_tensor.size(); ++i) {
    // padded at second dimension
    auto value_index = i + (i / 27) * (5 * 3 * 3);
    value[value_index] = actual_tensor[i];
  }

  auto name = node_arg_get_name(weight) + "_qdq";
  auto pad_tensor = tensor_proto_new_u16(name, shape, value);
  VAIP_ORT_API(graph_add_initialized_tensor)(*graph, *pad_tensor);
  return VAIP_ORT_API(node_arg_new)(
      *graph, name, &shape, ONNX_NAMESPACE::TensorProto_DataType_UINT16);
}

void handle_lstm(const onnxruntime::Graph& graph, const onnxruntime::Node& lstm,
                 std::vector<std::vector<int64_t>>& list_inp_shape,
                 std::vector<float>& list_scale, std::vector<uint16_t>& list_zp,
                 std::vector<const NodeArg*>& new_input,
                 std::vector<std::string>& list_wt_name) {
  auto input_node_args = node_get_input_node_args(lstm);
  auto inp_shape = node_arg_get_shape_i64(*input_node_args[0]);
  list_inp_shape.push_back(*inp_shape);

  auto w_shape = node_arg_get_shape_i64(*input_node_args[1]);
  auto r_shape = node_arg_get_shape_i64(*input_node_args[2]);
  auto b_shape = node_arg_get_shape_i64(*input_node_args[3]);

  auto inputs = node_get_inputs(lstm);
  auto inp_scale = node_arg_get_const_data_as_float(
      graph, *node_get_input_node_args(*inputs[0].node)[1]);
  auto w_scale = node_arg_get_const_data_as_float(
      graph, *node_get_input_node_args(*inputs[1].node)[1]);
  auto r_scale = node_arg_get_const_data_as_float(
      graph, *node_get_input_node_args(*inputs[2].node)[1]);
  auto b_scale = node_arg_get_const_data_as_float(
      graph, *node_get_input_node_args(*inputs[3].node)[1]);
  auto consumer =
      graph_get_consumer_nodes(graph, node_get_first_output_name(lstm))[0];
  auto out_scale = node_arg_get_const_data_as_float(
      graph, (*node_get_input_node_args(*consumer)[1]));

  list_scale.push_back(inp_scale);
  list_scale.push_back(w_scale);
  list_scale.push_back(r_scale);
  list_scale.push_back(b_scale);
  list_scale.push_back(out_scale);

  auto inp_zp = node_arg_get_const_data_as_u16(
      graph, *node_get_input_node_args(*inputs[0].node)[2]);
  auto w_zp = node_arg_get_const_data_as_u16(
      graph, *node_get_input_node_args(*inputs[1].node)[2]);
  auto r_zp = node_arg_get_const_data_as_u16(
      graph, *node_get_input_node_args(*inputs[2].node)[2]);
  auto b_zp = node_arg_get_const_data_as_u16(
      graph, *node_get_input_node_args(*inputs[3].node)[2]);
  auto out_zp = node_arg_get_const_data_as_u16(
      graph, (*node_get_input_node_args(*consumer)[2]));
  list_zp.push_back(inp_zp);
  list_zp.push_back(w_zp);
  list_zp.push_back(r_zp);
  list_zp.push_back(b_zp);
  list_zp.push_back(out_zp);

  auto input_1 = node_get_input_node_args(*inputs[1].node)[0];
  auto input_2 = node_get_input_node_args(*inputs[2].node)[0];
  auto input_3 = node_get_input_node_args(*inputs[3].node)[0];
  new_input.push_back(input_1);
  new_input.push_back(input_2);
  new_input.push_back(input_3);
  list_wt_name.push_back(node_arg_get_name(*input_1));
  list_wt_name.push_back(node_arg_get_name(*input_2));
  list_wt_name.push_back(node_arg_get_name(*input_3));
}

void handle_last_lstm(const onnxruntime::Graph& graph,
                      const onnxruntime::Node& lstm, NodeBuilder& building_node,
                      nlohmann::json& attr_array,
                      const std::vector<std::vector<int64_t>>& list_inp_shape,
                      const std::vector<float>& list_scale,
                      const std::vector<uint16_t>& list_zp) {
  nlohmann::json attr;
  attr["opType"] = "lstm";
  attr["opIfmDtype"] = "uint16";
  attr["opWtsDtype"] = "uint16";
  attr["opOfmDtype"] = "uint16";

  auto nn1 = graph_get_consumer_nodes(
      graph, node_arg_get_name(node_get_output_node_arg(lstm)))[0];
  auto nn2 = graph_get_consumer_nodes(
      graph, node_arg_get_name(node_get_output_node_arg(*nn1)))[0];
  auto nn3 = graph_get_consumer_nodes(
      graph, node_arg_get_name(node_get_output_node_arg(*nn2)))[0];
  auto nn4 = graph_get_consumer_nodes(
      graph, node_arg_get_name(node_get_output_node_arg(*nn3)))[0];
  auto nn5 = graph_get_consumer_nodes(
      graph, node_arg_get_name(node_get_output_node_arg(*nn4)))[0];
  auto nn6 = graph_get_consumer_nodes(
      graph, node_arg_get_name(node_get_output_node_arg(*nn5)))[0]; // reshape

  auto output_shape = node_arg_get_shape_i64(node_get_output_node_arg(*nn6));

  attr["input_shape"] = list_inp_shape[0];
  attr["output_shape"] = *output_shape;
  attr["list_scale"] = list_scale;
  attr["list_zero_point"] = list_zp;

  attr_array.push_back(attr);

  building_node.add("output_shape", *output_shape);

  auto inputs = node_get_inputs(lstm);
  int64_t inp_zp = node_arg_get_const_data_as_u16(
      graph, *node_get_input_node_args(*inputs[0].node)[2]);
  building_node.add("zero_point", inp_zp);
}
// utils.py
void get_merged_attributes_lstm_fc(NodeBuilder& building_node, NodeInput input,
                                   NodeInput output, onnxruntime::Graph* graph,
                                   binder_t* binder,
                                   std::string model_variant = "") {
  std::stringstream ss(model_variant);
  std::vector<std::string> model_variant_tokens;
  std::string token;
  while (std::getline(ss, token, '_')) {
    model_variant_tokens.push_back(token);
  }
  int64_t width = 0;
  std::vector<std::string> list_wt_name;
  std::vector<std::string> list_QDQ;
  std::vector<std::string> nodes;
  std::vector<const NodeArg*> input_list;
  nlohmann::json attr_array = nlohmann::json::array();

  bool is_last_lstm = false;                        // lstm
  std::vector<std::vector<int64_t>> list_inp_shape; // lstm
  std::vector<float> list_scale;                    // lstm
  std::vector<uint16_t> list_zp;                    // lstm

  auto in_shape = *(node_arg_get_shape_i64(*input.node_arg).get());
  auto out_shape = *(node_arg_get_shape_i64(*output.node_arg).get());
  if (out_shape.back() == 32632)
    out_shape.back() = 32768;
  else if (out_shape.back() == 201)
    out_shape.back() = 202;
  else if (out_shape.back() == 179)
    out_shape.back() = 180;

  std::vector<int64_t> first_conv_in_shape;
  std::vector<std::string> dd_op_out_shape;

  if (out_shape.size() != 3) {
    dd_op_out_shape = {std::to_string(out_shape[2]) + " " +
                       std::to_string(out_shape[0]) + " " +
                       std::to_string(out_shape[3]) + " " +
                       std::to_string(out_shape[1])};
  } else {
    std::string shape_str = "";
    for (auto di : out_shape) {
      auto s = std::to_string(di);
      shape_str = shape_str + s + " ";
    }
    shape_str.pop_back();
    dd_op_out_shape = {shape_str};
  }

  input_list.push_back(&(node_get_output_node_arg(
      *input.node))); // set_input_node_args would clean existing inputs

  std::vector<int64_t> act_input_shape;
  std::vector<int64_t> kernel_shape;
  int cnt = 0;
  bool has_conv = 0;
  for (auto& ni : *binder) {
    cnt++;
    if ((*node_arg_is_constant)(*graph, *ni.second.node_arg)) {
      continue;
    }
    auto node = ni.second.node;
    if (node != input.node) {
      nodes.push_back(node_arg_get_name(*ni.second.node_arg));
    }

    std::string op_type = VAIP_ORT_API(node_op_type)(*node);
    std::transform(op_type.begin(), op_type.end(), op_type.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    auto is_lstm = op_type.find("lstm") != std::string::npos;
    if (is_lstm) {
      handle_lstm(*graph, *node, list_inp_shape, list_scale, list_zp,
                  input_list, list_wt_name);
      if (is_last_lstm) {
        handle_last_lstm(*graph, *node, building_node, attr_array,
                         list_inp_shape, list_scale, list_zp);
      }
      is_last_lstm = true;
      continue;
    }

    auto is_maxpool = op_type.find("maxpool") != std::string::npos;
    if (is_maxpool) {
      auto attr_proto =
          node_attributes_get(node_get_attributes_ref(*node), "kernel_shape");
      auto maxpool_kernel_shape =
          VAIP_ORT_API(attr_proto_get_ints)(*attr_proto);
      attr_proto =
          node_attributes_get(node_get_attributes_ref(*node), "strides");
      auto maxpool_stride = VAIP_ORT_API(attr_proto_get_ints)(*attr_proto);
      attr_array[attr_array.size() - 1]["maxpool_kernel_shape"] =
          maxpool_kernel_shape;
      attr_array[attr_array.size() - 1]["maxpool_stride"] = maxpool_stride;
    }

    auto is_conv = op_type.find("conv") != std::string::npos;

    auto is_matmul = op_type.find("matmul") != std::string::npos;
    if (!is_conv && !is_matmul) {
      continue;
    }
    auto inputs = node_get_inputs(*node);

    // update attr
    act_input_shape = *node_arg_get_shape_i64(*inputs[0].node_arg);
    kernel_shape = *node_arg_get_shape_i64(*inputs[1].node_arg);
    std::vector<int64_t> output_shape = node_get_output_shape(*node, 0);
    nlohmann::json attr;
    // if (attr_array.size() == 0) {
    auto inp_shape = *node_arg_get_shape_i64(*inputs[0].node_arg);
    first_conv_in_shape = inp_shape;
    width = inp_shape[3];
    int in_channel_pad = 8;
    act_input_shape[1] = in_channel_pad;
    kernel_shape[1] = in_channel_pad;
    attr["graphID"] = std::stoi(model_variant_tokens[1]); // width;
    attr["inChannels"] = in_channel_pad;
    attr["outChannels"] = kernel_shape[0];
    //}

    add_attr_to_json(attr);
    if (is_matmul)
      attr["opType"] = "convformatmuladd";

    attr["group"] = node_get_attr_int_with_default(*node, "group", 1);
    attr["input_shape"] = act_input_shape;

    const Node* out_node = node;
    auto cn = node;
    bool relu_found = 0;
    for (int i = 0; i < 6; i++) {
      if (graph_get_consumer_nodes(
              *graph, node_arg_get_name(node_get_output_node_arg(*node)))
              .size() == 1) {
        auto nn = graph_get_consumer_nodes(
            *graph, node_arg_get_name(node_get_output_node_arg(*cn)))[0];
        std::string nn_op_type = VAIP_ORT_API(node_op_type)(*nn);
        std::transform(nn_op_type.begin(), nn_op_type.end(), nn_op_type.begin(),
                       [](unsigned char c) { return std::tolower(c); });
        auto is_nn_pool = nn_op_type.find("pool") != std::string::npos;
        auto is_nn_conv = nn_op_type.find("conv") != std::string::npos;
        if (is_nn_conv)
          break;
        if (is_nn_pool) {
          out_node = nn;
          break;
        }
        cn = nn;
      }
    }
    attr["output_shape"] = node_get_output_shape(*out_node, 0);

    attr["weight_shape"] = kernel_shape;
    int zp_node_ind;
    if (is_conv)
      zp_node_ind = 1;
    else
      zp_node_ind = 0;
    auto zero_point_node_arg =
        node_get_input_node_args(*inputs[zp_node_ind].node)[2];
    int64_t zero_point =
        node_arg_get_const_data_as_u16(*graph, *zero_point_node_arg);
    attr["zero_point"] = zero_point;

    attr["width"] = width;

    auto wt = node_get_input_node_args(*inputs[1].node)[0];
    // update node builder
    if (input_list.size() == 1) {
      const NodeArg& pad_input = concat_wgts_mswbjvw(graph, *wt);
      input_list.push_back(&pad_input);
    } else {
      input_list.push_back(wt);
    }
    list_wt_name.push_back(node_arg_get_name(*input_list.back()));

    if (is_conv) {
      has_conv = 1;
      auto input_scale = node_get_input_node_args(*inputs[0].node)[1];
      auto input_zp = node_get_input_node_args(*inputs[0].node)[2];
      auto weights = node_get_input_node_args(*inputs[1].node)[0];
      auto wts_scale = node_get_input_node_args(*inputs[1].node)[1];
      auto wts_zp = node_get_input_node_args(*inputs[1].node)[2];
      auto inp_shape = *node_arg_get_shape_i64(*inputs[0].node_arg);
      auto wts_shape = *node_arg_get_shape_i64(*inputs[1].node_arg);

      auto in_scale = node_arg_get_const_data_as_float(*graph, *input_scale);
      auto in_zero_point = node_arg_get_const_data_as_u16(*graph, *input_zp);
      auto wts = node_arg_get_const_data_as_u16s(*graph, *weights);
      auto w_scale = node_arg_get_const_data_as_float(*graph, *wts_scale);
      auto w_zero_point = node_arg_get_const_data_as_u16(*graph, *wts_zp);
      auto k_shape = node_arg_get_shape_i64(*inputs[1].node_arg);

      auto output_conv = get_all_child_nodes(*graph, node);
      auto output_scale = node_get_input_node_args(*output_conv[0])[1];
      auto output_zp = node_get_input_node_args(*output_conv[0])[2];

      auto out_scale = node_arg_get_const_data_as_float(*graph, *output_scale);
      auto out_zero_point = node_arg_get_const_data_as_u16(*graph, *output_zp);

      if (inputs.size() == 3) {
        auto bias = node_get_input_node_args(*inputs[2].node)[0];
        auto bias_scale = node_get_input_node_args(*inputs[2].node)[1];
        auto bias_zp = node_get_input_node_args(*inputs[2].node)[2];
        auto b_shape = *node_arg_get_shape_i64(*inputs[2].node_arg);
        auto b = node_arg_get_const_data_as_i32s(*graph, *bias);
        auto b_scale = node_arg_get_const_data_as_float(*graph, *bias_scale);
        auto b_zero_point = node_arg_get_const_data_as_i32(*graph, *bias_zp);
        auto [C0, C1, C2, conv_shift, shft_c2] =
            vaip::dd::qmatmulcalc::dq_uint16A_uint16W_conv_q_param_gen(
                in_scale, in_zero_point, wts, w_scale, w_zero_point, *k_shape,
                b, b_scale, b_zero_point, out_scale, out_zero_point);
        attr["c1"] = static_cast<int32_t>(C1);
        attr["c2"] = static_cast<int32_t>(C2);
        attr["shift_conv"] = static_cast<int32_t>(conv_shift);
        attr["shift_out"] = static_cast<int32_t>(shft_c2);

        auto node_name = node_arg_get_name(*input_list.back());
        auto& input_c0_arg = vaip::dd::insert_named_tensor_in_graph<int64_t>(
            graph, node_name + "_c0_", C0, std::vector({(int64_t)C0.size()}));
        std::vector<int32_t> input_qdq(16, 0);
        input_qdq[0] = static_cast<int32_t>(C1);
        input_qdq[1] = static_cast<int32_t>(C2);
        input_qdq[2] = static_cast<int32_t>(conv_shift);
        input_qdq[3] = static_cast<int32_t>(shft_c2);
        auto& input_qdq_arg = vaip::dd::insert_named_tensor_in_graph<int32_t>(
            graph, node_name + "_qdq_", input_qdq,
            std::vector({(int64_t)input_qdq.size()}));
        input_list.push_back(&input_c0_arg);
        input_list.push_back(&input_qdq_arg);

      } else {
        gsl::span<const int32_t> bias1;
        float bias_scale = 0.0f;
        uint32_t bias_zero_point = 0;
        auto [C0, C1, C2, conv_shift, shft_c2] =
            vaip::dd::qmatmulcalc::dq_uint16A_uint16W_conv_q_param_gen(
                in_scale, in_zero_point, wts, w_scale, w_zero_point, *k_shape,
                bias1, bias_scale, bias_zero_point, out_scale, out_zero_point);
        attr["c1"] = static_cast<int32_t>(C1);
        attr["c2"] = static_cast<int32_t>(C2);
        attr["shift_conv"] = static_cast<int32_t>(conv_shift);
        attr["shift_out"] = static_cast<int32_t>(shft_c2);
        auto node_name = node_arg_get_name(*input_list.back());
        auto& input_c0_arg = vaip::dd::insert_named_tensor_in_graph<int64_t>(
            graph, node_name + "_c0_", C0, std::vector({(int64_t)C0.size()}));
        std::vector<int32_t> input_qdq(16, 0);
        input_qdq[0] = static_cast<int32_t>(C1);
        input_qdq[1] = static_cast<int32_t>(C2);
        input_qdq[2] = static_cast<int32_t>(conv_shift);
        input_qdq[3] = static_cast<int32_t>(shft_c2);
        auto& input_qdq_arg = vaip::dd::insert_named_tensor_in_graph<int32_t>(
            graph, node_name + "_qdq_", input_qdq,
            std::vector({(int64_t)input_qdq.size()}));
        input_list.push_back(&input_c0_arg);
        input_list.push_back(&input_qdq_arg);
      }
    }
    if (is_matmul) {
      attr["output_shape"] = out_shape;
      attr["orig_output_shape"] = node_get_output_shape(*out_node, 0);

      auto input_scale = node_get_input_node_args(*inputs[0].node)[1];
      auto input_zp = node_get_input_node_args(*inputs[0].node)[2];
      auto weights = node_get_input_node_args(*inputs[1].node)[0];
      auto wts_scale = node_get_input_node_args(*inputs[1].node)[1];
      auto wts_zp = node_get_input_node_args(*inputs[1].node)[2];
      auto inp_shape = *node_arg_get_shape_i64(*inputs[0].node_arg);
      auto wts_shape = *node_arg_get_shape_i64(*inputs[1].node_arg);
      auto in_scale = node_arg_get_const_data_as_float(*graph, *input_scale);
      auto in_zero_point = node_arg_get_const_data_as_u16(*graph, *input_zp);
      auto weight_shape = node_arg_get_shape_i64(*weights);
      auto weight_tensor = vaip::dd::fold2D<uint16_t>(
          node_arg_get_const_data_as_u16s(*graph, *weights),
          *weight_shape.get());
      auto w_scale = node_arg_get_const_data_as_float(*graph, *wts_scale);
      auto w_zero_point = node_arg_get_const_data_as_u16(*graph, *wts_zp);
      auto k_shape = node_arg_get_shape_i64(*inputs[1].node_arg);
      auto nn1 = graph_get_consumer_nodes(
          *graph, node_arg_get_name(node_get_output_node_arg(*node)))[0];
      auto nn2 = graph_get_consumer_nodes(
          *graph, node_arg_get_name(node_get_output_node_arg(*nn1)))[0];
      auto nn3 = graph_get_consumer_nodes(
          *graph, node_arg_get_name(node_get_output_node_arg(*nn2)))[0];
      auto nn3_inps = node_get_inputs(*nn3);
      auto bias = node_get_input_node_args(*nn3_inps[1].node)[0];
      auto bias_shape = node_arg_get_shape_i64(*bias);
      auto bias_tensor = vaip::dd::fold1D<uint16_t>(
          node_arg_get_const_data_as_u16s(*graph, *bias), *(bias_shape.get()));
      auto bias_scale = node_get_input_node_args(*nn3_inps[1].node)[1];
      auto bias_zp = node_get_input_node_args(*nn3_inps[1].node)[2];
      auto b_scale = node_arg_get_const_data_as_float(*graph, *bias_scale);
      auto b_zero_point = node_arg_get_const_data_as_u16(*graph, *bias_zp);
      auto output_matmul = graph_get_consumer_nodes(
          *graph, node_arg_get_name(node_get_output_node_arg(*nn3)))[0];
      auto output_scale = node_get_input_node_args(*output_matmul)[1];
      auto output_zp = node_get_input_node_args(*output_matmul)[2];
      auto out_scale = node_arg_get_const_data_as_float(*graph, *output_scale);
      auto out_zero_point = node_arg_get_const_data_as_u16(*graph, *output_zp);
      std::map<std::string, std::vector<int>> mswbjvw_qdq_srs_shifts = {
          {"80", {7, 7, 7}},   {"160", {7, 7, 7}},  {"320", {7, 7, 7}},
          {"640", {8, 8, 8}},  {"1280", {9, 9, 9}}, {"2560", {8, 8, 8}},
          {"5120", {8, 8, 7}}, {"8000", {8, 8, 7}},
      };
      std::string graphID = model_variant_tokens[1]; // std::to_string(width);
      std::vector<int> shifts = mswbjvw_qdq_srs_shifts[graphID];
      auto [C0, C1, C2, conv_shift, shft_c2, mat_shift] =
          vaip::dd::qmatmulcalc::dq_uint16A_uint16W_bias_matmul_q_param_gen(
              in_scale, in_zero_point, weight_tensor, w_scale, w_zero_point,
              bias_tensor, b_scale, b_zero_point, out_scale, out_zero_point,
              shifts);
      attr["c1"] = static_cast<int32_t>(C1);
      attr["c2"] = static_cast<int32_t>(C2);
      attr["wts_zp"] = static_cast<int16_t>(w_zero_point);
      auto node_name = node_arg_get_name(*input_list.back());
      auto& input_c0_arg = vaip::dd::insert_named_tensor_in_graph<int64_t>(
          graph, node_name + "_c0_", C0, std::vector({(int64_t)C0.size()}));
      std::vector<int32_t> input_qdq(16, 0);
      input_qdq[0] = static_cast<int32_t>(C1);
      input_qdq[1] = static_cast<int32_t>(C2);
      input_qdq[2] = static_cast<int32_t>(conv_shift);
      input_qdq[3] = static_cast<int32_t>(shft_c2);
      input_qdq[4] = static_cast<int32_t>(mat_shift);
      auto& input_qdq_arg = vaip::dd::insert_named_tensor_in_graph<int32_t>(
          graph, node_name + "_qdq_", input_qdq,
          std::vector({(int64_t)input_qdq.size()}));
      input_list.push_back(&input_c0_arg);
      input_list.push_back(&input_qdq_arg);
    }
    attr_array.push_back(attr);
  }

  auto first_node = graph_get_consumer_nodes(
      *graph, node_arg_get_name(node_get_output_node_arg(*input.node)))[0];

  std::vector<float> input_q_params;
  input_q_params.push_back(node_arg_get_const_data_as_float(
      *graph, *node_get_input_node_args(*first_node)[1]));
  input_q_params.push_back(node_arg_get_const_data_as_u16(
      *graph, *node_get_input_node_args(*first_node)[2]));
  std::vector<float> output_q_params;
  output_q_params.push_back(node_arg_get_const_data_as_float(
      *graph, *node_get_input_node_args(*output.node)[1]));
  output_q_params.push_back(node_arg_get_const_data_as_u16(
      *graph, *node_get_input_node_args(*output.node)[2]));
  std::vector<std::string> in_dtypes(input_list.size(), "uint16");
  std::vector<std::string> out_dtypes = {"uint16"};

  std::string in_shape1;
  if (has_conv) {
    if (same_shapes(in_shape, first_conv_in_shape)) {
      in_shape1 = std::to_string((int64_t)in_shape[2]) + " " +
                  std::to_string((int64_t)in_shape[0]) + " " +
                  std::to_string(in_shape[3]) + " ";
      in_shape1 = in_shape[3] != 320 ? in_shape1 + "4" : in_shape1 + "8";
    } else {
      in_shape1 = std::to_string((int64_t)in_shape[0]) + " " +
                  std::to_string((int64_t)in_shape[1]) + " " +
                  std::to_string(in_shape[2]) + " ";
      in_shape1 = in_shape[2] != 320 ? in_shape1 + "4" : in_shape1 + "8";
    }
  } else {
    in_shape1 = std::to_string((int64_t)in_shape[0]) + " " +
                std::to_string((int64_t)in_shape[1]) + " " +
                std::to_string(in_shape[2]);
  }
  std::vector<std::string> dd_op_in_shape = {in_shape1};

  building_node.add("dd_op_in_shape", dd_op_in_shape);
  building_node.add("dd_op_out_shape", dd_op_out_shape);
  building_node.add("input_q_params", input_q_params);
  building_node.add("output_q_params", output_q_params);
  building_node.add("in_dtypes", in_dtypes);
  building_node.add("out_dtypes", out_dtypes);

  auto attr_str = attr_array.dump(2);
  building_node.set_input_node_args(input_list);
  // op_fusion.py
  building_node.add("input_shape", act_input_shape);
  building_node.add("weight_shape", kernel_shape);
  building_node.add("list_attrs", attr_str);
  building_node.add("list_wt_name", list_wt_name);

  building_node.add("ort_out_index",
                    static_cast<int64_t>(list_wt_name.size() + 1));
  building_node.add("nodes", nodes);
  building_node.add("orig_output_shape",
                    *node_arg_get_shape_i64(*output.node_arg));

  building_node.add("Node_dtype", "<class 'numpy.uint16'>");
}

} // namespace
