/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */
#pragma once
#include <fstream>
#include <glog/logging.h>
#include <iostream>

#include "subpass_util.h"
#include "vaip/vaip.hpp"
#include "vitis/ai/env_config.hpp"
using namespace vaip_core;

DEF_ENV_PARAM(DEBUG_HT_SUBPASS, "0")
#define HT_SUBPASS_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_HT_SUBPASS) >= n)

namespace vaip_vaiml_subgraph_processor {

struct HT_initializer_mapping_pass {
  HT_initializer_mapping_pass(IPass& self, MetaDefProto& initializer_map)
      : self_{self}, initializer_map_(initializer_map) {}

  void process(Graph& graph) {
    create_ht_start_lstm_320_rule(&self_)->apply(&graph);
    create_ht_lstm_1024_end_rule(&self_)->apply(&graph);
    create_ht_concat_rule(&self_)->apply(&graph);
    create_ht_slice_rule(&self_)->apply(&graph);

    // backwards compatible for HT1.2
    ht_inner_map_.at("ht_000_main")
        .insert("=LayerNormalization->/decoder/lnorm_layer/Add_1_output_0");
    ht_inner_map_.at("ht_000_main")
        .insert("=LayerNormalization->/decoder_embedding/embed_lnorm/"
                "Add_1_output_0");
    ht_inner_map_.at("ht_000_main")
        .insert("=LayerNormalization->hidden_QuantizeLinear_Input");
  }

  void wts_name_mapper(const std::string& alias_prefix, binder_t& binder,
                       int wts_idx, int scale_idx, int zeropoint_idx) {
    auto& generic_param = *initializer_map_.mutable_generic_param();
    auto wts = node_arg_get_name(*binder[wts_idx].node_arg);
    generic_param[alias_prefix] = wts;
    auto wts_s = node_arg_get_name(*binder[scale_idx].node_arg);
    generic_param[alias_prefix + "_s"] = wts_s;
    auto wts_zp = node_arg_get_name(*binder[zeropoint_idx].node_arg);
    generic_param[alias_prefix + "_zp"] = wts_zp;
  }

  void io_name_mapper(const std::string& alias_prefix, binder_t& binder,
                      int scale_idx, int zeropoint_idx) {
    auto& generic_param = *initializer_map_.mutable_generic_param();
    auto s = node_arg_get_name(*binder[scale_idx].node_arg);
    generic_param[alias_prefix + "_s"] = s;
    auto zp = node_arg_get_name(*binder[zeropoint_idx].node_arg);
    generic_param[alias_prefix + "_zp"] = zp;
  };

  std::unique_ptr<Rule> create_ht_start_lstm_320_rule(IPass* self) {
    auto builder = vaip_core::PatternBuilder();
    auto ret = std::shared_ptr<vaip_core::Pattern>();
#include "ht_1_2_pattern/ht_start_lstm_320.h.inc"

    auto pattern_ = ret;
    return Rule::create_rule(
        pattern_, [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
          HT_SUBPASS_LOG(1) << "processing ht_start_lstm_320";
          // exlcude input nodes
          // 33:/decoder/rnn/Slice_12_output_0_QuantizeLinear
          // 37:/decoder/rnn/Slice_13_output_0_QuantizeLinear
          // 0:/decoder_embedding/embed/Gather_output_0_QuantizeLinear
          collect_subgraph_nodes(ht_inner_map_.at("ht_000_main"), binder,
                                 {33, 37, 0});

          { // lnorm_0
            // /decoder_embedding/embed/Gather_output_0_DequantizeLinear
            // decoder_embedding.embed.weight_scale
            // decoder_embedding.embed.weight_zero_point
            io_name_mapper("lnorm_0_x", binder, constant_0->get_id(),
                           constant_1->get_id());
            // decoder_embedding.embed_lnorm.weight_DequantizeLinear
            // decoder_embedding.embed_lnorm.weight_quantized
            // decoder_embedding.embed_lnorm.weight_scale
            // decoder_embedding.embed_lnorm.weight_zero_point
            wts_name_mapper("lnorm_0_wts", binder, constant_2->get_id(),
                            constant_3->get_id(), constant_4->get_id());
            // decoder_embedding.embed_lnorm.bias_DequantizeLinear
            // decoder_embedding.embed_lnorm.bias_quantized
            // decoder_embedding.embed_lnorm.bias_quantized_scale
            // decoder.lnorm_layer.bias_quantized_zero_point
            wts_name_mapper("lnorm_0_bias", binder, constant_5->get_id(),
                            constant_6->get_id(), constant_7->get_id());
            // /decoder_embedding/embed_lnorm/Add_1_output_0_QuantizeLinear
            // /decoder_embedding/embed_lnorm/Add_1_output_0_scale
            // /decoder_embedding/embed_lnorm/Add_1_output_0_zero_point
            io_name_mapper("lnorm_0_y", binder, constant_8->get_id(),
                           constant_9->get_id());
          }
          { // sigmod
            // /decoder_embedding/embed_lnorm/Add_1_output_0_DequantizeLinear
            // /decoder_embedding/embed_lnorm/Add_1_output_0_scale
            // /decoder_embedding/embed_lnorm/Add_1_output_0_zero_point
            io_name_mapper("Sigmoid_input_0", binder, constant_10->get_id(),
                           constant_11->get_id());
            // /decoder_embedding/sigmoid/Sigmoid_output_0_QuantizeLinear
            // /decoder_embedding/sigmoid/Sigmoid_output_0_scale
            // /decoder_embedding/sigmoid/Sigmoid_output_0_zero_point
            io_name_mapper("Sigmoid_output_0", binder, constant_12->get_id(),
                           constant_13->get_id());
          }
          { // lstm320

            // /decoder/rnn/Transpose_output_0_DequantizeLinear
            // /decoder_embedding/sigmoid/Sigmoid_output_0_scale
            // /decoder_embedding/sigmoid/Sigmoid_output_0_zero_point
            io_name_mapper("lstm320_x", binder, constant_18->get_id(),
                           constant_19->get_id());

            // /decoder/rnn/Slice_12_output_0_DequantizeLinear
            // x: h0_scale /decoder/rnn/Slice_12_output_0_zero_point
            // y: lstm320_init_h
            io_name_mapper("lstm320_init_h", binder, constant_20->get_id(),
                           constant_21->get_id());

            // /decoder/rnn/Slice_13_output_0_DequantizeLinear
            // x: /decoder/rnn/Slice_13_output_0_scale
            // /decoder/rnn/Slice_13_output_0_zero_point y: lstm320_init_c
            io_name_mapper("lstm320_init_c", binder, constant_22->get_id(),
                           constant_23->get_id());

            // /decoder/rnn/Unsqueeze_output_0_DequantizeLinear
            wts_name_mapper("lstm320_x_wts", binder, constant_24->get_id(),
                            constant_25->get_id(), constant_26->get_id());

            // /decoder/rnn/Unsqueeze_1_output_0_DequantizeLinear
            wts_name_mapper("lstm320_h_wts", binder, constant_30->get_id(),
                            constant_31->get_id(), constant_32->get_id());

            // /decoder/rnn/Unsqueeze_2_output_0_DequantizeLinear
            wts_name_mapper("lstm320_bias", binder, constant_27->get_id(),
                            constant_28->get_id(), constant_29->get_id());
          }

          return false; // false means not changing the graph
        });
  }

  std::unique_ptr<Rule> create_ht_lstm_1024_end_rule(IPass* self) {
    auto builder = vaip_core::PatternBuilder();
    auto ret = std::shared_ptr<vaip_core::Pattern>();
#include "ht_1_2_pattern/ht_lstm_1024_end.h.inc"

    auto pattern_ = ret;
    return Rule::create_rule(
        pattern_, [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
          HT_SUBPASS_LOG(1) << "processing ht_lstm_1024_end";
          // exclude input nodes
          // 7:/decoder/rnn/Slice_26_output_0_QuantizeLinear
          // 11:/decoder/rnn/Slice_27_output_0_QuantizeLinear
          collect_subgraph_nodes(ht_inner_map_.at("ht_000_main"), binder,
                                 {7, 11});

          { // lstm1024
            // /decoder/rnn/Slice_26_output_0_DequantizeLinear
            // x: /decoder/rnn/Slice_26_output_0_scale
            // /decoder/rnn/Slice_26_output_0_zero_point y: lstm1024_init_h
            io_name_mapper("lstm1024_init_h", binder, constant_4->get_id(),
                           constant_5->get_id());

            // /decoder/rnn/Slice_27_output_0_DequantizeLinear
            // x: /decoder/rnn/Slice_27_output_0_scale
            // /decoder/rnn/Slice_27_output_0_zero_point y: lstm1024_init_c
            io_name_mapper("lstm1024_init_c", binder, constant_6->get_id(),
                           constant_7->get_id());

            // /decoder/rnn/Squeeze_output_0_DequantizeLinear
            // /decoder/rnn/LSTM_output_1_scale
            // /decoder/rnn/LSTM_output_0_zero_point
            io_name_mapper("lstm1024_x", binder, constant_2->get_id(),
                           constant_3->get_id());

            // /decoder/rnn/Unsqueeze_3_output_0_DequantizeLinear
            wts_name_mapper("lstm1024_x_wts", binder, constant_14->get_id(),
                            constant_15->get_id(), constant_16->get_id());

            // /decoder/rnn/Unsqueeze_4_output_0_DequantizeLinear
            wts_name_mapper("lstm1024_h_wts", binder, constant_11->get_id(),
                            constant_12->get_id(), constant_13->get_id());

            // /decoder/rnn/Unsqueeze_5_output_0_DequantizeLinear
            wts_name_mapper("lstm1024_bias", binder, constant_8->get_id(),
                            constant_9->get_id(), constant_10->get_id());
          }
          { // lnorm_1
            // /decoder/rnn/Transpose_1_output_0_DequantizeLinear
            // /decoder/rnn/LSTM_1_output_0_scale
            // /decoder/rnn/LSTM_1_output_0_zero_point
            io_name_mapper("lnorm_1_x", binder, constant_28->get_id(),
                           constant_29->get_id());
            // decoder.lnorm_layer.weight_DequantizeLinear
            // decoder.lnorm_layer.weight_quantized
            // decoder.lnorm_layer.weight_scale
            // decoder.lnorm_layer.weight_zero_point
            wts_name_mapper("lnorm_1_wts", binder, constant_30->get_id(),
                            constant_31->get_id(), constant_32->get_id());
            // decoder.lnorm_layer.bias_DequantizeLinear
            // decoder.lnorm_layer.bias_quantized
            // decoder.lnorm_layer.bias_quantized_scale
            // decoder.lnorm_layer.bias_quantized_zero_point
            wts_name_mapper("lnorm_1_bias", binder, constant_33->get_id(),
                            constant_34->get_id(), constant_35->get_id());
            // /decoder/lnorm_layer/Add_1_output_0_DequantizeLinear
            // /decoder/lnorm_layer/Add_1_output_0_scale
            // /decoder/lnorm_layer/Add_1_output_0_zero_point
            io_name_mapper("lnorm_1_y", binder, constant_38->get_id(),
                           constant_39->get_id());
          }
          { // /lin_dec/fc/MatMul
            // /decoder/lnorm_layer/Add_1_output_0_DequantizeLinear_Output
            // /decoder/lnorm_layer/Add_1_output_0_scale
            // /decoder/lnorm_layer/Add_1_output_0_zero_point
            io_name_mapper("lin_dec_fc_matmul_in_1", binder,
                           constant_38->get_id(), constant_39->get_id());
            // /lin_dec/fc/Transpose_output_0_DequantizeLinear
            // /lin_dec/fc/Transpose_output_0_quantized
            // /lin_dec/fc/Transpose_output_0_scale
            // /lin_dec/fc/Transpose_output_0_zero_point
            wts_name_mapper("lin_dec_fc_matmul_in_2", binder,
                            constant_40->get_id(), constant_41->get_id(),
                            constant_42->get_id());
            // /lin_dec/fc/MatMul_output_0_DequantizeLinear
            // /lin_dec/fc/MatMul_output_0_scale
            // /lin_dec/fc/MatMul_output_0_zero_point
            io_name_mapper("lin_dec_fc_matmul_out", binder,
                           constant_45->get_id(), constant_46->get_id());
          }
          {
            // /lin_dec/fc/MatMul_output_0_DequantizeLinear
            // /lin_dec/fc/MatMul_output_0_scale
            // /lin_dec/fc/MatMul_output_0_zero_point
            io_name_mapper("lin_dec_fc_add_in", binder, constant_45->get_id(),
                           constant_46->get_id());
            // joint_network.lin_dec.fc.bias_DequantizeLinear
            // joint_network.lin_dec.fc.bias_quantized
            // joint_network.lin_dec.fc.bias_scale
            // joint_network.lin_dec.fc.bias_zero_point
            wts_name_mapper("joint_network_lin_dec_fc_bias", binder,
                            constant_47->get_id(), constant_48->get_id(),
                            constant_49->get_id());
            // /lin_dec/fc/Add_output_0_scale
            // /lin_dec/fc/Add_output_0_zero_point
            // /lin_dec/fc/Add_output_0_QuantizeLinear_Output
            io_name_mapper("lin_dec_fc_add_out", binder, constant_50->get_id(),
                           constant_51->get_id());
          }
          { // lnorm_2
            // /lin_dec/fc/Add_output_0_DequantizeLinear
            // /lin_dec/fc/Add_output_0_scale
            // /lin_dec/fc/Add_output_0_zero_point
            io_name_mapper("lnorm_2_x", binder, constant_52->get_id(),
                           constant_53->get_id());
            // joint_network.lin_dec.Lnorm.weight_DequantizeLinear
            // joint_network.lin_dec.Lnorm.weight_quantized
            // joint_network.lin_dec.Lnorm.weight_scale
            // joint_network.lin_dec.Lnorm.weight_zero_point
            wts_name_mapper("lnorm_2_wts", binder, constant_54->get_id(),
                            constant_55->get_id(), constant_56->get_id());
            // joint_network.lin_dec.Lnorm.bias_DequantizeLinear
            // joint_network.lin_dec.Lnorm.bias_quantized
            // joint_network.lin_dec.Lnorm.bias_quantized_scale
            // decoder.lnorm_layer.bias_quantized_zero_point
            wts_name_mapper("lnorm_2_bias", binder, constant_57->get_id(),
                            constant_58->get_id(), constant_59->get_id());
            // /hidden_DequantizeLinear
            // hidden_scale
            // hidden_zero_point
            io_name_mapper("lnorm_2_y", binder, constant_60->get_id(),
                           constant_61->get_id());
          }
          return false; // false means not changing the graph
        });
  }
  std::unique_ptr<Rule> create_ht_concat_rule(IPass* self) {
    auto builder = vaip_core::PatternBuilder();
    auto ret = std::shared_ptr<vaip_core::Pattern>();
#include "ht_1_2_pattern/ht_concat.h.inc"

    auto pattern_ = ret;
    return Rule::create_rule(
        pattern_, [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
          HT_SUBPASS_LOG(1) << "processing ht_concat";
          {
            std::string output_name = node_arg_get_name(
                *binder[DequantizeLinear_2->get_id()].node_arg);
            if (output_name == "h1") {
              // /decoder/rnn/LSTM_output_1_DequantizeLinear
              // /decoder/rnn/LSTM_output_1_scale
              // /decoder/rnn/LSTM_output_1_zero_point
              io_name_mapper("lstm320_output_1", binder, constant_0->get_id(),
                             constant_1->get_id());

              // /decoder/rnn/LSTM_1_output_1_DequantizeLinear
              // /decoder/rnn/LSTM_1_output_0_scale
              // /decoder/rnn/LSTM_1_output_1_zero_point
              io_name_mapper("lstm1024_output_1", binder, constant_2->get_id(),
                             constant_3->get_id());

              // h1_DequantizeLinear
              // /decoder/rnn/LSTM_output_1_scale
              // h1_zero_point
              io_name_mapper("h1", binder, constant_4->get_id(),
                             constant_5->get_id());
            } else if (output_name == "c1") {
              // /decoder/rnn/LSTM_output_2_DequantizeLinear
              // /decoder/rnn/LSTM_output_2_scale
              // /decoder/rnn/LSTM_output_2_zero_point
              io_name_mapper("lstm320_output_2", binder, constant_0->get_id(),
                             constant_1->get_id());

              // /decoder/rnn/LSTM_1_output_2_DequantizeLinear
              // /decoder/rnn/LSTM_1_output_2_scale
              // /decoder/rnn/LSTM_1_output_2_zero_point
              io_name_mapper("lstm1024_output_2", binder, constant_2->get_id(),
                             constant_3->get_id());

              // c1_DequantizeLinear
              // c1_scale c1_zero_point
              io_name_mapper("c1", binder, constant_4->get_id(),
                             constant_5->get_id());
            }
          }
          // excludes input node
          // 0:/decoder/rnn/LSTM_output_1_QuantizeLinear
          // 1:/decoder/rnn/LSTM_1_output_1_QuantizeLinear
          collect_subgraph_nodes(ht_inner_map_.at("ht_002_concat"), binder,
                                 {0, 4});
          // patch for main lstm subgraph, add missing lstm_out_1, lstm_1_out_1
          // Q node due to we can't describe multi-output pattern
          ht_inner_map_.at("ht_000_main")
              .insert(VAIP_ORT_API(node_get_name)(*binder[0].node));
          ht_inner_map_.at("ht_000_main")
              .insert(VAIP_ORT_API(node_get_name)(*binder[4].node));

          return false; // false means not changing the graph
        });
  }

  std::unique_ptr<Rule> create_ht_slice_rule(IPass* self) {
    auto builder = vaip_core::PatternBuilder();
    auto ret = std::shared_ptr<vaip_core::Pattern>();
#include "ht_1_2_pattern/ht_slice.h.inc"

    auto pattern_ = ret;
    return Rule::create_rule(
        pattern_, [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
          HT_SUBPASS_LOG(1) << "processing ht_slice";

          io_name_mapper(slice_name_map.at(slice_sg_idx_), binder,
                         constant_7->get_id(), constant_8->get_id());
          // FIXME: is this order safe?
          slice_sg_idx_++;
          collect_subgraph_nodes(ht_inner_map_.at("ht_001_slice"), binder);
          return false; // false means not changing the graph
        });
  }

  std::map<std::string, std::vector<std::string>> get_partitioned_nodes() {
    std::map<std::string, std::vector<std::string>> res;
    for (auto iter : ht_inner_map_) {
      res[iter.first] =
          std::vector<std::string>(iter.second.begin(), iter.second.end());
    }
    return res;
  }
  IPass& self_;
  MetaDefProto& initializer_map_;
  size_t slice_sg_idx_ = 0;
  const std::map<size_t, std::string> slice_name_map = {
      {0, "Slice_12_output_0"},
      {1, "Slice_13_output_0"},
      {2, "Slice_26_output_0"},
      {3, "Slice_27_output_0"}};
  std::map<std::string, std::set<std::string>> ht_inner_map_ = {
      {"ht_000_main", {}}, {"ht_001_slice", {}}, {"ht_002_concat", {}}};
};

} // namespace vaip_vaiml_subgraph_processor

#undef HT_SUBPASS_LOG
