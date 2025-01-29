/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */
#pragma once
#include <fstream>
#include <glog/logging.h>
#include <iostream>

#include "gt_1_3_pattern/main_block_patch/concat_wildcard.h.inc"
#include "subpass_util.h"
#include "vaip/vaip.hpp"
#include "vitis/ai/env_config.hpp"
#include <algorithm>
#define CONCAT_WILDCARD_RULE(N)                                                \
  std::unique_ptr<Rule> create_concat_wildcard_rule_##N(IPass* self) {         \
    using namespace concat_##N;                                                \
    auto pattern_ = ret;                                                       \
    return Rule::create_rule(                                                  \
        pattern_, [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {   \
          GT_SUBPASS_LOG(1) << "processing gt concat_wildcard_" << #N;         \
          collect_subgraph_nodes(gt_inner_map_.at("gt_002_transformer"),       \
                                 binder, {});                                  \
          return false;                                                        \
        });                                                                    \
  }
using namespace vaip_core;
DEF_ENV_PARAM(DEBUG_GT_SUBPASS, "0")
#define GT_SUBPASS_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_GT_SUBPASS) >= n)
namespace vaip_vaiml_subgraph_processor {
struct GT_initializer_mapping_pass {
  GT_initializer_mapping_pass(IPass& self, MetaDefProto& initializer_map)
      : self_{self}, initializer_map_(initializer_map) {}
  void process(Graph& graph) {
    create_transformer_rule(&self_)->apply(&graph);
    create_tail_lin_enc_rule(&self_)->apply(&graph);
    create_cache_frame_rule(&self_)->apply(&graph);
    create_gt_front_rule(&self_)->apply(&graph);
    create_gt_inp_reshape_gather_unsqueeze_rule(&self_)->apply(&graph);
    create_gt_mask_slice_rule(&self_)->apply(&graph);
    create_gt_concat_36_rule(&self_)->apply(&graph);
    create_gt_concat_32_rule(&self_)->apply(&graph);
    { // adaptive rules for dynamic number of transformer block
      create_concat_wildcard_rule_40(&self_)->apply(&graph);
      create_concat_wildcard_rule_39(&self_)->apply(&graph);
      create_concat_wildcard_rule_38(&self_)->apply(&graph);
      create_concat_wildcard_rule_37(&self_)->apply(&graph);
      create_concat_wildcard_rule_36(&self_)->apply(&graph);
      create_concat_wildcard_rule_35(&self_)->apply(&graph);
      create_concat_wildcard_rule_34(&self_)->apply(&graph);
      create_concat_wildcard_rule_33(&self_)->apply(&graph);
      create_concat_wildcard_rule_32(&self_)->apply(&graph);
      create_concat_wildcard_rule_31(&self_)->apply(&graph);
      create_concat_wildcard_rule_30(&self_)->apply(&graph);
      create_concat_wildcard_rule_29(&self_)->apply(&graph);
      create_concat_wildcard_rule_28(&self_)->apply(&graph);
    }

    create_gt_oup_lid_rule(&self_)->apply(&graph);
    auto& generic_param = *initializer_map_.mutable_generic_param();
    generic_param["__TRANSFORMER_NUM__"] = std::to_string(transformer_cnt);
    generic_param["__OUP_LID_IDX__"] = std::to_string(oup_lid_idx_);

    GT_SUBPASS_LOG(1) << "__TRANSFORMER_NUM__ "
                      << generic_param["__TRANSFORMER_NUM__"];
    GT_SUBPASS_LOG(1) << "__OUP_LID_IDX__ " << generic_param["__OUP_LID_IDX__"];
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
  std::unique_ptr<Rule> create_transformer_rule(IPass* self) {
    auto builder = vaip_core::PatternBuilder();
    auto ret = std::shared_ptr<vaip_core::Pattern>();
#include "gt_1_3_pattern/transformer.h.inc"

    auto pattern_ = ret;
    return Rule::create_rule(
        pattern_, [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
          GT_SUBPASS_LOG(1)
              << "processing " << transformer_cnt << " transformer block";
          { // /norm1/Add_1_output_0
            io_name_mapper(str_fmt("tf_%d_ln_0_in", transformer_cnt), binder,
                           constant_48->get_id(), constant_49->get_id());
            wts_name_mapper(str_fmt("tf_%d_ln_0_wts", transformer_cnt), binder,
                            constant_45->get_id(), constant_46->get_id(),
                            constant_47->get_id());
            wts_name_mapper(str_fmt("tf_%d_ln_0_bias", transformer_cnt), binder,
                            constant_42->get_id(), constant_43->get_id(),
                            constant_44->get_id());
            io_name_mapper(str_fmt("tf_%d_ln_0_out", transformer_cnt), binder,
                           constant_50->get_id(), constant_51->get_id());
          }
          { // _v_3393
            io_name_mapper(str_fmt("tf_%d_kqv_mm_in", transformer_cnt), binder,
                           constant_52->get_id(), constant_53->get_id());
            wts_name_mapper(str_fmt("tf_%d_kqv_mm_wts", transformer_cnt),
                            binder, constant_39->get_id(),
                            constant_40->get_id(), constant_41->get_id());
            // encoder.encoders.0.self_attn.linear_k
            wts_name_mapper(str_fmt("tf_%d_kqv_mm_bias_k", transformer_cnt),
                            binder, constant_36->get_id(),
                            constant_37->get_id(), constant_38->get_id());
            wts_name_mapper(str_fmt("tf_%d_kqv_mm_bias_q", transformer_cnt),
                            binder, constant_87->get_id(),
                            constant_88->get_id(), constant_89->get_id());
            wts_name_mapper(str_fmt("tf_%d_kqv_mm_bias_v", transformer_cnt),
                            binder, constant_187->get_id(),
                            constant_188->get_id(), constant_189->get_id());
            // /linear_k/Add_output_0_QuantizeLinear_Output
            io_name_mapper(str_fmt("tf_%d_kqv_mm_out_k", transformer_cnt),
                           binder, constant_63->get_id(),
                           constant_64->get_id());
            io_name_mapper(str_fmt("tf_%d_kqv_mm_out_q", transformer_cnt),
                           binder, constant_94->get_id(),
                           constant_95->get_id());
            io_name_mapper(str_fmt("tf_%d_kqv_mm_out_v", transformer_cnt),
                           binder, constant_194->get_id(),
                           constant_195->get_id());
          }
          { // /Mul_output_0
            io_name_mapper(str_fmt("tf_%d_q_mul_0_in", transformer_cnt), binder,
                           constant_105->get_id(), constant_106->get_id());
            wts_name_mapper(str_fmt("tf_%d_q_mul_0_wts", transformer_cnt),
                            binder, constant_84->get_id(),
                            constant_85->get_id(), constant_86->get_id());
            io_name_mapper(str_fmt("tf_%d_q_mul_0_out", transformer_cnt),
                           binder, constant_107->get_id(),
                           constant_108->get_id());
          }
          {
            // kv cache
            // v /Unsqueeze_25_output_0_DequantizeLinear
            io_name_mapper(str_fmt("tf_%d_v_cache_in", transformer_cnt), binder,
                           constant_185->get_id(), constant_186->get_id());
            // v /Slice_3_output_0_QuantizeLinear
            io_name_mapper(str_fmt("tf_%d_v_cache_out", transformer_cnt),
                           binder, constant_272->get_id(),
                           constant_273->get_id());
            // k /Unsqueeze_26_output_0_DequantizeLinear
            io_name_mapper(str_fmt("tf_%d_k_cache_in", transformer_cnt), binder,
                           constant_34->get_id(), constant_35->get_id());
            // k /Slice_2_output_0_QuantizeLinear
            io_name_mapper(str_fmt("tf_%d_k_cache_out", transformer_cnt),
                           binder, constant_280->get_id(),
                           constant_281->get_id());
          }
          {
            // q bmm_0 /MatMul_1_output_0
            // /Transpose_5_output_0_QuantizeLinear_Output
            io_name_mapper(str_fmt("tf_%d_q_bmm_0_in_0", transformer_cnt),
                           binder, constant_127->get_id(),
                           constant_128->get_id());
            wts_name_mapper(str_fmt("tf_%d_q_bmm_0_in_1", transformer_cnt),
                            binder, constant_115->get_id(),
                            constant_116->get_id(), constant_117->get_id());
            io_name_mapper(str_fmt("tf_%d_q_bmm_0_out", transformer_cnt),
                           binder, constant_129->get_id(),
                           constant_130->get_id());
          }
          {
            // k bmm
            // /Matmul_output_0
            io_name_mapper(str_fmt("tf_%d_k_bmm_in_0", transformer_cnt), binder,
                           constant_109->get_id(), constant_110->get_id());
            io_name_mapper(str_fmt("tf_%d_k_bmm_in_1", transformer_cnt), binder,
                           constant_82->get_id(), constant_83->get_id());
            io_name_mapper(str_fmt("tf_%d_k_bmm_out", transformer_cnt), binder,
                           constant_111->get_id(), constant_112->get_id());
          }
          {
            // q add_0 /Add_output_0
            io_name_mapper(str_fmt("tf_%d_q_add_0_in_0", transformer_cnt),
                           binder, constant_113->get_id(),
                           constant_114->get_id()); // k bmm
            io_name_mapper(str_fmt("tf_%d_q_add_0_in_1", transformer_cnt),
                           binder, constant_140->get_id(),
                           constant_141->get_id()); // q bmm
            io_name_mapper(str_fmt("tf_%d_q_add_0_out", transformer_cnt),
                           binder, constant_142->get_id(),
                           constant_143->get_id());
          }
          {
            // q mul_1, before reducemin /Mul_3_output_0
            io_name_mapper(str_fmt("tf_%d_q_mul_1_in_0", transformer_cnt),
                           binder, constant_144->get_id(),
                           constant_145->get_id());
            io_name_mapper(str_fmt("tf_%d_q_mul_1_in_1", transformer_cnt),
                           binder, constant_30->get_id(),
                           constant_31->get_id());
            io_name_mapper(str_fmt("tf_%d_q_mul_1_out", transformer_cnt),
                           binder, constant_146->get_id(),
                           constant_147->get_id());
          }
          {
            // q mul_2, after reducemin /Mul_4_output_0
            // reuse out s/zp of /Mul_3_output_0
            // io_name_mapper(str_fmt("tf_%d_q_mul_2_in_0", transformer_cnt),
            //                constant_167->get_id(), constant_168->get_id());
            io_name_mapper(str_fmt("tf_%d_q_mul_2_in_1", transformer_cnt),
                           binder, constant_26->get_id(),
                           constant_27->get_id());
            io_name_mapper(str_fmt("tf_%d_q_mul_2_out", transformer_cnt),
                           binder, constant_169->get_id(),
                           constant_170->get_id());
          }
          {
            // q add_1 /Add_2_output_0
            io_name_mapper(str_fmt("tf_%d_q_add_1_in_0", transformer_cnt),
                           binder, constant_173->get_id(),
                           constant_174->get_id());
            io_name_mapper(str_fmt("tf_%d_q_add_1_in_1", transformer_cnt),
                           binder, constant_171->get_id(),
                           constant_172->get_id());
            io_name_mapper(str_fmt("tf_%d_q_add_1_out", transformer_cnt),
                           binder, constant_175->get_id(),
                           constant_176->get_id());
          }
          {
            // /Softmax
            io_name_mapper(str_fmt("tf_%d_q_softmax_in", transformer_cnt),
                           binder, constant_177->get_id(),
                           constant_178->get_id());
            io_name_mapper(str_fmt("tf_%d_q_softmax_out", transformer_cnt),
                           binder, constant_179->get_id(),
                           constant_180->get_id());
          }
          {
            // q bmm_1 /MatMul_2_output_0
            io_name_mapper(str_fmt("tf_%d_q_bmm_1_in_0", transformer_cnt),
                           binder, constant_181->get_id(),
                           constant_182->get_id());
            io_name_mapper(str_fmt("tf_%d_q_bmm_1_in_1", transformer_cnt),
                           binder, constant_209->get_id(),
                           constant_210->get_id());
            io_name_mapper(str_fmt("tf_%d_q_bmm_1_out", transformer_cnt),
                           binder, constant_211->get_id(),
                           constant_212->get_id());
          }
          { // linear_out mm+biasadd
            io_name_mapper(str_fmt("tf_%d_linear_out_mmb_in", transformer_cnt),
                           binder, constant_222->get_id(),
                           constant_223->get_id());
            wts_name_mapper(
                str_fmt("tf_%d_linear_out_mmb_wts", transformer_cnt), binder,
                constant_21->get_id(), constant_22->get_id(),
                constant_23->get_id());
            wts_name_mapper(
                str_fmt("tf_%d_linear_out_mmb_bias", transformer_cnt), binder,
                constant_18->get_id(), constant_19->get_id(),
                constant_20->get_id());
            io_name_mapper(str_fmt("tf_%d_linear_out_mmb_out", transformer_cnt),
                           binder, constant_228->get_id(),
                           constant_229->get_id());
          }
          { // linear_out add, /Add_3
            io_name_mapper(
                str_fmt("tf_%d_linear_out_add_in_0", transformer_cnt), binder,
                constant_230->get_id(), constant_231->get_id());
            io_name_mapper(
                str_fmt("tf_%d_linear_out_add_in_1", transformer_cnt), binder,
                constant_232->get_id(), constant_233->get_id());
            io_name_mapper(str_fmt("tf_%d_linear_out_add_out", transformer_cnt),
                           binder, constant_234->get_id(),
                           constant_235->get_id());
          }
          { // /norm2/Add_1_output_0
            io_name_mapper(str_fmt("tf_%d_ln_1_in", transformer_cnt), binder,
                           constant_236->get_id(), constant_237->get_id());
            wts_name_mapper(str_fmt("tf_%d_ln_1_wts", transformer_cnt), binder,
                            constant_15->get_id(), constant_16->get_id(),
                            constant_17->get_id());
            wts_name_mapper(str_fmt("tf_%d_ln_1_bias", transformer_cnt), binder,
                            constant_12->get_id(), constant_13->get_id(),
                            constant_14->get_id());
            io_name_mapper(str_fmt("tf_%d_ln_1_out", transformer_cnt), binder,
                           constant_238->get_id(), constant_239->get_id());
          }
          { // feed_forward 0  /feed_forward/w_1/MatMul
            io_name_mapper(
                str_fmt("tf_%d_feed_forward_0_mmb_in", transformer_cnt), binder,
                constant_240->get_id(), constant_241->get_id());
            wts_name_mapper(
                str_fmt("tf_%d_feed_forward_0_mmb_wts", transformer_cnt),
                binder, constant_9->get_id(), constant_10->get_id(),
                constant_11->get_id());
            wts_name_mapper(
                str_fmt("tf_%d_feed_forward_0_mmb_bias", transformer_cnt),
                binder, constant_6->get_id(), constant_7->get_id(),
                constant_8->get_id());
            io_name_mapper(
                str_fmt("tf_%d_feed_forward_0_mmb_out", transformer_cnt),
                binder, constant_246->get_id(), constant_247->get_id());
          }
          { // feed_forward 0  /feed_forward/w_2/MatMul
            io_name_mapper(
                str_fmt("tf_%d_feed_forward_1_mmb_in", transformer_cnt), binder,
                constant_252->get_id(), constant_253->get_id());
            wts_name_mapper(
                str_fmt("tf_%d_feed_forward_1_mmb_wts", transformer_cnt),
                binder, constant_3->get_id(), constant_4->get_id(),
                constant_5->get_id());
            wts_name_mapper(
                str_fmt("tf_%d_feed_forward_1_mmb_bias", transformer_cnt),
                binder, constant_0->get_id(), constant_1->get_id(),
                constant_2->get_id());
            io_name_mapper(
                str_fmt("tf_%d_feed_forward_1_mmb_out", transformer_cnt),
                binder, constant_258->get_id(), constant_259->get_id());
          }
          { // feed_forward add, /Add_4
            io_name_mapper(
                str_fmt("tf_%d_feed_forward_add_in_0", transformer_cnt), binder,
                constant_260->get_id(), constant_261->get_id());
            io_name_mapper(
                str_fmt("tf_%d_feed_forward_add_in_1", transformer_cnt), binder,
                constant_262->get_id(), constant_263->get_id());
            io_name_mapper(
                str_fmt("tf_%d_feed_forward_add_out", transformer_cnt), binder,
                constant_264->get_id(), constant_265->get_id());
          }
          std::set<int> exlcude_nodes = {};
          if (transformer_cnt == 0) {
            /// out/Add_output_0_QuantizeLinear_Output
            exlcude_nodes.insert(69);
          }
          collect_subgraph_nodes(gt_inner_map_.at("gt_002_transformer"), binder,
                                 exlcude_nodes);
          transformer_cnt++;
          transformer_final_add_nodes_.push_back(
              VAIP_ORT_API(node_get_name)(*binder[Add_9->get_id()].node));
          return false; // false means not changing the graph
        });
  }
  std::unique_ptr<Rule> create_tail_lin_enc_rule(IPass* self) {
    auto builder = vaip_core::PatternBuilder();
    auto ret = std::shared_ptr<vaip_core::Pattern>();
#include "gt_1_3_pattern/ln_mm_add_ln.h.inc"

    auto pattern_ = ret;
    return Rule::create_rule(
        pattern_, [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
          GT_SUBPASS_LOG(1) << "processing tail lin_enc";
          { // /after_norm/Add_1_output_0
            io_name_mapper("lin_enc_ln_0_in", binder, constant_18->get_id(),
                           constant_19->get_id());
            wts_name_mapper("lin_enc_ln_0_wts", binder, constant_15->get_id(),
                            constant_16->get_id(), constant_17->get_id());
            wts_name_mapper("lin_enc_ln_0_bias", binder, constant_12->get_id(),
                            constant_13->get_id(), constant_14->get_id());
            io_name_mapper("lin_enc_ln_0_out", binder, constant_20->get_id(),
                           constant_21->get_id());
          }
          { // /lin_enc/fc/MatMul_output_0"
            io_name_mapper("lin_enc_mmb_in", binder, constant_22->get_id(),
                           constant_23->get_id());
            wts_name_mapper("lin_enc_mmb_wts", binder, constant_9->get_id(),
                            constant_10->get_id(), constant_11->get_id());
            wts_name_mapper("lin_enc_mmb_bias", binder, constant_6->get_id(),
                            constant_7->get_id(), constant_8->get_id());
            io_name_mapper("lin_enc_mmb_out", binder, constant_28->get_id(),
                           constant_29->get_id());
          }
          { // hidden_state_QuantizeLinear_Input
            io_name_mapper("lin_enc_ln_1_in", binder, constant_30->get_id(),
                           constant_31->get_id());
            wts_name_mapper("lin_enc_ln_1_wts", binder, constant_3->get_id(),
                            constant_4->get_id(), constant_5->get_id());
            wts_name_mapper("lin_enc_ln_1_bias", binder, constant_0->get_id(),
                            constant_1->get_id(), constant_2->get_id());
            io_name_mapper("lin_enc_ln_1_out", binder, constant_32->get_id(),
                           constant_33->get_id());
          }
          // 24:/Add_179_output_0_QuantizeLinear_Output
          collect_subgraph_nodes(gt_inner_map_.at("gt_003_tail"), binder, {24});

          return false; // false means not changing the graph
        });
  }
  std::unique_ptr<Rule> create_cache_frame_rule(IPass* self) {
    auto builder = vaip_core::PatternBuilder();
    auto ret = std::shared_ptr<vaip_core::Pattern>();
#include "gt_1_3_pattern/cache_frame_slice.h.inc"

    auto pattern_ = ret;
    return Rule::create_rule(
        pattern_, [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
          GT_SUBPASS_LOG(1) << "processing cache_frame_slice";
          { // cache_frames_DequantizeLinear_Output
            io_name_mapper("cache_frame", binder, constant_5->get_id(),
                           constant_6->get_id());
          }
          // 4:cache_frames_QuantizeLinear_Output
          // this pattern includes front sub, which need to be removed from the
          // subgraph nodes
          // 3:encoder_embedding.global_mean_DequantizeLinear_Output, sub wts
          // 7:cache_frames_DequantizeLinear_Output/duplicated
          // 8:/encoder_embedding/Sub_output_0
          collect_subgraph_nodes(gt_inner_map_.at("gt_004_cache_frame_slice"),
                                 binder, {3, 4, 7, 8});
          return false; // false means not changing the graph
        });
  }
  std::unique_ptr<Rule> create_gt_front_rule(IPass* self) {
    auto builder = vaip_core::PatternBuilder();
    auto ret = std::shared_ptr<vaip_core::Pattern>();
#include "gt_1_3_pattern/gt_front.h.inc"

    auto pattern_ = ret;
    return Rule::create_rule(
        pattern_, [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
          GT_SUBPASS_LOG(1) << "processing gt front conv";
          {
            // sub
            io_name_mapper("front_sub_in", binder, constant_24->get_id(),
                           constant_25->get_id());
            wts_name_mapper("front_sub_wts", binder, constant_21->get_id(),
                            constant_22->get_id(), constant_23->get_id());
            io_name_mapper("front_sub_out", binder, constant_26->get_id(),
                           constant_27->get_id());
            // mul
            io_name_mapper("front_mul_in", binder, constant_26->get_id(),
                           constant_27->get_id());
            wts_name_mapper("front_mul_wts", binder, constant_18->get_id(),
                            constant_19->get_id(), constant_20->get_id());
            io_name_mapper("front_mul_out", binder, constant_30->get_id(),
                           constant_31->get_id());
            // CONV1
            io_name_mapper("front_conv1_in", binder, constant_30->get_id(),
                           constant_31->get_id());
            wts_name_mapper("front_conv1_wts", binder, constant_15->get_id(),
                            constant_16->get_id(), constant_17->get_id());
            wts_name_mapper(
                "front_conv1_bias", binder, constant_12->get_id(),
                constant_13->get_id(),
                constant_14->get_id()); // the zero_point comes from norm's data
            io_name_mapper("front_conv1_out", binder, constant_39->get_id(),
                           constant_40->get_id());
            // CONV2
            io_name_mapper("front_conv2_in", binder, constant_39->get_id(),
                           constant_40->get_id());
            wts_name_mapper("front_conv2_wts", binder, constant_9->get_id(),
                            constant_10->get_id(), constant_11->get_id());
            wts_name_mapper(
                "front_conv2_bias", binder, constant_6->get_id(),
                constant_7->get_id(),
                constant_8->get_id()); // the zero_point comes from norm's data
            io_name_mapper("front_conv2_out", binder, constant_47->get_id(),
                           constant_48->get_id());
            // /out/MatMul_output_0
            io_name_mapper("front_mmb_in", binder, constant_62->get_id(),
                           constant_63->get_id());
            wts_name_mapper("front_mmb_wts", binder, constant_3->get_id(),
                            constant_4->get_id(), constant_5->get_id());
            wts_name_mapper("front_mmb_bias", binder, constant_0->get_id(),
                            constant_1->get_id(), constant_2->get_id());
            io_name_mapper("front_mmb_out", binder, constant_68->get_id(),
                           constant_69->get_id());
          }
          // 3:encoder.embed.out.bias_DequantizeLinear_Output
          // 7:/out/Transpose_output_0_DequantizeLinear_Output
          // 100:/Reshape_2_output_0_DequantizeLinear_Output
          // 101:/out/MatMul_output_0
          // 104:/out/MatMul_output_0_QuantizeLinear_Output
          // 107:/out/MatMul_output_0_DequantizeLinear_Output
          // 108:/out/Add_output_0
          // 111:/out/Add_output_0_QuantizeLinear_Output
          std::set<int> front_mm_bias_node_idx = {3,   7,   100, 101,
                                                  104, 107, 108, 111};
          for (auto i : front_mm_bias_node_idx) {
            gt_inner_map_.at("gt_001_front_mm")
                .insert(VAIP_ORT_API(node_get_name)(*binder[i].node));
          }
          // 32:cache_frames_QuantizeLinear_Output
          std::set<int> conv_exclude_idx = front_mm_bias_node_idx;
          conv_exclude_idx.insert(32);
          collect_subgraph_nodes(gt_inner_map_.at("gt_000_front_conv"), binder,
                                 conv_exclude_idx);
          return false; // false means not changing the graph
        });
  }

  std::unique_ptr<Rule>
  create_gt_inp_reshape_gather_unsqueeze_rule(IPass* self) {
    auto builder = vaip_core::PatternBuilder();
    auto ret = std::shared_ptr<vaip_core::Pattern>();
#include "gt_1_3_pattern/main_block_patch/inp_reshape_gather_unsqueeze.h.inc"
    auto pattern_ = ret;
    return Rule::create_rule(
        pattern_, [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
          GT_SUBPASS_LOG(1) << "processing gt inp gather " << gather_cnt++;
          collect_subgraph_nodes(gt_inner_map_.at("gt_002_transformer"), binder,
                                 {});
          return false; // false means not changing the graph
        });
  }
  std::unique_ptr<Rule> create_gt_mask_slice_rule(IPass* self) {
    auto builder = vaip_core::PatternBuilder();
    auto ret = std::shared_ptr<vaip_core::Pattern>();
#include "gt_1_3_pattern/main_block_patch/mask_slice.h.inc"
    auto pattern_ = ret;
    return Rule::create_rule(
        pattern_, [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
          GT_SUBPASS_LOG(1) << "processing gt mask slice " << mask_slice_cnt++;
          // 3:mask_QuantizeLinear_Output
          collect_subgraph_nodes(gt_inner_map_.at("gt_002_transformer"), binder,
                                 {3});
          return false; // false means not changing the graph
        });
  }

  std::unique_ptr<Rule> create_gt_concat_36_rule(IPass* self) {
    auto builder = vaip_core::PatternBuilder();
    auto ret = std::shared_ptr<vaip_core::Pattern>();
#include "gt_1_3_pattern/main_block_patch/concat_36.h.inc"
    auto pattern_ = ret;
    return Rule::create_rule(
        pattern_, [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
          GT_SUBPASS_LOG(1) << "processing gt concat " << concat_cnt++;
          collect_subgraph_nodes(gt_inner_map_.at("gt_002_transformer"), binder,
                                 {});
          return false; // false means not changing the graph
        });
  }
  std::unique_ptr<Rule> create_gt_concat_32_rule(IPass* self) {
    auto builder = vaip_core::PatternBuilder();
    auto ret = std::shared_ptr<vaip_core::Pattern>();
#include "gt_1_3_pattern/main_block_patch/concat_32.h.inc"
    auto pattern_ = ret;
    return Rule::create_rule(
        pattern_, [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
          GT_SUBPASS_LOG(1) << "processing gt concat " << concat_cnt++;
          collect_subgraph_nodes(gt_inner_map_.at("gt_002_transformer"), binder,
                                 {});
          return false; // false means not changing the graph
        });
  }
  CONCAT_WILDCARD_RULE(40)
  CONCAT_WILDCARD_RULE(39)
  CONCAT_WILDCARD_RULE(38)
  CONCAT_WILDCARD_RULE(37)
  CONCAT_WILDCARD_RULE(36)
  CONCAT_WILDCARD_RULE(35)
  CONCAT_WILDCARD_RULE(34)
  CONCAT_WILDCARD_RULE(33)
  CONCAT_WILDCARD_RULE(32)
  CONCAT_WILDCARD_RULE(31)
  CONCAT_WILDCARD_RULE(30)
  CONCAT_WILDCARD_RULE(29)
  CONCAT_WILDCARD_RULE(28)

  std::unique_ptr<Rule> create_gt_oup_lid_rule(IPass* self) {
    auto builder = vaip_core::PatternBuilder();
    auto ret = std::shared_ptr<vaip_core::Pattern>();
#include "gt_1_3_pattern/main_block_patch/oup_lid.h.inc"
    auto pattern_ = ret;
    return Rule::create_rule(
        pattern_, [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
          GT_SUBPASS_LOG(1) << "processing gt oup_lid";
          collect_subgraph_nodes(gt_inner_map_.at("gt_002_transformer"), binder,
                                 {});
          std::string add_name =
              VAIP_ORT_API(node_get_name)(*binder[Add_0->get_id()].node);
          auto iter = std::find(transformer_final_add_nodes_.begin(),
                                transformer_final_add_nodes_.end(), add_name);
          oup_lid_idx_ =
              std::distance(transformer_final_add_nodes_.begin(), iter) +
              1;        // 1-indexed
          return false; // false means not changing the graph
        });
  }
  std::map<std::string, std::vector<std::string>> get_partitioned_nodes() {
    std::map<std::string, std::vector<std::string>> res;
    for (auto iter : gt_inner_map_) {
      res[iter.first] =
          std::vector<std::string>(iter.second.begin(), iter.second.end());
    }
    return res;
  }
  IPass& self_;
  MetaDefProto& initializer_map_;
  // pattern is matched in reversed dfs order,
  // so the matched tf block is naturally from 0 to 35
  size_t transformer_cnt = 0;
  size_t gather_cnt = 0;
  size_t mask_slice_cnt = 0;
  size_t concat_cnt = 0;
  int oup_lid_idx_ = -1;
  std::vector<std::string> transformer_final_add_nodes_;
  std::map<std::string, std::set<std::string>> gt_inner_map_ = {
      {"gt_000_front_conv", {}},
      {"gt_001_front_mm", {}},
      {"gt_002_transformer", {}},
      {"gt_003_tail", {}},
      {"gt_004_cache_frame_slice", {}}};
};

} // namespace vaip_vaiml_subgraph_processor

#undef GT_SUBPASS_LOG
#undef CONCAT_WILDCARD_RULE
