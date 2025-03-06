/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

#include "onnxruntime_api.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <codecvt>
#include <deque>
#include <filesystem>
#include <fstream>
#include <glog/logging.h>
#include <iostream>
#include <istream>
#include <memory>
#include <nlohmann/json.hpp>
#include <sstream>

#include "../common/bf16_utils.h"
#include "custom_op_ht_1_2.hpp"
// #include "constant_fold_result.h"
// #include "gen_gt_wts.h"
#include "../common/timer.h"
#include "../common/utils.h"
#include <vaip/util.hpp>
#include <vaip/vaip.hpp>

namespace vaip_vaiml_custom_op {

SUBGRAPH_ID
MyCustomOpHT1_2::IdentifySubgraph(
    const std::shared_ptr<MetaDefProto>& meta_def) {
  VAIML_DEBUG_PRINT("IdentifySubgraph sg_name=", sg_name_,
                    " meta_def size=", meta_def->nodes().size());
  SUBGRAPH_ID sg = SUBGRAPH_ID::UNKNOWN;
  // works for both EPContext graph and normal onnx graph
  if (sg_name_ == "ht_000_main") {
    sg = SUBGRAPH_ID::HT_LN_SG_LSTM;
  } else if (sg_name_ == "ht_001_slice") {
    sg = SUBGRAPH_ID::HT_SLICE;
  } else if (sg_name_ == "ht_002_concat") {
    sg = SUBGRAPH_ID::HT_CONCAT;
  } else {
    throw std::runtime_error("Cannot identify subgraph ID for " + sg_name_);
  }
  return sg;
}

inline std::string
get_xclbin_fullpath(std::shared_ptr<const PassContext> context,
                    const std::string& xclbin) {
  VAIML_DEBUG_PRINT("    xclbin: ", xclbin);
  // Check if xclbin is cached
  std::string xclbin_config_path =
      context->xclbin_path_to_cache_files(std::filesystem::path(xclbin))
          .string();
  VAIML_DEBUG_PRINT("    xclbin_config_path: ", xclbin_config_path);
  if (!xclbin_config_path.empty()) {
    if (fs::exists(xclbin_config_path) &&
        (xclbin_config_path.find(".xclbin") != std::string::npos)) {
      return (xclbin_config_path);
    }
  }

  return xclbin;
}

MyCustomOpHT1_2::MyCustomOpHT1_2(std::shared_ptr<const PassContext> context,
                                 const std::shared_ptr<MetaDefProto>& meta_def,
                                 onnxruntime::Model* model)
    : CustomOpImp(context, meta_def, model) {
  auto& session_option = context->get_config_proto().provider_options();
  // FIXME: remove polymorphism when preemption is by default
  if (session_option.at("enable_preemption") == "1") {
    runner_ = std::make_unique<vaiml_elf_runner::hw_elf_runner>();
  } else {
    runner_ = std::make_unique<hw_runner>();
  }
  model_version_ = session_option.at("model_name");
  sg_name_ = meta_def->vaiml_param().vaiml_model_path();
  subgraph_id_ = IdentifySubgraph(meta_def);

  auto initializer_map_c8 =
      context->read_file_c8("ht_init_map.proto.bin").value();
  MetaDefProto global_initializer_map;
  global_initializer_map.ParseFromString(
      std::string(initializer_map_c8.begin(), initializer_map_c8.end()));
  initializer_map_ = std::unordered_map<std::string, std::string>(
      global_initializer_map.generic_param().begin(),
      global_initializer_map.generic_param().end());

  TIMER(CONSTRUCTOR_TOP, sg_name_ + " MyCustomOpHT1_2 constructor total ")
  VAIML_DEBUG_PRINT("MyCustomOpHT1_2::MyCustomOpHT1_2 ", sg_name_,
                    " id: ", subgraph_id_, " model_version=", model_version_);
  size_t batchsize = 0;
  for (auto& vaiml_shapes : meta_def->vaiml_param().output_shapes()) {
    ort_output_shapes_.emplace_back(vaiml_shapes.shapes().begin(),
                                    vaiml_shapes.shapes().end());
    auto& a_shape = ort_output_shapes_.back();
    // handle GT_FRONT specially. Only set batchsize for
    // out/Add_output_0_QuantizeLinear_Output (1x25x512)
    if (subgraph_id_ == SUBGRAPH_ID::GT_FRONT) {
      if (a_shape[1] == 25 && a_shape[2] == 512) {
        batchsize = a_shape[0];
      }
    } else {
      batchsize = a_shape[0];
    }

    std::string str = "==>";
    for (auto item : a_shape) {
      str += " ";
      str += std::to_string(item);
    }
    VAIML_DEBUG_PRINT2("    output shape dump: ", str);
  }

  bool gt_mode_ = model_version_ == "GT_v1.2";
  datatype_to_size[ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT] = sizeof(float);
  datatype_to_size[ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8] = sizeof(uint8_t);
  datatype_to_size[ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8] = sizeof(int8_t);
  datatype_to_size[ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16] = sizeof(uint16_t);
  datatype_to_size[ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16] = sizeof(int16_t);
  datatype_to_size[ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32] = sizeof(int32_t);
  datatype_to_size[ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64] = sizeof(int64_t);
  datatype_to_size[ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE] = sizeof(double);
  datatype_to_size[ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32] = sizeof(uint32_t);
  datatype_to_size[ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64] = sizeof(uint64_t);

  datatype_to_string[ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT] = "float";
  datatype_to_string[ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8] = "uint8_t";
  datatype_to_string[ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8] = "int8_t";
  datatype_to_string[ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16] = "uint16_t";
  datatype_to_string[ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16] = "int16_t";
  datatype_to_string[ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32] = "int32_t";
  datatype_to_string[ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64] = "int64_t";
  datatype_to_string[ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE] = "double";
  datatype_to_string[ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32] = "uint32_t";
  datatype_to_string[ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64] = "uint64_t";

  // retrieve wts file from the cache directory to populate wts_
  auto wts_file_opt = context->read_file_c8(constants_file_name_);
  std::vector<char>& wts_file = wts_file_opt.value();

  auto const_info = meta_def->vaiml_param().const_data_info();
  wts_buffers_.resize(const_info.size());

  VAIML_DEBUG_PRINT("const_info.size(): ", const_info.size());
  int wts_vec_idx = 0;
  // Following constants are preformatted for HT model, so they need to be
  // excluded
  std::vector<std::string> ht_weight_names_preformat = {
      Alias("lstm320_h_wts"),  Alias("lstm320_x_wts"),  Alias("lstm320_bias"),
      Alias("lstm1024_h_wts"), Alias("lstm1024_x_wts"), Alias("lstm1024_bias")};

  for (auto it = const_info.begin(); it != const_info.end(); ++it) {
    flexmlrt::client::ErtIoTypeNew wts_vec;
    auto const_name = it->first;

    // if (std::find(ht_weight_names_preformat.begin(),
    // ht_weight_names_preformat.end(), const_name) !=
    // ht_weight_names_preformat.end()) {
    //   VAIML_DEBUG_PRINT("Skip preformatted weight: ", const_name);
    //   continue;
    // }

    VAIML_DEBUG_PRINT2("Constant idx: ", wts_vec_idx, " name: ", const_name);
    auto const_data_info = const_info.at(const_name);

    wts_vec.name = const_name;
    wts_vec.idx = wts_vec_idx;
    wts_vec.type = data_type_to_string(const_data_info.type());
    wts_vec.size = const_data_info.size();

    if (wts_vec.size == 0) {
      // Preformateed weights. Skip
      VAIML_DEBUG_PRINT("    Skip preformatted weight: ", const_name);
      continue;
    }

    std::string wt_shape_str;
    wts_vec.shape.push_back(batchsize);
    wt_shape_str += std::to_string(wts_vec.shape[wts_vec.shape.size() - 1]);
    for (auto dim : const_data_info.shape()) {
      wts_vec.shape.push_back(dim);
      wt_shape_str += " ";
      wt_shape_str += std::to_string(wts_vec.shape[wts_vec.shape.size() - 1]);
    }

    // Seek to the specified position
    size_t rd_offset = const_data_info.offset();
    if (rd_offset >= wts_file.size()) {
      throw std::runtime_error(
          "Failed to seek to the specified position in the wts file.");
    }
    // Resize the buffer to the specified size
    wts_buffers_[wts_vec_idx].resize(wts_vec.size);
    if (rd_offset + wts_vec.size > wts_file.size()) {
      throw std::runtime_error("Failed to read the specified amount of data.");
    }
    std::memcpy(wts_buffers_[wts_vec_idx].data(), wts_file.data() + rd_offset,
                wts_vec.size);
    wts_vec.data = wts_buffers_[wts_vec_idx].data();
    wts_[wts_vec.name] = wts_vec;
    wts_vec_idx++;

    VAIML_DEBUG_PRINT2("    weight name: ", wts_vec.name);
    VAIML_DEBUG_PRINT2("    weight id: ", wts_vec.idx);
    VAIML_DEBUG_PRINT2("    weight size: ", wts_vec.size);
    VAIML_DEBUG_PRINT2("    weight type: ", wts_vec.type);
    VAIML_DEBUG_PRINT2("    wt_shape_str: ", wt_shape_str);
    VAIML_DEBUG_PRINT2("    weight offset: ", rd_offset);
  }

  VAIML_DEBUG_PRINT("Total constants loaded: ", wts_vec_idx);
  // FIXME: remove load txn/ctrl bin when preemption is by default
  std::string xclbinFileName;
  std::string txnBinFile_front;
  std::string txnBinFile;
  std::vector<std::string> txnbins;
  std::string ctrlPktBinFile;
  std::vector<std::string> ctrlPktbins;
  std::vector<XRTRunOffset> xrt_offset;
  std::vector<std::stringstream> v_elf_istream;
  xclbinFileName = get_xclbin_fullpath(context, session_option.at("xclbin"));
  // xclbinFileName = vaiml_model_path_ + '/' + "design.xclbin";

  VAIML_DEBUG_PRINT("xclbin path: ", xclbinFileName);

  size_t wts_size_front = 0;
  size_t ifm_size_front = 0;
  size_t ofm_size_front = 0;
  size_t tmp_size_front = 0;
  size_t wts_size = 0;
  size_t ifm_size = 0;
  size_t ofm_size = 0;
  size_t tmp_size = 0;
  std::vector<KERNEL_NM> kernel_indices;
  std::vector<BO_ORDER> bo_orders;
  XRTRunOffset off;
  // All configuration below are version specific
  if (model_version_ == "HT_v1.2") {
    wts_size = WTS_SIZE_HT;
    ifm_size = IFM_SIZE_HT + 128; // additional 128 bytes for MM and Add RTP,
                                  // placed after LSTM-1024 h0/c0
    ofm_size = OFM_SIZE_HT;
    tmp_size = TMP_SIZE_HT;
    txnBinFile = getHt_binaries_ml_txn(model_version_);
    v_elf_istream.emplace_back(getElf_ht(model_version_));
    runner_->set_bo_order_vec({
        BO_ORDER::ODR_HT,
    });
    txnbins.push_back(txnBinFile);
    xrt_offset.push_back(off); // dummy offset
    kernel_indices.push_back(KERNEL_NM::HT);
  }
  if (subgraph_id_ < GT_CPU_OR_CONSTANT) {
    auto read_xclbin = context->read_xclbin(xclbinFileName);
    auto xclbin = std::vector<char>(read_xclbin.value().begin(),
                                    read_xclbin.value().end());
    runner_->load_xclbin(xclbin);

    VAIML_DEBUG_PRINT("load xclbin done");
    runner_->load_txn_bin(txnbins);
    runner_->load_elf(v_elf_istream);

    VAIML_DEBUG_PRINT("load txn done");
    if (ctrlPktbins.size() != 0) {
      // not used by HT
      runner_->load_ctrl_pkt_bin(ctrlPktbins);
    }
    VAIML_DEBUG_PRINT("load ctrl pkt done");
    runner_->hw_runner_init(ifm_size, wts_size, ofm_size, tmp_size, gt_mode_,
                            xrt_offset, kernel_indices);
    runner_->get_bo_ptrs(ifm_ptr_, wts_ptr_, ofm_ptr_);
  }

  TIMER(CONSTRUCTOR_WEIGHTS_FROMAT, "    " + sg_name_ + " weight format total ")
  VAIML_DEBUG_PRINT("Begin wts format for ", model_version_);
  if (model_version_ == "GT_v1.2") {
  } else if (model_version_ == "HT_v1.2" && subgraph_id_ < GT_CPU_OR_CONSTANT) {
    TIMER(CONSTRUCTOR_InitHtWeight, "    " + sg_name_ + " InitHtWeight total ")
    VAIML_DEBUG_PRINT("Running InitHtWeight subgraph_id_: ", subgraph_id_);
    InitHtWeight(wts_, wts_ptr_, *context);
  }
  if (subgraph_id_ == SUBGRAPH_ID::HT_SLICE) {
    scales_["Slice_13_output_0_s"] =
        *((float*)(wts_.at(Alias("Slice_13_output_0_s")).data));
    scales_["Slice_27_output_0_s"] =
        *((float*)(wts_.at(Alias("Slice_27_output_0_s")).data));
    scales_["Slice_12_output_0_s"] =
        *((float*)(wts_.at(Alias("Slice_12_output_0_s")).data));
    scales_["Slice_26_output_0_s"] =
        *((float*)(wts_.at(Alias("Slice_26_output_0_s")).data));

    zero_points_["Slice_13_output_0_zp"] =
        *((int8_t*)(wts_.at(Alias("Slice_13_output_0_zp")).data));
    zero_points_["Slice_27_output_0_zp"] =
        *((int8_t*)(wts_.at(Alias("Slice_27_output_0_zp")).data));
    zero_points_["Slice_12_output_0_zp"] =
        *((int8_t*)(wts_.at(Alias("Slice_12_output_0_zp")).data));
    zero_points_["Slice_26_output_0_zp"] =
        *((int8_t*)(wts_.at(Alias("Slice_26_output_0_zp")).data));
  }
  if (subgraph_id_ == SUBGRAPH_ID::HT_CONCAT) {
    scales_["lstm320_output_1_s"] =
        *((float*)(wts_.at(Alias("lstm320_output_1_s")).data));
    scales_["lstm1024_output_1_s"] =
        *((float*)(wts_.at(Alias("lstm1024_output_1_s")).data));
    scales_["lstm320_output_2_s"] =
        *((float*)(wts_.at(Alias("lstm320_output_2_s")).data));
    scales_["lstm1024_output_2_s"] =
        *((float*)(wts_.at(Alias("lstm1024_output_2_s")).data));

    zero_points_["lstm320_output_1_zp"] =
        *((int8_t*)(wts_.at(Alias("lstm320_output_1_zp")).data));
    zero_points_["lstm1024_output_1_zp"] =
        *((int8_t*)(wts_.at(Alias("lstm1024_output_1_zp")).data));
    zero_points_["lstm320_output_2_zp"] =
        *((int8_t*)(wts_.at(Alias("lstm320_output_2_zp")).data));
    zero_points_["lstm1024_output_2_zp"] =
        *((int8_t*)(wts_.at(Alias("lstm1024_output_2_zp")).data));
  }
  VAIML_DEBUG_PRINT("Finish wts formated");
  // read_file_c8(wts_ptr_, vaiml_model_path_ + '/' + "wts32.txt", wts_size);

  if (subgraph_id_ < GT_CPU_OR_CONSTANT) {
    runner_->pre_run_bo_sync();
  }
  VAIML_DEBUG_PRINT("DEBUG: MyCustomOpHT1_2 created for ", sg_name_);
  { wts_buffers_.clear(); }
}

MyCustomOpHT1_2::~MyCustomOpHT1_2() {}

bool MyCustomOpHT1_2::InitHtWeight(
    std::unordered_map<std::string, flexmlrt::client::ErtIoTypeNew>& wts_,
    int8_t* wts_ptr, const vaip_core::PassContext& context) {
#define GET_S_ZP(op, name_s, name_zp)                                          \
  float op##_scale = *((float*)(wts_[name_s].data));                           \
  int8_t op##_zp = *((int8_t*)(wts_[name_zp].data));
#define CONCATENATE_DETAIL(x, y) x##y
#define CONCATENATE(x, y) CONCATENATE_DETAIL(x, y)
  TIMER(InitHtWeight, "        InitHtWeight total ")

  int32_t wts_ptr_offset = 0;
  {
    auto start = std::chrono::steady_clock::now();
    // Layernorm
    double scale_bias = *((float*)(wts_.at(Alias("lnorm_0_bias_s")).data));
    int64_t zp_bias = *((int32_t*)(wts_.at(Alias("lnorm_0_bias_zp")).data));
    double scale_weight = *((float*)(wts_.at(Alias("lnorm_0_wts_s")).data));
    int64_t zp_weight = *((int8_t*)(wts_.at(Alias("lnorm_0_wts_zp")).data));
    double scale_ifm = *((float*)(wts_.at(Alias("lnorm_0_x_s")).data));
    int64_t zp_ifm = *((int8_t*)(wts_.at(Alias("lnorm_0_x_zp")).data));
    double scale_ofm = *((float*)(wts_.at(Alias("lnorm_0_y_s")).data));
    int64_t zp_ofm = *((int8_t*)(wts_.at(Alias("lnorm_0_y_zp")).data));
    int8_t* ln_w_ptr = (int8_t*)(wts_.at(Alias("lnorm_0_wts")).data);
    int32_t* ln_b_ptr = (int32_t*)(wts_.at(Alias("lnorm_0_bias")).data);
    std::vector<uint16_t> ln_part1_weight_vec = wts_gen_layernorm(
        ln_b_ptr, ln_w_ptr, 320, scale_bias, zp_bias, scale_weight, zp_weight,
        scale_ifm, zp_ifm, scale_ofm, zp_ofm);
    memcpy(wts_ptr + wts_ptr_offset, (int8_t*)(ln_part1_weight_vec.data()),
           ln_part1_weight_vec.size() * sizeof(uint16_t));
    wts_ptr_offset += ln_part1_weight_vec.size() * sizeof(uint16_t);

#ifdef VAIP_CUSTOM_OP_VAIML_PROFILING
    auto end = std::chrono::steady_clock::now();
    double time_sec = std::chrono::duration<double>(end - start).count();
    VAIML_DEBUG_PRINT("    Layernorm time (sec): ", time_sec);
#endif
  }

  {
#ifdef VAIP_CUSTOM_OP_VAIML_PROFILING
    auto start = std::chrono::steady_clock::now();
#endif
    // Sigmoid
    double scale_ifm = *((float*)(wts_.at(Alias("Sigmoid_input_0_s")).data));
    int64_t zp_ifm = *((int8_t*)(wts_.at(Alias("Sigmoid_input_0_zp")).data));
    double scale_ofm = *((float*)(wts_.at(Alias("Sigmoid_output_0_s")).data));
    int64_t zp_ofm = *((int8_t*)(wts_.at(Alias("Sigmoid_output_0_zp")).data));
    std::vector<uint16_t> sigmoid_part1_weight_vec =
        wts_gen_sigmoid(scale_ifm, zp_ifm, scale_ofm, zp_ofm);
    memcpy(wts_ptr + wts_ptr_offset, (int8_t*)(sigmoid_part1_weight_vec.data()),
           sigmoid_part1_weight_vec.size() * sizeof(uint16_t));
    wts_ptr_offset += sigmoid_part1_weight_vec.size() * sizeof(uint16_t);
#ifdef VAIP_CUSTOM_OP_VAIML_PROFILING
    auto end = std::chrono::steady_clock::now();
    double time_sec = std::chrono::duration<double>(end - start).count();
    VAIML_DEBUG_PRINT("    sigmoid time (sec): ", time_sec);
#endif
  }
  {
    lstm_init_wts lstm_in{};
#ifdef VAIP_CUSTOM_OP_VAIML_PROFILING
    auto start1 = std::chrono::steady_clock::now();
#endif

    // ht_wts_gen_lstm_b2b loads preforamted weights from cache_dir_
    std::vector<uint8_t> wts = ht_wts_gen_lstm_b2b(lstm_in, context);
    memcpy(lstm_320_rtp, lstm_in.lstm_320_rtp, 64);
    memcpy(lstm_1024_rtp, lstm_in.lstm_1024_rtp, 64);

#ifdef VAIP_CUSTOM_OP_VAIML_PROFILING
    auto end1 = std::chrono::steady_clock::now();
    double time_sec1 = std::chrono::duration<double>(end1 - start1).count();
    VAIML_DEBUG_PRINT("    ht_wts_gen_lstm_b2b time (sec): ", time_sec1);
#endif
    memcpy(wts_ptr + wts_ptr_offset, (int8_t*)(wts.data()),
           wts.size() * sizeof(int8_t));
    wts_ptr_offset += wts.size() * sizeof(uint8_t);
  }

  // layernorm
  {
#ifdef VAIP_CUSTOM_OP_VAIML_PROFILING
    auto start = std::chrono::steady_clock::now();
#endif
    double scale_bias = *((float*)(wts_.at(Alias("lnorm_1_bias_s")).data));
    int64_t zp_bias = *((int32_t*)(wts_.at(Alias("lnorm_1_bias_zp")).data));
    double scale_weight = *((float*)(wts_.at(Alias("lnorm_1_wts_s")).data));
    int64_t zp_weight = *((int8_t*)(wts_.at(Alias("lnorm_1_wts_zp")).data));
    double scale_ifm = *((float*)(wts_.at(Alias("lnorm_1_x_s")).data));
    int64_t zp_ifm = *((int8_t*)(wts_.at(Alias("lnorm_1_x_zp")).data));
    double scale_ofm = *((float*)(wts_.at(Alias("lnorm_1_y_s")).data));
    int64_t zp_ofm = *((int8_t*)(wts_.at(Alias("lnorm_1_y_zp")).data));
    int8_t* ln_w_ptr = (int8_t*)(wts_.at(Alias("lnorm_1_wts")).data);
    int32_t* ln_b_ptr = (int32_t*)(wts_.at(Alias("lnorm_1_bias")).data);
    std::vector<uint16_t> ln_part3_weight_vec = wts_gen_layernorm(
        ln_b_ptr, ln_w_ptr, 1024, scale_bias, zp_bias, scale_weight, zp_weight,
        scale_ifm, zp_ifm, scale_ofm, zp_ofm, 1, 1.0 / 64);
    memcpy(wts_ptr + wts_ptr_offset, (int8_t*)(ln_part3_weight_vec.data()),
           ln_part3_weight_vec.size() * sizeof(uint16_t));
    wts_ptr_offset += ln_part3_weight_vec.size() * sizeof(uint16_t);

#ifdef VAIP_CUSTOM_OP_VAIML_PROFILING
    auto end = std::chrono::steady_clock::now();
    double time_sec = std::chrono::duration<double>(end - start).count();
    VAIML_DEBUG_PRINT("    Layernorm 1 time (sec): ", time_sec);
#endif
  }
  {
    // matmul
#ifdef VAIP_CUSTOM_OP_VAIML_PROFILING
    auto start = std::chrono::steady_clock::now();
#endif
    float scale_matA =
        *((float*)(wts_.at(Alias("lin_dec_fc_matmul_in_1_s")).data));
    int64_t zp_matA =
        *((int8_t*)(wts_.at(Alias("lin_dec_fc_matmul_in_1_zp")).data));
    float scale_matB =
        *((float*)(wts_.at(Alias("lin_dec_fc_matmul_in_2_s")).data));
    int64_t zp_matB =
        *((int8_t*)(wts_.at(Alias("lin_dec_fc_matmul_in_2_zp")).data));
    float scale_matC =
        *((float*)(wts_.at(Alias("lin_dec_fc_matmul_out_s")).data));
    int64_t zp_matC =
        *((int8_t*)(wts_.at(Alias("lin_dec_fc_matmul_out_zp")).data));

    uint32_t M = 1;
    uint32_t K = 1024;
    uint32_t N = 512;
    uint32_t sv_M = 1;
    uint32_t sv_K = 128;
    uint32_t sv_N = 64;

    int8_t* B_ptr = (int8_t*)(wts_.at(Alias("lin_dec_fc_matmul_in_2")).data);

    std::vector<uint8_t> mm_weight_vec = wts_gen_matmul(
        B_ptr, M, K, N, sv_M, sv_K, sv_N, scale_matA, zp_matA, scale_matB,
        zp_matB, scale_matC, zp_matC, 1.0 / 64, 1.0 / 64, mm_add_rtp);
    memcpy(wts_ptr + wts_ptr_offset, (int8_t*)(mm_weight_vec.data()),
           mm_weight_vec.size() * sizeof(uint8_t));
    wts_ptr_offset += mm_weight_vec.size() * sizeof(uint8_t);

#ifdef VAIP_CUSTOM_OP_VAIML_PROFILING
    auto end = std::chrono::steady_clock::now();
    double time_sec = std::chrono::duration<double>(end - start).count();
    VAIML_DEBUG_PRINT("    matmul time (sec): ", time_sec);
#endif
  }
  {
    // add
#ifdef VAIP_CUSTOM_OP_VAIML_PROFILING
    auto start = std::chrono::steady_clock::now();
#endif
    int8_t* B_ptr =
        (int8_t*)(wts_.at(Alias("joint_network_lin_dec_fc_bias")).data);

    for (int i = 0; i < 512; ++i) {
      ((uint16_t*)(wts_ptr + wts_ptr_offset))[i] =
          uint16_t(((int8_t*)B_ptr)[i] + 128);
    }
    wts_ptr_offset += 512 * sizeof(uint16_t);
    float scale_matA = *((float*)(wts_.at(Alias("lin_dec_fc_add_in_s")).data));
    int64_t zp_matA = *((int8_t*)(wts_.at(Alias("lin_dec_fc_add_in_zp")).data));
    float scale_matB =
        *((float*)(wts_.at(Alias("joint_network_lin_dec_fc_bias_s")).data));
    int64_t zp_matB =
        *((int8_t*)(wts_.at(Alias("joint_network_lin_dec_fc_bias_zp")).data));
    float scale_matC = *((float*)(wts_.at(Alias("lin_dec_fc_add_out_s")).data));
    int64_t zp_matC =
        *((int8_t*)(wts_.at(Alias("lin_dec_fc_add_out_zp")).data));
    rtp_gen_add(scale_matA, zp_matA, scale_matB, zp_matB, scale_matC, zp_matC,
                1.0 / 64, 1.0 / 64, mm_add_rtp + 16);
#ifdef VAIP_CUSTOM_OP_VAIML_PROFILING
    auto end = std::chrono::steady_clock::now();
    double time_sec = std::chrono::duration<double>(end - start).count();
    VAIML_DEBUG_PRINT("        add time (sec): ", time_sec);
#endif
  }
  {
#ifdef VAIP_CUSTOM_OP_VAIML_PROFILING
    auto start = std::chrono::steady_clock::now();
#endif
    double scale_bias = *((float*)(wts_.at(Alias("lnorm_2_bias_s")).data));
    int64_t zp_bias = *((int32_t*)(wts_.at(Alias("lnorm_2_bias_zp")).data));
    double scale_weight = *((float*)(wts_.at(Alias("lnorm_2_wts_s")).data));
    int64_t zp_weight = *((int8_t*)(wts_.at(Alias("lnorm_2_wts_zp")).data));
    double scale_ifm = *((float*)(wts_.at(Alias("lnorm_2_x_s")).data));
    int64_t zp_ifm = *((int8_t*)(wts_.at(Alias("lnorm_2_x_zp")).data));
    double scale_ofm = *((float*)(wts_.at(Alias("lnorm_2_y_s")).data));
    int64_t zp_ofm = *((int8_t*)(wts_.at(Alias("lnorm_2_y_zp")).data));
    int8_t* ln_w_ptr = (int8_t*)(wts_.at(Alias("lnorm_2_wts")).data);
    int32_t* ln_b_ptr = (int32_t*)(wts_.at(Alias("lnorm_2_bias")).data);

    std::vector<uint16_t> ln_part3_weight_vec = wts_gen_layernorm(
        ln_b_ptr, ln_w_ptr, 512, scale_bias, zp_bias, scale_weight, zp_weight,
        scale_ifm, zp_ifm, scale_ofm, zp_ofm, 1.0 / 64, 1);
    memcpy(wts_ptr + wts_ptr_offset, (int8_t*)(ln_part3_weight_vec.data()),
           ln_part3_weight_vec.size() * sizeof(uint16_t));
    wts_ptr_offset += ln_part3_weight_vec.size() * sizeof(uint16_t);
#ifdef VAIP_CUSTOM_OP_VAIML_PROFILING
    auto end = std::chrono::steady_clock::now();
    double time_sec = std::chrono::duration<double>(end - start).count();
    VAIML_DEBUG_PRINT("        joint_network time (sec): ", time_sec);
#endif
  }
  return true;
}

int32_t MyCustomOpHT1_2::GetInputDataAndSet(Ort::KernelContext& ctx, int index,
                                            int8_t* ifm_ptr) const {
  std::vector<int64_t> inputs_dim;
  auto inputvalue = ctx.GetInput(index);
  const void* input = inputvalue.GetTensorRawData();
  auto type_and_shape = inputvalue.GetTensorTypeAndShapeInfo();

  auto input_type = type_and_shape.GetElementType();
  CHECK(datatype_to_size.count(input_type))
      << "unsupported data type " << input_type;

  auto shape_dim = type_and_shape.GetShape();
  inputs_dim = std::move(shape_dim);
  int inputsize = 1;
  int batchsize = inputs_dim[0];
  for (size_t j = 0; j < inputs_dim.size(); j++) {
    inputsize = inputs_dim[j] * inputsize;
  }
  for (int i = 0; i < inputsize; ++i) {
    ((uint16_t*)ifm_ptr)[i] = uint16_t(((int8_t*)input)[i] + 128);
  }

  // VAIML_DEBUG_PRINT("    input ", index, " dim size: ", inputsize);
  // flexmlrt::client::ErtIoType ifm_vec;
  // ifm_vec.data = std::move(const_cast<void*>(input));
  // ifm_vec.name =
  //     (index == 0) ? "ifm_ddr" : ("ifm_ddr_" + std::to_string(index));
  // ifm_vec.idx = (int)index;
  // ifm_vec.size = datatype_to_size.at(input_type) * inputsize;
  // VAIML_DEBUG_PRINT("    input ", index, " data buffer size: ",
  // ifm_vec.size); ifm_vec.type = datatype_to_string.at(input_type);
  // VAIML_DEBUG_PRINT("    input ", index, " type: ", ifm_vec.type);
  // for (int i = 0; i < 10; ++i)
  //   VAIML_DEBUG_PRINT("    input[", index, "]  ",
  //                     (int)(((const int8_t*)(input))[i]));

  return inputsize * sizeof(uint16_t); // ht convert all int8 to uint16
}

int32_t MyCustomOpHT1_2::GetOnputDataAndSet(Ort::KernelContext& ctx, int index,
                                            int8_t* ofm_ptr) const {
  auto output_shapes = ort_output_shapes_;
  auto& output_shape = output_shapes[index];
  auto ortvalue =
      ctx.GetOutput(index, output_shape.data(), output_shape.size());
  auto type_and_shape = ortvalue.GetTensorTypeAndShapeInfo();
  auto tensor_type = type_and_shape.GetElementType();
  auto num_elements = type_and_shape.GetElementCount();

  void* data = ortvalue.GetTensorMutableRawData();
  for (int i = 0; i < num_elements; ++i) {
    ((int8_t*)data)[i] = int8_t(((uint16_t*)ofm_ptr)[i] - 128);
  }

  return num_elements * sizeof(uint16_t);
}

int32_t MyCustomOpHT1_2::SliceCompute_HT(Ort::KernelContext& ctx) const {
  const float* c0 = ctx.GetInput(0).GetTensorData<float>();
  const float* h0 = ctx.GetInput(1).GetTensorData<float>();

  auto output_shapes = ort_output_shapes_;
  int8_t* slice_27 =
      ctx.GetOutput(0, output_shapes[0].data(), output_shapes[0].size())
          .GetTensorMutableData<int8_t>();
  int8_t* slice_26 =
      ctx.GetOutput(1, output_shapes[1].data(), output_shapes[1].size())
          .GetTensorMutableData<int8_t>();
  int8_t* slice_13 =
      ctx.GetOutput(2, output_shapes[2].data(), output_shapes[2].size())
          .GetTensorMutableData<int8_t>();
  int8_t* slice_12 =
      ctx.GetOutput(3, output_shapes[3].data(), output_shapes[3].size())
          .GetTensorMutableData<int8_t>();

  q_int8(c0, slice_13, scales_.at("Slice_13_output_0_s"),
         zero_points_.at("Slice_13_output_0_zp"), 1024);
  q_int8(c0 + 1024, slice_27, scales_.at("Slice_27_output_0_s"),
         zero_points_.at("Slice_27_output_0_zp"), 1024);

  q_int8(h0, slice_12, scales_.at("Slice_12_output_0_s"),
         zero_points_.at("Slice_12_output_0_zp"), 1024);
  q_int8(h0 + 1024, slice_26, scales_.at("Slice_26_output_0_s"),
         zero_points_.at("Slice_26_output_0_zp"), 1024);

  return 2 * 1024 * 2 * sizeof(int8_t);
}

int32_t MyCustomOpHT1_2::ConcatCompute_HT(Ort::KernelContext& ctx) const {
  const int8_t* lstm_1_1 = ctx.GetInput(0).GetTensorData<int8_t>();
  const int8_t* lstm_1_2 = ctx.GetInput(1).GetTensorData<int8_t>();
  const int8_t* lstm_0_1 = ctx.GetInput(2).GetTensorData<int8_t>();
  const int8_t* lstm_0_2 = ctx.GetInput(3).GetTensorData<int8_t>();

  auto output_shapes = ort_output_shapes_;

  float* h1 = ctx.GetOutput(0, output_shapes[0].data(), output_shapes[0].size())
                  .GetTensorMutableData<float>();
  float* c1 = ctx.GetOutput(1, output_shapes[1].data(), output_shapes[1].size())
                  .GetTensorMutableData<float>();
  // dq_int8(lstm_0_1, h1, 0.0073768566362559795, 6, 1024);
  // dq_int8(lstm_1_1, h1 + 1024, 0.007354839239269495, 6, 1024);
  //
  // dq_int8(lstm_0_2, c1, 0.02116578258574009, 18, 1024);
  // dq_int8(lstm_1_2, c1 + 1024, 0.02068982645869255, 36, 1024);

  dq_int8(lstm_0_1, h1, scales_.at("lstm320_output_1_s"),
          zero_points_.at("lstm320_output_1_zp"), 1024);
  dq_int8(lstm_1_1, h1 + 1024, scales_.at("lstm1024_output_1_s"),
          zero_points_.at("lstm1024_output_1_zp"), 1024);

  dq_int8(lstm_0_2, c1, scales_.at("lstm320_output_2_s"),
          zero_points_.at("lstm320_output_2_zp"), 1024);
  dq_int8(lstm_1_2, c1 + 1024, scales_.at("lstm1024_output_2_s"),
          zero_points_.at("lstm1024_output_2_zp"), 1024);
  return 2 * 1024 * 2 * sizeof(float);
}
void MyCustomOpHT1_2::Compute(const OrtApi* api,
                              OrtKernelContext* context) const {
  if (Ort::Global<void>::api_ == nullptr) {
    Ort::Global<void>::api_ = api;
  }
  Ort::KernelContext ctx(context);
  TIMER(Compute, sg_name_ + " ort compute total ")

  if (subgraph_id_ == SUBGRAPH_ID::HT_SLICE) {
    SliceCompute_HT(ctx);
    return;
  }
  if (subgraph_id_ == SUBGRAPH_ID::HT_CONCAT) {
    ConcatCompute_HT(ctx);
    return;
  }
  auto num_inputs = ctx.GetInputCount();
  VAIML_DEBUG_PRINT("    inputs number: ", num_inputs);
  int32_t ifm_offset = 0;
  if (model_version_ == "HT_v1.2") {
    ifm_offset += GetInputDataAndSet(ctx, 4, ifm_ptr_ + ifm_offset); // ifm
    memcpy(ifm_ptr_ + ifm_offset, (int8_t*)lstm_lut.data(),
           lstm_lut.size() * sizeof(int16_t));                       // lstm_lut
    ifm_offset += lstm_lut.size() * sizeof(int16_t);
    ifm_offset += GetInputDataAndSet(ctx, 0, ifm_ptr_ + ifm_offset); // h0
    ifm_offset += GetInputDataAndSet(ctx, 1, ifm_ptr_ + ifm_offset); // c0
    memcpy(ifm_ptr_ + ifm_offset, lstm_320_rtp, 64);
    ifm_offset += 64;
    memcpy(ifm_ptr_ + ifm_offset, (int8_t*)lstm_lut.data(),
           lstm_lut.size() * sizeof(int16_t));                       // lstm_lut
    ifm_offset += lstm_lut.size() * sizeof(int16_t);
    ifm_offset += GetInputDataAndSet(ctx, 2, ifm_ptr_ + ifm_offset); // h0-lstm1
    ifm_offset += GetInputDataAndSet(ctx, 3, ifm_ptr_ + ifm_offset); // c0-lstm1
    memcpy(ifm_ptr_ + ifm_offset, lstm_1024_rtp, 64);
    ifm_offset += 64;
    memcpy(ifm_ptr_ + ifm_offset, mm_add_rtp, 128);
  }

  // write_file((uint32_t*)ifm_ptr_, ifm_offset, vaiml_model_path_ +
  // '/'+sg_name_+"_ifm.txt");

  auto output_shapes = ort_output_shapes_;
  VAIML_DEBUG_PRINT("    outputs number: ", output_shapes.size());
  auto err_status =
      runner_->run((void*)ifm_ptr_, (void*)wts_ptr_, (void*)ofm_ptr_);

  int32_t ofm_offset = 0;
  if (model_version_ == "HT_v1.2") {
    ofm_offset += 128;
    ofm_offset += GetOnputDataAndSet(ctx, 2, ofm_ptr_ + ofm_offset); // h1
    ofm_offset += GetOnputDataAndSet(ctx, 4, ofm_ptr_ + ofm_offset); // h1-lstm1
    ofm_offset += GetOnputDataAndSet(ctx, 1, ofm_ptr_ + ofm_offset); // c1
    ofm_offset += GetOnputDataAndSet(ctx, 3, ofm_ptr_ + ofm_offset); // c1-lstm1
    ofm_offset += GetOnputDataAndSet(ctx, 0, ofm_ptr_ + ofm_offset); // ofm
  }
  if (err_status == -2) {
    printf("BOARD CRASHED\n");
  }

  // write_file((uint32_t*)ofm_ptr_, ofm_offset, vaiml_model_path_ +
  // '/'+sg_name_+"_ofm.txt");
  // write_file_binary((uint32_t*)ofm_ptr_, 12928000, vaiml_model_path_ +
  // '/'+sg_name_+"_ofm.txt");
}

} // namespace vaip_vaiml_custom_op
