/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */
#include "custom_op_gt_1_3.hpp"
#include "../../common/bf16_utils.h"
#include "../GT_1_2/gen_gt_wts.h"
#include "constants_gt_1_3.hpp"
#include "elf_pkg_gt_1_3.hpp"
#include "txn_pkg_gt_1_3.hpp"
// #define GT_FRONT_SZ 16480 + 552960
#define GT_FRONT_SZ 0
#include <cstdlib>
#include <sstream>
#include <vaip/util.hpp>
#include <vaip/vaip.hpp>
namespace vaip_vaiml_custom_op {
constexpr unsigned conv_ifm_size = 16480;
constexpr unsigned conv_ofm_size = 552960;
constexpr unsigned conv_wts_size = 2501056;
constexpr unsigned linear_out_tmp_size = 2129920;
constexpr unsigned linear_out_ofm_size = 25600;
constexpr unsigned linear_out_wts_size = 5418432;
constexpr unsigned transformer_io_size =
    4300800;  // unified size for 3 in 1 transformer block
constexpr unsigned transformer_wts_size =
    11608064; // this is the same as gt1.2, but there are holes internally
constexpr unsigned transformer_tmp_size =
    102400 + 576000 + 499200; // same as 1.2
constexpr unsigned rtp_skip_size = 1472;

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
std::map<std::string, std::vector<char>> MyCustomOpGT1_3::node_cache;

SUBGRAPH_ID
MyCustomOpGT1_3::IdentifySubgraphVector(
    const std::shared_ptr<MetaDefProto>& meta_def) {
  SUBGRAPH_ID sg = SUBGRAPH_ID::UNKNOWN;
  // works for both EPContext graph and normal onnx graph
  if (sg_name_ == "gt_000_front_conv") {
    sg = SUBGRAPH_ID::GT_FRONT;
  } else if (sg_name_ == "gt_001_front_mm") {
    sg = SUBGRAPH_ID::GT_FRONT_MM;
  } else if (sg_name_ == "gt_002_transformer") {
    sg = SUBGRAPH_ID::GT_TRANSFORMER_BLOCK;
  } else if (sg_name_ == "gt_003_tail") {
    sg = SUBGRAPH_ID::GT_LN_MATMUL_ADD_LN;
  } else if (sg_name_ == "gt_004_cache_frame_slice") {
    sg = SUBGRAPH_ID::GT_CACHE_FRAMES_SLICE;
  } else {
    throw std::runtime_error("Cannot identify subgraph ID for " + sg_name_);
  }
  return sg;
}

std::vector<std::vector<uint8_t>> SplitTransformerHeadMMWts(
    const std::string& mm_wts_name,
    const std::unordered_map<std::string, flexmlrt::client::ErtIoTypeNew>&
        wts_) {
  // split 512x1536 to 3 512x512 (i.e.
  // 512x1536--reshape-->512x3x512--transpose(1, 0, 2)-->3x512x512)
  uint8_t* w_ptr_orig = (uint8_t*)(wts_.at(mm_wts_name).data);
  std::vector<std::vector<uint8_t>> res(3);
  for (int i = 0; i < 3; ++i) {
    res[i].resize(512 * 512);
    for (int j = 0; j < 512; ++j) {
      memcpy(res[i].data() + j * 512, w_ptr_orig + j * 1536 + i * 512, 512);
    }
  }
  return res;
}

MyCustomOpGT1_3::MyCustomOpGT1_3(std::shared_ptr<const PassContext> context,
                                 const std::shared_ptr<MetaDefProto>& meta_def,
                                 onnxruntime::Model* model)
    : CustomOpImp(context, meta_def, model),
      model_version_(
          context->get_config_proto().provider_options().at("model_name")) {
  if (model_version_ == "GTC_v1.0") {
    model_version_ = "GT_v1.3"; // to reuse txn bins and wts functions
  }

  auto& session_option = context->get_config_proto().provider_options();
  // FIXME: remove polymorphism when preemption is by default
  if (session_option.at("enable_preemption") == "1") {
    runner_ = std::make_unique<vaiml_elf_runner::hw_elf_runner>();
  } else {
    runner_ = std::make_unique<hw_runner>();
  }
  sg_name_ = meta_def->vaiml_param().vaiml_model_path();
  LoadConstantsToWts(context, meta_def);
  subgraph_id_ = IdentifySubgraphVector(meta_def);
  auto initializer_map_c8 =
      context->read_file_c8("gt_init_map.proto.bin").value();
  MetaDefProto global_initializer_map;
  global_initializer_map.ParseFromString(
      std::string(initializer_map_c8.begin(), initializer_map_c8.end()));
  initializer_map_ = std::unordered_map<std::string, std::string>(
      global_initializer_map.generic_param().begin(),
      global_initializer_map.generic_param().end());
  transformer_block_num_ =
      std::stoi(initializer_map_.at("__TRANSFORMER_NUM__"));
  oup_lid_idx_ = std::stoi(initializer_map_.at("__OUP_LID_IDX__"));
  // FIXME: remove load txn/ctrl bin when preemption is by default
  std::vector<std::string> v_txn_bins;
  std::vector<std::string> v_ctrl_pkt_bins;
  std::vector<XRTRunOffset> xrt_offset;
  std::vector<KERNEL_NM> v_kernel_indices;
  std::vector<BO_ORDER> v_bo_order;
  std::vector<std::stringstream> v_elf_istream;
  size_t wts_size, ifm_size, ofm_size, tmp_size;
  // setting output_shapes to be used by getting output from onnx
  for (auto& vaiml_shapes : meta_def->vaiml_param().output_shapes()) {
    ort_output_shapes_.emplace_back(vaiml_shapes.shapes().begin(),
                                    vaiml_shapes.shapes().end());
  }
  PrepareHwRunner(v_txn_bins, v_ctrl_pkt_bins, v_elf_istream, xrt_offset,
                  v_kernel_indices, v_bo_order, ifm_size, ofm_size, wts_size,
                  tmp_size);
  runner_->set_bo_order_vec(v_bo_order);

  if (subgraph_id_ < GT_CPU_OR_CONSTANT) {
    std::string xclbinFileName =
        get_xclbin_fullpath(context, session_option.at("xclbin"));
    auto read_xclbin = context->read_xclbin(xclbinFileName);
    auto xclbin = std::vector<char>(read_xclbin.value().begin(),
                                    read_xclbin.value().end());
    runner_->load_xclbin(xclbin);
    VAIML_DEBUG_PRINT("load xclbin done at ", xclbinFileName);
    runner_->load_txn_bin(v_txn_bins);
    VAIML_DEBUG_PRINT("load txn done");
    runner_->load_ctrl_pkt_bin(v_ctrl_pkt_bins);
    runner_->load_elf(v_elf_istream);
    VAIML_DEBUG_PRINT("load ctrl pkt done");
    runner_->hw_runner_init(ifm_size, wts_size, ofm_size, tmp_size,
                            true /*gt_mode*/, xrt_offset, v_kernel_indices);
    VAIML_DEBUG_PRINT("hw runner init done");
    runner_->get_bo_ptrs(ifm_ptr_, wts_ptr_, ofm_ptr_);
  }

  VAIML_DEBUG_PRINT("Begin wts format for ", model_version_);
  InitWeights();
}

void MyCustomOpGT1_3::PrepareHwRunner(
    std::vector<std::string>& v_txn_bins,
    std::vector<std::string>& v_ctrl_pkt_bins,
    std::vector<std::stringstream>& v_elf_istream,
    std::vector<XRTRunOffset>& xrt_offset,
    std::vector<KERNEL_NM>& v_kernel_indices, std::vector<BO_ORDER>& v_bo_order,
    size_t& ifm_size, size_t& ofm_size, size_t& wts_size, size_t& tmp_size) {
  if (subgraph_id_ == GT_FRONT) {
    ifm_size = 16480 /*conv input*/ + 552960 /*conv output*/;
    wts_size = 2501056 /*conv weights*/;
    ofm_size = 0;
    tmp_size = 2129920;
    /// xrt_offset.push_back({.ifm_offset = 0, .ofm_offset = 16480, .wts_offset
    /// = 0, .tmp_offset=0});
    xrt_offset.emplace_back(0, 0, 16480, 0);
    v_elf_istream.emplace_back(getElf_front_sub(model_version_));
    v_txn_bins.push_back(getBins_front_sub_ml_txn_1_3(model_version_));
    v_ctrl_pkt_bins.push_back(
        getBins_out_matmul_bias_ctrl_pkt_1_3(model_version_));
    v_kernel_indices.push_back(KERNEL_NM::GT_CONV);
    v_bo_order.push_back(BO_ORDER::ODR_GT_CONV);
  } else if (subgraph_id_ == GT_FRONT_MM) {
    ifm_size = 16480 /*conv input*/ + 552960 /*conv transpose reshape output*/ +
               25600 /*mm and add output*/;
    wts_size = 2501056 /*conv weights*/ + 5418432 /*mm and add weight */;
    ofm_size = 0;
    tmp_size = 2129920;
    xrt_offset.emplace_back(16480 /*input offset*/, 2501056 /*wts offset*/,
                            16480 + 552960 /*ofm offset*/, 0);
    v_elf_istream.emplace_back(getElf_out_matmul_bias(model_version_));
    v_txn_bins.push_back(getBins_out_matmul_bias_ml_txn_1_3(model_version_));
    v_ctrl_pkt_bins.push_back(
        getBins_out_matmul_bias_ctrl_pkt_1_3(model_version_));
    v_kernel_indices.push_back(KERNEL_NM::GT_MM);
    v_bo_order.push_back(BO_ORDER::ODR_GT_HEAD);
  } else if (subgraph_id_ == GT_LN_MATMUL_ADD_LN) {
    ifm_size = 25600 /*first ln input*/ + 25600 /*last ln output*/;
    wts_size = 590400;
    ofm_size = 0;
    tmp_size = 51200;
    xrt_offset.emplace_back(0 /*input offset*/, 0 /*wts offset*/,
                            25600 /*ofm offset*/, 0); // ofm offset should be 0?
    v_elf_istream.emplace_back(getElf_ln_matmul_bias_ln(model_version_));
    v_txn_bins.push_back(getBins_ln_matmul_bias_ln_ml_txn_1_3(model_version_));
    v_ctrl_pkt_bins.push_back(
        getBins_ln_matmul_bias_ln_ctrl_pkt_1_3(model_version_));
    v_kernel_indices.push_back(KERNEL_NM::GT_MM);
    v_bo_order.push_back(BO_ORDER::ODR_GT_TAIL);
  } else if (subgraph_id_ == GT_TRANSFORMER_BLOCK) {
    ifm_size = 4300800 * transformer_block_num_ + 25600;
    wts_size = 11608064 * transformer_block_num_;
    ofm_size = 0;
    tmp_size = 1536000;
    for (int i = 0; i < transformer_block_num_; i++) {
      size_t inout_base_ddr = i * 4300800 + GT_FRONT_SZ;
      size_t wts_base_ddr = i * 11608064;
      xrt_offset.emplace_back(inout_base_ddr /*input offset*/,
                              wts_base_ddr /*wts offset*/,
                              inout_base_ddr /*ofm offset*/, 0);
      v_txn_bins.push_back(
          getBins_transformer_layers_ml_txn_1_3(model_version_));
      v_ctrl_pkt_bins.push_back(
          getBins_transformer_layers_ctrl_pkt_1_3(model_version_));
      v_kernel_indices.push_back(KERNEL_NM::GT_MM);
      v_bo_order.push_back(BO_ORDER::ODR_GT_TRANSFORMER);
    }
    v_elf_istream.emplace_back(
        getElf_transformer_layers(model_version_)); // avoid creating same elf
  }
}

void MyCustomOpGT1_3::LoadConstantsToWts(
    std::shared_ptr<const PassContext> context,
    const std::shared_ptr<MetaDefProto>& meta_def) {
  std::vector<char> wts_file;
  auto wts_file_opt = context->read_file_c8("wts.bin");
  wts_file = wts_file_opt.value();
  auto const_info = meta_def->vaiml_param().const_data_info();
  wts_buffers_.resize(const_info.size());
  VAIML_DEBUG_PRINT("const_info.size(): ", const_info.size());
  int wts_vec_idx = 0;
  for (auto it = const_info.begin(); it != const_info.end(); ++it) {
    flexmlrt::client::ErtIoTypeNew wts_vec;
    auto const_name = it->first;
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

    for (auto dim : const_data_info.shape()) {
      wts_vec.shape.push_back(dim);
    }
    size_t rd_offset = const_data_info.offset();
    if (rd_offset >= wts_file.size()) {
      throw std::runtime_error(
          "Failed to seek to the specified position in the wts file.");
    }
    wts_buffers_[wts_vec_idx].resize(wts_vec.size);
    if (rd_offset + wts_vec.size > wts_file.size()) {
      throw std::runtime_error("Failed to read the specified amount of data.");
    }
    std::memcpy(wts_buffers_[wts_vec_idx].data(), wts_file.data() + rd_offset,
                wts_vec.size);
    wts_vec.data = wts_buffers_[wts_vec_idx].data();
    wts_[wts_vec.name] = wts_vec;
    wts_vec_idx++;
    VAIML_DEBUG_PRINT("const name loaded: ", wts_vec.name);
  }
  VAIML_DEBUG_PRINT("Total constants loaded: ", wts_vec_idx);
}

void to_file(std::string f_name, size_t size, int8_t* ptr) {
  std::ofstream fs(f_name, std::ios::out | std::ios::binary);
  fs.write(reinterpret_cast<char*>(ptr), size);
  fs.close();
}

void MyCustomOpGT1_3::InitWeights() {
  // init conv part
  std::string ifm_s_name, ifm_z_name, wts_s_name, wts_z_name, ofm_s_name,
      ofm_z_name, wts_w_name;
  size_t total_wts_bytes = 0;

  if (subgraph_id_ == SUBGRAPH_ID::GT_FRONT) {
    // SUB
    int8_t* wts_ptr_front = wts_ptr_;
    ifm_s_name = Alias("front_sub_in_s");  //"cache_frames_scale";
    ifm_z_name = Alias("front_sub_in_zp"); //"cache_frames_zero_point";
    wts_s_name =
        Alias("front_sub_wts_s");  //"encoder_embedding.global_mean_scale";
    wts_z_name =
        Alias("front_sub_wts_zp"); //"encoder_embedding.global_mean_zero_point";
    ofm_s_name =
        Alias("front_sub_out_s");  //"/encoder_embedding/Sub_output_0_scale";
    ofm_z_name = Alias(
        "front_sub_out_zp");    //"/encoder_embedding/Sub_output_0_zero_point";
    wts_w_name =
        Alias("front_sub_wts"); //"encoder_embedding.global_mean_quantized";
    // MUL
    std::string ifm_s_name1, ifm_z_name1, wts_s_name1, wts_z_name1, ofm_s_name1,
        ofm_z_name1, wts_w_name1;
    ifm_s_name1 =
        Alias("front_mul_in_s"); //"/encoder_embedding/Sub_output_0_scale";
    ifm_z_name1 = Alias(
        "front_mul_in_zp");      //"/encoder_embedding/Sub_output_0_zero_point";
    wts_s_name1 =
        Alias("front_mul_wts_s"); //"encoder_embedding.global_invstd_scale";
    wts_z_name1 = Alias(
        "front_mul_wts_zp"); //"encoder_embedding.global_invstd_zero_point";
    ofm_s_name1 =
        Alias("front_mul_out_s"); //"/encoder_embedding/Mul_output_0_scale";
    ofm_z_name1 = Alias(
        "front_mul_out_zp");    //"/encoder_embedding/Mul_output_0_zero_point";
    wts_w_name1 =
        Alias("front_mul_wts"); //"encoder_embedding.global_invstd_quantized";
    total_wts_bytes += GT_SUB_MUL_WTS_convert(
        wts_, wts_ptr_front, ifm_s_name, ifm_z_name, wts_s_name, wts_z_name,
        ofm_s_name, ofm_z_name, wts_w_name, ifm_s_name1, ifm_z_name1,
        wts_s_name1, wts_z_name1, ofm_s_name1, ofm_z_name1, wts_w_name1,
        model_version_);
    // CONV1
    std::string bias_s_name, bias_z_name, bias_w_name;
    ifm_s_name =
        Alias("front_conv1_in_s"); //"/encoder_embedding/Mul_output_0_scale";
    ifm_z_name = Alias(
        "front_conv1_in_zp"); //"/encoder_embedding/Mul_output_0_zero_point";
    wts_s_name =
        Alias("front_conv1_wts_s");  //"encoder.embed.conv.0.weight_scale";
    wts_z_name =
        Alias("front_conv1_wts_zp"); //"encoder.embed.conv.0.weight_zero_point";
    bias_s_name = Alias(
        "front_conv1_bias_s"); //"encoder.embed.conv.0.bias_quantized_scale";
    bias_z_name = Alias(
        "front_conv1_bias_zp"); //"encoder.embed.conv.0.bias_quantized_zero_point";
    ofm_s_name =
        Alias("front_conv1_out_s");  //"/conv/conv.1/Relu_output_0_scale";
    ofm_z_name =
        Alias("front_conv1_out_zp"); //"/conv/conv.1/Relu_output_0_zero_point";
    wts_w_name =
        Alias("front_conv1_wts");    //"encoder.embed.conv.0.weight_quantized";
    bias_w_name =
        Alias("front_conv1_bias");   //"encoder.embed.conv.0.bias_quantized";
    total_wts_bytes += GT_CONV_WTS_convert(
        wts_, wts_ptr_front, ifm_s_name, ifm_z_name, wts_s_name, wts_z_name,
        bias_s_name, bias_z_name, ofm_s_name, ofm_z_name, wts_w_name,
        bias_w_name, 512, 1, 3, 3, 8, const_cast<uint32_t*>(conv0_c0),
        const_cast<uint32_t*>(conv0_lp), model_version_);
    // CONV2
    ifm_s_name =
        Alias("front_conv2_in_s");   //"/conv/conv.1/Relu_output_0_scale";
    ifm_z_name =
        Alias("front_conv2_in_zp");  //"/conv/conv.1/Relu_output_0_zero_point";
    wts_s_name =
        Alias("front_conv2_wts_s");  //"encoder.embed.conv.2.weight_scale";
    wts_z_name =
        Alias("front_conv2_wts_zp"); //"encoder.embed.conv.2.weight_zero_point";
    bias_s_name = Alias(
        "front_conv2_bias_s"); //"encoder.embed.conv.2.bias_quantized_scale";
    bias_z_name = Alias(
        "front_conv2_bias_zp"); //"encoder.embed.conv.2.bias_quantized_zero_point";
    ofm_s_name =
        Alias("front_conv2_out_s");  //"/conv/conv.3/Relu_output_0_scale";
    ofm_z_name =
        Alias("front_conv2_out_zp"); //"/conv/conv.3/Relu_output_0_zero_point";
    wts_w_name =
        Alias("front_conv2_wts");    //"encoder.embed.conv.2.weight_quantized";
    bias_w_name =
        Alias("front_conv2_bias");   //"encoder.embed.conv.2.bias_quantized";
    total_wts_bytes += GT_CONV_WTS_convert(
        wts_, wts_ptr_front, ifm_s_name, ifm_z_name, wts_s_name, wts_z_name,
        bias_s_name, bias_z_name, ofm_s_name, ofm_z_name, wts_w_name,
        bias_w_name, 512, 512, 3, 3, 32, const_cast<uint32_t*>(conv1_c0),
        const_cast<uint32_t*>(conv1_lp), model_version_);
    VAIML_DEBUG_PRINT(
        "Finish to format WTS for front subgraph CONV, total bytes: ",
        total_wts_bytes);
  } else if (subgraph_id_ == SUBGRAPH_ID::GT_FRONT_MM) {
    std::string i_s_name = Alias("front_mmb_in_s");
    std::string i_z_name = Alias("front_mmb_in_zp");
    std::string w_s_name = Alias("front_mmb_wts_s");
    std::string w_z_name = Alias("front_mmb_wts_zp");
    std::string b_s_name = Alias("front_mmb_bias_s");
    std::string b_z_name = Alias("front_mmb_bias_zp");
    std::string o_s_name = Alias("front_mmb_out_s");
    std::string o_z_name = Alias("front_mmb_out_zp");
    std::string w_name = Alias("front_mmb_wts");
    std::string b_name = Alias("front_mmb_bias");
    int8_t* wts_ptr_linear_out =
        (int8_t*)(wts_ptr_ + 2501056 + 1472); // leave space for rtp
    int8_t* rtp_ptr =
        wts_ptr_ + 2501056 + gt_global_rtp_offset_.at(subgraph_id_);
    total_wts_bytes += GT_MMB_WTS_convert_ptr(
        wts_, wts_ptr_linear_out, rtp_ptr, i_s_name, i_z_name, w_s_name,
        w_z_name, b_s_name, b_z_name, o_s_name, o_z_name, w_name, b_name, 25,
        10240 /*gemm k, padded from 19*512=9728*/, 512, 25 /*bias broadcast*/,
        16, 256, 16 /*sv_N changed to 16*/, true /*performs padding*/,
        model_version_);
    VAIML_DEBUG_PRINT("Finish formatting wts for linear out, total bytes: ",
                      total_wts_bytes);
  } else if (subgraph_id_ == SUBGRAPH_ID::GT_LN_MATMUL_ADD_LN) {
    int8_t* wts_ptr = wts_ptr_ + 1472; // leave space for rtp
    int8_t* rtp_ptr = wts_ptr_ + gt_global_rtp_offset_.at(subgraph_id_);
    total_wts_bytes += GT_LN_WTS_convert(
        wts_, wts_ptr, Alias("lin_enc_ln_0_in_s"), Alias("lin_enc_ln_0_in_zp"),
        Alias("lin_enc_ln_0_wts_s"), Alias("lin_enc_ln_0_wts_zp"),
        Alias("lin_enc_ln_0_bias_s"), Alias("lin_enc_ln_0_bias_zp"),
        Alias("lin_enc_ln_0_out_s"), Alias("lin_enc_ln_0_out_zp"),
        Alias("lin_enc_ln_0_wts"), Alias("lin_enc_ln_0_bias"), model_version_);
    total_wts_bytes += GT_MMB_WTS_convert_ptr(
        wts_, wts_ptr, rtp_ptr, Alias("lin_enc_mmb_in_s"),
        Alias("lin_enc_mmb_in_zp"), Alias("lin_enc_mmb_wts_s"),
        Alias("lin_enc_mmb_wts_zp"), Alias("lin_enc_mmb_bias_s"),
        Alias("lin_enc_mmb_bias_zp"), Alias("lin_enc_mmb_out_s"),
        Alias("lin_enc_mmb_out_zp"), Alias("lin_enc_mmb_wts"),
        Alias("lin_enc_mmb_bias"), 25, 512, 512, 25 /*bias broadcast*/, 32, 64,
        16 /*sv_N changed to 16*/, false /*performs padding*/, model_version_);
    wts_ptr = wts_ptr_ + 588288;
    total_wts_bytes += GT_LN_WTS_convert(
        wts_, wts_ptr, Alias("lin_enc_ln_1_in_s"), Alias("lin_enc_ln_1_in_zp"),
        Alias("lin_enc_ln_1_wts_s"), Alias("lin_enc_ln_1_wts_zp"),
        Alias("lin_enc_ln_1_bias_s"), Alias("lin_enc_ln_1_bias_zp"),
        Alias("lin_enc_ln_1_out_s"), Alias("lin_enc_ln_1_out_zp"),
        Alias("lin_enc_ln_1_wts"), Alias("lin_enc_ln_1_bias"), model_version_);
    VAIML_DEBUG_PRINT("Finish formatting wts for tail, total bytes: ",
                      total_wts_bytes);
    // std::string out_filename("gt_tail_dump_wts32.txt");
    // to_file(out_filename, 590400, wts_ptr_ );
  } else if (subgraph_id_ == SUBGRAPH_ID::GT_TRANSFORMER_BLOCK) {
    // set wts from onnx model
    for (int i = 0; i < transformer_block_num_; i++) {
      size_t wts_base = 11608064 * i;
      subgraph_id_ = SUBGRAPH_ID::GT_QKV;
      InitTransformerBlockWeights(wts_, wts_ptr_ + wts_base, i);
      subgraph_id_ = SUBGRAPH_ID::GT_MATMUL_REDUCE;
      InitTransformerBlockWeights(wts_, wts_ptr_ + wts_base + 1783296, i);
      subgraph_id_ = SUBGRAPH_ID::GT_SM_LINEAR_OUT_FEED_FORWARD;
      InitTransformerBlockWeights(wts_, wts_ptr_ + wts_base + 1843712, i);
    }
    subgraph_id_ = SUBGRAPH_ID::GT_TRANSFORMER_BLOCK;

    // to_file("wts_tf_gen.bin", 11608064 * transformer_block_num_, wts_ptr_);

    { // q-bmm wts
      uint8_t* bmm_wts = (uint8_t*)wts_.at(Alias("tf_0_q_bmm_0_in_1")).data;
      uint8_t bmm_wts_zp =
          *(uint8_t*)wts_.at(Alias("tf_0_q_bmm_0_in_1_zp")).data;
      auto cache_iter = node_cache.find("BMM_IFM2");
      if (cache_iter == node_cache.end()) {
        std::vector<char> bmm_ifm2_cache(25 * 64 * 480 * sizeof(uint16_t), 0);
        uint16_t* bmm_ifm2 = (uint16_t*)bmm_ifm2_cache.data();
        // 25x64x475 -> 25x64x480 plus unpack to uint16
        for (int i = 0; i < 25 * 64; i++) {
          for (int j = 0; j < 475; j++) {
            bmm_ifm2[i * 480 + j] = bmm_wts[i * 475 + j];
          }
          for (int j = 0; j < 5; j++) {
            bmm_ifm2[i * 480 + 475 + j] = bmm_wts_zp; // zp padding
          }
        }
        node_cache.emplace("BMM_IFM2", std::move(bmm_ifm2_cache));
      }
    }
  } else if (subgraph_id_ == SUBGRAPH_ID::GT_CACHE_FRAMES_SLICE) {
    cache_frame_s_ = *((float*)(wts_.at(Alias("cache_frame_s")).data));
    cache_frame_zp_ = *((uint16_t*)(wts_.at(Alias("cache_frame_zp")).data));
  }
}

void MyCustomOpGT1_3::InitTransformerBlockWeights(
    std::unordered_map<std::string, flexmlrt::client::ErtIoTypeNew>& wts_,
    int8_t* wts_ptr, int tf_idx) {
  int8_t* ori_wts_ptr = wts_ptr;
  int8_t* rtp_ptr = nullptr;
  if (subgraph_id_ == SUBGRAPH_ID::GT_QKV ||
      subgraph_id_ == SUBGRAPH_ID::GT_MATMUL_REDUCE ||
      subgraph_id_ == SUBGRAPH_ID::GT_SM_LINEAR_OUT_FEED_FORWARD ||
      subgraph_id_ == SUBGRAPH_ID::GT_LN_MATMUL_ADD_LN ||
      subgraph_id_ == SUBGRAPH_ID::GT_FRONT) {
    rtp_ptr = wts_ptr + gt_global_rtp_offset_.at(subgraph_id_);
    wts_ptr += 1472;
  }
  const int8_t* rtp_ptr_start = rtp_ptr;
  std::string ifm_s_name, ifm_z_name, wts_s_name, wts_z_name, ofm_s_name,
      ofm_z_name, wts_w_name;
  std::string scale_s_name, scale_z_name, bias_s_name, bias_z_name, scale_name,
      bias_name;
  size_t total_wts_bytes = 0, wts_sz_loaded = 0;
  if (subgraph_id_ == SUBGRAPH_ID::GT_SM_LINEAR_OUT_FEED_FORWARD) {
    ifm_s_name = Alias(str_fmt("tf_%d_linear_out_mmb_in_s", tf_idx));
    ifm_z_name = Alias(str_fmt("tf_%d_linear_out_mmb_in_zp", tf_idx));
    wts_s_name = Alias(str_fmt("tf_%d_linear_out_mmb_wts_s", tf_idx));
    wts_z_name = Alias(str_fmt("tf_%d_linear_out_mmb_wts_zp", tf_idx));
    wts_w_name = Alias(str_fmt("tf_%d_linear_out_mmb_wts", tf_idx));
    bias_s_name = Alias(str_fmt("tf_%d_linear_out_mmb_bias_s", tf_idx));
    bias_z_name = Alias(str_fmt("tf_%d_linear_out_mmb_bias_zp", tf_idx));
    bias_name = Alias(str_fmt("tf_%d_linear_out_mmb_bias", tf_idx));
    ofm_s_name = Alias(str_fmt("tf_%d_linear_out_mmb_out_s", tf_idx));
    ofm_z_name = Alias(str_fmt("tf_%d_linear_out_mmb_out_zp", tf_idx));
    wts_sz_loaded = GT_MMB_WTS_convert_ptr(
        wts_, wts_ptr, rtp_ptr, ifm_s_name, ifm_z_name, wts_s_name, wts_z_name,
        bias_s_name, bias_z_name, ofm_s_name, ofm_z_name, wts_w_name, bias_name,
        25, 512, 512, 25, 32, 64, 16, false, model_version_);
    VAIML_DEBUG_PRINT(
        "Finish to format WTS for linear_out_feed_forward subgraph-", tf_idx,
        "-> MM ", wts_sz_loaded, "loaded",
        " , rtp offset: ", rtp_ptr - rtp_ptr_start);
    // update wts ptr position
    wts_ptr += 584704 - wts_sz_loaded + 64;
    rtp_ptr += 64;

    ifm_s_name = Alias(str_fmt("tf_%d_linear_out_add_in_0_s", tf_idx));
    ifm_z_name = Alias(str_fmt("tf_%d_linear_out_add_in_0_zp", tf_idx));
    wts_s_name = Alias(str_fmt("tf_%d_linear_out_add_in_1_s", tf_idx));
    wts_z_name = Alias(str_fmt("tf_%d_linear_out_add_in_1_zp", tf_idx));
    ofm_s_name = Alias(str_fmt("tf_%d_linear_out_add_out_s", tf_idx));
    ofm_z_name = Alias(str_fmt("tf_%d_linear_out_add_out_zp", tf_idx));
    wts_w_name = "";
    // no wts name change for add in middle
    wts_sz_loaded = GT_ADD_WTS_QDQ_convert(
        wts_, wts_ptr, rtp_ptr, ifm_s_name, ifm_z_name, wts_s_name, wts_z_name,
        ofm_s_name, ofm_z_name, wts_w_name, 25 * 512, model_version_);
    VAIML_DEBUG_PRINT(
        "Finish to format WTS for linear_out_feed_forward subgraph-", tf_idx,
        "-> ADD3", wts_sz_loaded,
        "loaded, rtp offset: ", rtp_ptr - rtp_ptr_start);

    ifm_s_name = Alias(str_fmt("tf_%d_ln_1_in_s", tf_idx));
    ifm_z_name = Alias(str_fmt("tf_%d_ln_1_in_zp", tf_idx));
    scale_s_name = Alias(str_fmt("tf_%d_ln_1_wts_s", tf_idx));
    scale_z_name = Alias(str_fmt("tf_%d_ln_1_wts_zp", tf_idx));
    bias_s_name = Alias(str_fmt("tf_%d_ln_1_bias_s", tf_idx));
    bias_z_name = Alias(str_fmt("tf_%d_ln_1_bias_zp", tf_idx));
    ofm_s_name = Alias(str_fmt("tf_%d_ln_1_out_s", tf_idx));
    ofm_z_name = Alias(str_fmt("tf_%d_ln_1_out_zp", tf_idx));
    scale_name = Alias(str_fmt("tf_%d_ln_1_wts", tf_idx));
    bias_name = Alias(str_fmt("tf_%d_ln_1_bias", tf_idx));
    wts_sz_loaded =
        GT_LN_WTS_convert(wts_, wts_ptr, ifm_s_name, ifm_z_name, scale_s_name,
                          scale_z_name, bias_s_name, bias_z_name, ofm_s_name,
                          ofm_z_name, scale_name, bias_name, model_version_);
    VAIML_DEBUG_PRINT(
        "Finish to format WTS for linear_out_feed_forward subgraph-", tf_idx,
        "-> LN", wts_sz_loaded);

    // feed forward, first part from matmul, second from add
    ifm_s_name = Alias(str_fmt("tf_%d_feed_forward_0_mmb_in_s", tf_idx));
    ifm_z_name = Alias(str_fmt("tf_%d_feed_forward_0_mmb_in_zp", tf_idx));
    wts_s_name = Alias(str_fmt("tf_%d_feed_forward_0_mmb_wts_s", tf_idx));
    wts_z_name = Alias(str_fmt("tf_%d_feed_forward_0_mmb_wts_zp", tf_idx));
    wts_w_name = Alias(str_fmt("tf_%d_feed_forward_0_mmb_wts", tf_idx));
    bias_s_name = Alias(str_fmt("tf_%d_feed_forward_0_mmb_bias_s", tf_idx));
    bias_z_name = Alias(str_fmt("tf_%d_feed_forward_0_mmb_bias_zp", tf_idx));
    ofm_s_name = Alias(str_fmt("tf_%d_feed_forward_0_mmb_out_s", tf_idx));
    ofm_z_name = Alias(str_fmt("tf_%d_feed_forward_0_mmb_out_zp", tf_idx));
    bias_name = Alias(str_fmt("tf_%d_feed_forward_0_mmb_bias", tf_idx));
    wts_sz_loaded = GT_MMB_WTS_convert_ptr(
        wts_, wts_ptr, rtp_ptr, ifm_s_name, ifm_z_name, wts_s_name, wts_z_name,
        bias_s_name, bias_z_name, ofm_s_name, ofm_z_name, wts_w_name, bias_name,
        25, 512, 4096, 25, 32, 64, 16, false, model_version_);
    VAIML_DEBUG_PRINT(
        "Finish to format WTS for linear_out_feed_forward subgraph-", tf_idx,
        "-> MM_W1", wts_sz_loaded,
        "loaded, rtp offset: ", rtp_ptr - rtp_ptr_start);
    wts_ptr += 4677632 - wts_sz_loaded + 64;
    rtp_ptr += 64;

    ifm_s_name = Alias(str_fmt("tf_%d_feed_forward_1_mmb_in_s", tf_idx));
    ifm_z_name = Alias(str_fmt("tf_%d_feed_forward_1_mmb_in_zp", tf_idx));
    wts_s_name = Alias(str_fmt("tf_%d_feed_forward_1_mmb_wts_s", tf_idx));
    wts_z_name = Alias(str_fmt("tf_%d_feed_forward_1_mmb_wts_zp", tf_idx));
    wts_w_name = Alias(str_fmt("tf_%d_feed_forward_1_mmb_wts", tf_idx));
    bias_s_name = Alias(str_fmt("tf_%d_feed_forward_1_mmb_bias_s", tf_idx));
    bias_z_name = Alias(str_fmt("tf_%d_feed_forward_1_mmb_bias_zp", tf_idx));
    ofm_s_name = Alias(str_fmt("tf_%d_feed_forward_1_mmb_out_s", tf_idx));
    ofm_z_name = Alias(str_fmt("tf_%d_feed_forward_1_mmb_out_zp", tf_idx));
    bias_name = Alias(str_fmt("tf_%d_feed_forward_1_mmb_bias", tf_idx));
    wts_sz_loaded = GT_MMB_WTS_convert_ptr(
        wts_, wts_ptr, rtp_ptr, ifm_s_name, ifm_z_name, wts_s_name, wts_z_name,
        bias_s_name, bias_z_name, ofm_s_name, ofm_z_name, wts_w_name, bias_name,
        25, 4096, 512, 25, 32, 64, 16, false, model_version_);
    VAIML_DEBUG_PRINT(
        "Finish to format WTS for linear_out_feed_forward subgraph-", tf_idx,
        "-> MM_W2 ", wts_sz_loaded,
        " loaded, rtp offset: ", rtp_ptr - rtp_ptr_start);
    wts_ptr += 4498432 - wts_sz_loaded + 64;
    rtp_ptr += 64;

    ifm_s_name = Alias(str_fmt("tf_%d_feed_forward_add_in_0_s", tf_idx));
    ifm_z_name = Alias(str_fmt("tf_%d_feed_forward_add_in_0_zp", tf_idx));
    wts_s_name = Alias(str_fmt("tf_%d_feed_forward_add_in_1_s", tf_idx));
    wts_z_name = Alias(str_fmt("tf_%d_feed_forward_add_in_1_zp", tf_idx));
    ofm_s_name = Alias(str_fmt("tf_%d_feed_forward_add_out_s", tf_idx));
    ofm_z_name = Alias(str_fmt("tf_%d_feed_forward_add_out_zp", tf_idx));
    wts_w_name = "";
    wts_sz_loaded = GT_ADD_WTS_QDQ_convert(
        wts_, wts_ptr, rtp_ptr, ifm_s_name, ifm_z_name, wts_s_name, wts_z_name,
        ofm_s_name, ofm_z_name, wts_w_name, 25 * 512, model_version_);
    VAIML_DEBUG_PRINT(
        "Finish to format WTS for linear_out_feed_forward subgraph-", tf_idx,
        "-> ADD4", wts_sz_loaded,
        " loaded, rtp offset: ", rtp_ptr - rtp_ptr_start);
  } else if (subgraph_id_ == SUBGRAPH_ID::GT_QKV) {
    ifm_s_name = Alias(str_fmt("tf_%d_ln_0_in_s", tf_idx));
    ifm_z_name = Alias(str_fmt("tf_%d_ln_0_in_zp", tf_idx));
    scale_s_name = Alias(str_fmt("tf_%d_ln_0_wts_s", tf_idx));
    scale_z_name = Alias(str_fmt("tf_%d_ln_0_wts_zp", tf_idx));
    bias_s_name = Alias(str_fmt("tf_%d_ln_0_bias_s", tf_idx));
    bias_z_name = Alias(str_fmt("tf_%d_ln_0_bias_zp", tf_idx));
    ofm_s_name = Alias(str_fmt("tf_%d_ln_0_out_s", tf_idx));
    ofm_z_name = Alias(str_fmt("tf_%d_ln_0_out_zp", tf_idx));
    scale_name = Alias(str_fmt("tf_%d_ln_0_wts", tf_idx));
    bias_name = Alias(str_fmt("tf_%d_ln_0_bias", tf_idx));
    wts_sz_loaded =
        GT_LN_WTS_convert(wts_, wts_ptr, ifm_s_name, ifm_z_name, scale_s_name,
                          scale_z_name, bias_s_name, bias_z_name, ofm_s_name,
                          ofm_z_name, scale_name, bias_name, model_version_);
    VAIML_DEBUG_PRINT("Finish to format WTS for qkv subgraph-", tf_idx,
                      "-> LN ", wts_sz_loaded, " loaded");
    std::string qkv_mm_wts_name = Alias(str_fmt("tf_%d_kqv_mm_wts", tf_idx));
    auto wts_mmb_splitted = SplitTransformerHeadMMWts(qkv_mm_wts_name, wts_);
    int fan_id = 0;
    for (std::string fanout : std::vector<std::string>({"q", "k", "v"})) {
      ifm_s_name = Alias(str_fmt("tf_%d_kqv_mm_in_s", tf_idx));
      ifm_z_name = Alias(str_fmt("tf_%d_kqv_mm_in_zp", tf_idx));
      wts_s_name = Alias(str_fmt("tf_%d_kqv_mm_wts_s", tf_idx));
      wts_z_name = Alias(str_fmt("tf_%d_kqv_mm_wts_zp", tf_idx));
      bias_s_name =
          Alias(str_fmt("tf_%d_kqv_mm_bias_%s_s", tf_idx, fanout.c_str()));
      bias_z_name =
          Alias(str_fmt("tf_%d_kqv_mm_bias_%s_zp", tf_idx, fanout.c_str()));
      ofm_s_name =
          Alias(str_fmt("tf_%d_kqv_mm_out_%s_s", tf_idx, fanout.c_str()));
      ofm_z_name =
          Alias(str_fmt("tf_%d_kqv_mm_out_%s_zp", tf_idx, fanout.c_str()));
      bias_name =
          Alias(str_fmt("tf_%d_kqv_mm_bias_%s", tf_idx, fanout.c_str()));

      wts_sz_loaded = GT_MMB_WTS_convert_raw_ptr(
          wts_, wts_ptr, rtp_ptr, (int8_t*)wts_mmb_splitted[fan_id].data(),
          ifm_s_name, ifm_z_name, wts_s_name, wts_z_name, bias_s_name,
          bias_z_name, ofm_s_name, ofm_z_name, bias_name, 25, 512, 512, 25,
          32 /*sv_M*/, 64 /*sv_K*/, 16 /*sv_N*/, model_version_);
      ++fan_id;
      wts_ptr += 584704 - wts_sz_loaded + 64;
      rtp_ptr += 64;
      VAIML_DEBUG_PRINT("Finish to format WTS for qkv subgraph-", tf_idx,
                        "-> MMB ", fanout, ", ", wts_sz_loaded,
                        "loaded, rtp offset", rtp_ptr - rtp_ptr_start);
    } // end of qkv matmul + add

    // skip linear_v reshape + transpose + concat don't have weights

    VAIML_DEBUG_PRINT("Finish loading wts for qkv MMB");
    // start of linear_q reshape + transpose + mul + reshape + transpose + bmm
    ifm_s_name = Alias(str_fmt("tf_%d_q_mul_0_in_s", tf_idx));
    ifm_z_name = Alias(str_fmt("tf_%d_q_mul_0_in_zp", tf_idx));
    wts_s_name = Alias(str_fmt("tf_%d_q_mul_0_wts_s", tf_idx));
    wts_z_name = Alias(str_fmt("tf_%d_q_mul_0_wts_zp", tf_idx));
    ofm_s_name = Alias(str_fmt("tf_%d_q_mul_0_out_s", tf_idx));
    ofm_z_name = Alias(str_fmt("tf_%d_q_mul_0_out_zp", tf_idx));
    wts_w_name = Alias(str_fmt("tf_%d_q_mul_0_wts", tf_idx));
    wts_sz_loaded = GT_MUL_WTS_convert(
        wts_, wts_ptr, rtp_ptr, ifm_s_name, ifm_z_name, wts_s_name, wts_z_name,
        ofm_s_name, ofm_z_name, wts_w_name, model_version_);
    VAIML_DEBUG_PRINT2("Finish to format WTS for qkv subgraph-", tf_idx,
                       "-> Mul ", wts_sz_loaded,
                       "loaded, rtp offset: ", rtp_ptr - rtp_ptr_start);
    rtp_ptr += 64; // shall skip 64 bytes to continue concat related param

    // end of linear_q reshape + transpose + mul + reshape + transpose + bmm

    // skip linear_k reshape + transpose + concat don't have weights

    // qdq rtps for k,v gather-concat
    ifm_s_name = Alias(str_fmt("tf_%d_v_cache_in_s", tf_idx));
    ifm_z_name = Alias(str_fmt("tf_%d_v_cache_in_zp", tf_idx));
    ofm_s_name = Alias(str_fmt("tf_%d_v_cache_out_s", tf_idx));
    ofm_z_name = Alias(str_fmt("tf_%d_v_cache_out_zp", tf_idx));
    wts_sz_loaded =
        GT_QDQ_convert(wts_, wts_ptr, rtp_ptr, ifm_s_name, ifm_z_name,
                       ofm_s_name, ofm_z_name, model_version_);
    VAIML_DEBUG_PRINT2("Finish to format WTS for qkv subgraph-", tf_idx,
                       "-> v concat QDQ ", wts_sz_loaded,
                       "loaded, rtp offset: ", rtp_ptr - rtp_ptr_start);

    ifm_s_name = Alias(str_fmt("tf_%d_k_cache_in_s", tf_idx));
    ifm_z_name = Alias(str_fmt("tf_%d_k_cache_in_zp", tf_idx));
    ofm_s_name = Alias(str_fmt("tf_%d_k_cache_out_s", tf_idx));
    ofm_z_name = Alias(str_fmt("tf_%d_k_cache_out_zp", tf_idx));
    wts_sz_loaded =
        GT_QDQ_convert(wts_, wts_ptr, rtp_ptr, ifm_s_name, ifm_z_name,
                       ofm_s_name, ofm_z_name, model_version_);
    VAIML_DEBUG_PRINT2("Finish to format WTS for qkv subgraph-", tf_idx,
                       "-> k concat QDQ ", wts_sz_loaded,
                       "loaded, rtp offset: ", rtp_ptr - rtp_ptr_start);
  } else if (subgraph_id_ == SUBGRAPH_ID::GT_MATMUL_REDUCE) {
    // now bmm_1 moved to matmul_reduce
    ifm_s_name = Alias(str_fmt("tf_%d_q_bmm_0_in_0_s", tf_idx));
    ifm_z_name = Alias(str_fmt("tf_%d_q_bmm_0_in_0_zp", tf_idx));
    wts_s_name = Alias(str_fmt("tf_%d_q_bmm_0_in_1_s", tf_idx));
    wts_z_name = Alias(str_fmt("tf_%d_q_bmm_0_in_1_zp", tf_idx));
    ofm_s_name = Alias(str_fmt("tf_%d_q_bmm_0_out_s", tf_idx));
    ofm_z_name = Alias(str_fmt("tf_%d_q_bmm_0_out_zp", tf_idx));
    wts_sz_loaded = GT_BMM_WTS_convert<uint8_t /*wts dtype*/>(
        wts_, wts_ptr, rtp_ptr, ifm_s_name, ifm_z_name, wts_s_name, wts_z_name,
        ofm_s_name, ofm_z_name, wts_w_name, false, 8, 64, 512, 16, 64, 32,
        model_version_); // 8 64 475,  N / row / col
    VAIML_DEBUG_PRINT2("Finish to format WTS for matmul_reduce subgraph-",
                       tf_idx, "-> BMM ", wts_sz_loaded,
                       "loaded, rtp_offset: ", rtp_ptr - rtp_ptr_start);

    ifm_s_name = Alias(str_fmt("tf_%d_k_bmm_in_0_s", tf_idx));
    ifm_z_name = Alias(str_fmt("tf_%d_k_bmm_in_0_zp", tf_idx));
    wts_s_name = Alias(str_fmt("tf_%d_k_bmm_in_1_s", tf_idx));
    wts_z_name = Alias(str_fmt("tf_%d_k_bmm_in_1_zp", tf_idx));
    ofm_s_name = Alias(str_fmt("tf_%d_k_bmm_out_s", tf_idx));
    ofm_z_name = Alias(str_fmt("tf_%d_k_bmm_out_zp", tf_idx));
    wts_sz_loaded = GT_BMM_WTS_convert(
        wts_, wts_ptr, rtp_ptr, ifm_s_name, ifm_z_name, wts_s_name, wts_z_name,
        ofm_s_name, ofm_z_name, "", true, 25, 64, 512, 32, 64, 32,
        model_version_); // 25 64 475, N / row / col
    VAIML_DEBUG_PRINT2("Finish to format WTS for Matmul-Reduce subgraph-",
                       tf_idx, "-> BMM", wts_sz_loaded,
                       "loaded, rtp_offset: ", rtp_ptr - rtp_ptr_start);

    ifm_s_name = Alias(str_fmt("tf_%d_q_add_0_in_0_s", tf_idx));
    ifm_z_name = Alias(str_fmt("tf_%d_q_add_0_in_0_zp", tf_idx));
    wts_s_name = Alias(str_fmt("tf_%d_q_add_0_in_1_s", tf_idx));
    wts_z_name = Alias(str_fmt("tf_%d_q_add_0_in_1_zp", tf_idx));
    ofm_s_name = Alias(str_fmt("tf_%d_q_add_0_out_s", tf_idx));
    ofm_z_name = Alias(str_fmt("tf_%d_q_add_0_out_zp", tf_idx));
    wts_w_name = "";
    wts_sz_loaded = GT_ADD_WTS_QDQ_convert(
        wts_, wts_ptr, rtp_ptr, ifm_s_name, ifm_z_name, wts_s_name, wts_z_name,
        ofm_s_name, ofm_z_name, wts_w_name, 8 * 25 * 475, model_version_);
    VAIML_DEBUG_PRINT2("Finish to format WTS for Matmul-Reduce subgraph-",
                       tf_idx, "-> ADD", wts_sz_loaded,
                       " loaded, rtp offset: ", rtp_ptr - rtp_ptr_start,
                       "rtp_ptr absolute offset: ", rtp_ptr - wts_ptr_);
    ifm_s_name = Alias(str_fmt("tf_%d_q_mul_1_in_0_s", tf_idx));
    ifm_z_name = Alias(str_fmt("tf_%d_q_mul_1_in_0_zp", tf_idx));
    wts_s_name = Alias(str_fmt("tf_%d_q_mul_1_in_1_s", tf_idx));
    wts_z_name = Alias(str_fmt("tf_%d_q_mul_1_in_1_zp", tf_idx));
    ofm_s_name = Alias(str_fmt("tf_%d_q_mul_1_out_s", tf_idx));
    ofm_z_name = Alias(str_fmt("tf_%d_q_mul_1_out_zp", tf_idx));
    wts_w_name = "";
    wts_sz_loaded = GT_MUL_WTS_QDQ_convert(
        wts_, wts_ptr, rtp_ptr, ifm_s_name, ifm_z_name, wts_s_name, wts_z_name,
        ofm_s_name, ofm_z_name, wts_w_name, 8 * 25 * 480, model_version_);
    VAIML_DEBUG_PRINT2("Finish to format WTS for Matmul-Reduce subgraph-",
                       tf_idx, "-> Mul ", wts_sz_loaded,
                       " loaded, rtp offset: ", rtp_ptr - rtp_ptr_start);
    // Mul-4
    // if (subgraph_index_matmul_reduce > 0) {
    //   ifm_s_name = NAME_CONCAT("/Add_", 0, 5, "_output_0_scale");
    //   // mul2_scale.at(subgraph_index_matmul_reduce);
    //   ifm_z_name = NAME_CONCAT("/Mul_", 3, 5, "_output_0_zero_point");
    //   wts_s_name = "/Softmax_13_output_0_scale";
    //   wts_z_name = "/Slice_output_0_zero_point";
    //   ofm_s_name = mul2_scale.at(subgraph_index_matmul_reduce);
    //   ofm_z_name = NAME_CONCAT("/Mul_", 4, 5, "_output_0_zero_point");
    //   wts_w_name = "";
    // } else {
    //   ifm_s_name = "/Add_output_0_scale";
    //   ifm_z_name = "/Mul_3_output_0_zero_point";
    //   wts_s_name = "/Softmax_13_output_0_scale";
    //   wts_z_name = "/Slice_output_0_zero_point";
    //   ofm_s_name = "/Mul_4_output_0_scale";
    //   ofm_z_name = "/Mul_4_output_0_zero_point";
    //   wts_w_name = "";
    // }
    // FIXME should be q_mul_1_out_s
    // ifm_s_name = Alias(str_fmt("tf_%d_q_add_0_out_s", tf_idx));
    ifm_s_name = Alias(str_fmt("tf_%d_q_mul_1_out_s", tf_idx));
    ifm_z_name = Alias(str_fmt("tf_%d_q_mul_1_out_zp", tf_idx));
    wts_s_name = Alias(str_fmt("tf_%d_q_mul_2_in_1_s", tf_idx));
    wts_z_name = Alias(str_fmt("tf_%d_q_mul_2_in_1_zp", tf_idx));
    ofm_s_name = Alias(str_fmt("tf_%d_q_mul_2_out_s", tf_idx));
    ofm_z_name = Alias(str_fmt("tf_%d_q_mul_2_out_zp", tf_idx));
    wts_sz_loaded = GT_MUL_WTS_QDQ_convert(
        wts_, wts_ptr, rtp_ptr, ifm_s_name, ifm_z_name, wts_s_name, wts_z_name,
        ofm_s_name, ofm_z_name, wts_w_name, 8 * 25 * 480, model_version_);
    VAIML_DEBUG_PRINT2("Finish to format WTS for Matmul-Reduce subgraph-",
                       tf_idx, "-> Mul ", wts_sz_loaded,
                       "loaded, rtp offset: ", rtp_ptr - rtp_ptr_start);

    ifm_s_name = Alias(str_fmt("tf_%d_q_add_1_in_0_s", tf_idx));
    ifm_z_name = Alias(str_fmt("tf_%d_q_add_1_in_0_zp", tf_idx));
    wts_s_name = Alias(str_fmt("tf_%d_q_add_1_in_1_s", tf_idx));
    wts_z_name = Alias(str_fmt("tf_%d_q_add_1_in_1_zp", tf_idx));
    ofm_s_name = Alias(str_fmt("tf_%d_q_add_1_out_s", tf_idx));
    ofm_z_name = Alias(str_fmt("tf_%d_q_add_1_out_zp", tf_idx));
    wts_sz_loaded = GT_ADD_WTS_QDQ_convert(
        wts_, wts_ptr, rtp_ptr, ifm_s_name, ifm_z_name, wts_s_name, wts_z_name,
        ofm_s_name, ofm_z_name, wts_w_name, 8 * 25 * 480, model_version_);
    VAIML_DEBUG_PRINT2("Finish to format WTS for Matmul-Reduce subgraph-",
                       tf_idx, "-> ADD", wts_sz_loaded,
                       "loaded, rtp offset: ", rtp_ptr - rtp_ptr_start);

    // softmax and bmm2 is also in matmul reduce now
    ifm_s_name = Alias(str_fmt("tf_%d_q_softmax_in_s", tf_idx));
    ifm_z_name = Alias(str_fmt("tf_%d_q_softmax_in_zp", tf_idx));
    ofm_s_name = Alias(str_fmt("tf_%d_q_softmax_out_s", tf_idx));
    ofm_z_name = Alias(str_fmt("tf_%d_q_softmax_out_zp", tf_idx));
    // softmax shall take only wts size and no rtp will be used
    wts_sz_loaded = GT_SOFTMAX_WTS_convert(
        wts_, wts_ptr, ifm_s_name, ifm_z_name, ofm_s_name, ofm_z_name,
        480 /*K*/, 475 /*K_valid*/, model_version_);
    VAIML_DEBUG_PRINT2("Finish to format WTS for matmul_reduce subgraph-",
                       tf_idx, "-> Softmax", wts_sz_loaded,
                       "loaded, rtp offset", rtp_ptr - rtp_ptr_start);

    ifm_s_name = Alias(str_fmt("tf_%d_q_bmm_1_in_0_s", tf_idx));
    ifm_z_name = Alias(str_fmt("tf_%d_q_bmm_1_in_0_zp", tf_idx));
    wts_s_name = Alias(str_fmt("tf_%d_q_bmm_1_in_1_s", tf_idx));
    wts_z_name = Alias(str_fmt("tf_%d_q_bmm_1_in_1_zp", tf_idx));
    ofm_s_name = Alias(str_fmt("tf_%d_q_bmm_1_out_s", tf_idx));
    ofm_z_name = Alias(str_fmt("tf_%d_q_bmm_1_out_zp", tf_idx));
    wts_sz_loaded = GT_BMM_WTS_convert(
        wts_, wts_ptr, rtp_ptr, ifm_s_name, ifm_z_name, wts_s_name, wts_z_name,
        ofm_s_name, ofm_z_name, wts_w_name, false, 25, 512, 64, 16, 128, 16,
        model_version_); // this will be the last bmm in our, padding occurs!
    VAIML_DEBUG_PRINT(
        "Finish to format WTS for linear_out_feed_forward subgraph-", tf_idx,
        "-> MM2 ", wts_sz_loaded,
        ", loaded, rtp offset: ", rtp_ptr - rtp_ptr_start);
    // note we are not using up all available wts buffer, the pointer is updated
    // upon next enter with subgraph_id_ == sm_feed_forward
  }

  if (subgraph_id_ == SUBGRAPH_ID::GT_QKV) {
    // rtp layer k, q, v
    // wts layer q, k, v
    // need to swap rtp of k, q
    int8_t swap_buffer[128];
    memcpy(swap_buffer, ori_wts_ptr, 128);
    memcpy(ori_wts_ptr, ori_wts_ptr + 128, 128);
    memcpy(ori_wts_ptr + 128, swap_buffer, 128);
  }
}

int32_t MyCustomOpGT1_3::Slice144Compute_GT(Ort::KernelContext& ctx) const {
  auto inputvalue = ctx.GetInput(0);
  const void* input = inputvalue.GetTensorRawData();
  auto output_shapes = ort_output_shapes_;
  auto& output_shape = output_shapes[0];
  auto ortvalue = ctx.GetOutput(0, output_shape.data(), output_shape.size());
  void* output = ortvalue.GetTensorMutableRawData();
  int zp = cache_frame_zp_;
  float scale = cache_frame_s_;
  for (int i = 0; i < 3 * 80; i++) {
    ((float*)output)[i] = (((const uint16_t*)input)[100 * 80 + i] - zp) * scale;
  }
  return 3 * 80 * sizeof(float);
}

int32_t MyCustomOpGT1_3::MainBlockInputs(Ort::KernelContext& ctx) const {
  /* onnx order:
  --- GT
      "/out/Add_output_0_QuantizeLinear_Output",
      "inp_cache_k",
      "inp_cache_v",
      "mask_QuantizeLinear_Output"
  --- GTC
      "/out/Add_output_0_QuantizeLinear_Output",
      "inp_cache_k",
      "inp_cache_v",
      "mask_q_to_dq",
      "mask_q_to_dq_token_240" (not used)
    txn order:
      add-ln, bmm, v gather, k gather,
      |mul after reducemin(0:475 of mask), mul before reducemin(475:950 of
    mask)|
  */
  auto get_ort_pointer = [&ctx](size_t idx) {
    return ctx.GetInput(idx).GetTensorRawData();
  };
  // { // conv
  //   const void* input = get_ort_pointer(0);
  //   memcpy(ifm_ptr_, input, 103 * 80 * sizeof(uint16_t));
  // }
  { // TODO(zhonnian): add-ln, will be removed for super cmdlist
    const void* input = get_ort_pointer(0);
    memcpy(ifm_ptr_, input, 25 * 512 * sizeof(uint16_t));
  }

  for (int tf_iter = 0; tf_iter < transformer_block_num_; tf_iter++) {
    size_t base_ddr = tf_iter * 4300800;
    { // bmm
      int8_t* bmm_input = ifm_ptr_ + GT_FRONT_SZ + base_ddr + 25600;
      const std::vector<char>& bmm_ifm2_cache = node_cache.at("BMM_IFM2");
      memcpy(bmm_input, bmm_ifm2_cache.data(), bmm_ifm2_cache.size());
    }
    // v k gather
    // slice and pad
    // 36x8x450x64 -> 1x8x475x64
    for (auto concat_input_idx : {2, 1}) {
      bool is_v_gather = concat_input_idx == 2;
      size_t word_offset = is_v_gather ? 1561600 : 2048000;
      int8_t* concat_input_ptr =
          ifm_ptr_ + GT_FRONT_SZ + base_ddr + word_offset;
      const void* input = get_ort_pointer(concat_input_idx);
      for (int i = 0; i < 8; i++) {
        float_to_bfloat16_avx512_unrolled(
            (const float*)input + tf_iter * 8 * 450 * 64 + i * 450 * 64,
            (uint16_t*)concat_input_ptr + i * 475 * 64, 450 * 64);
      }
    }

    {
      // Mul before & after ReduceMin
      // need to perform slicing ?? not for single tf block
      // 8x25x950 -> 8x25x475(0-475, padded to 480) after reducemin
      // 8x25x950 -> 8x25x475(475-950, padded to 480) before reducemin
      int8_t* mul_input_ptr = ifm_ptr_ + GT_FRONT_SZ + base_ddr + 2534400;
      const void* input = get_ort_pointer(3);
      auto cache_iter = node_cache.find("Slice");
      if (cache_iter == node_cache.end()) {
        for (int i = 0; i < 8 * 25; i++) {
          memcpy((uint16_t*)mul_input_ptr + i * 480,
                 (const uint16_t*)input + i * 950, 475 * sizeof(uint16_t));
          memcpy((uint16_t*)mul_input_ptr + 8 * 25 * 480 + i * 480,
                 (const uint16_t*)input + i * 950 + 475,
                 475 * sizeof(uint16_t));
        }
        // const void* slice_input = get_ort_pointer(3);
        // const void* slice_1_input = get_ort_pointer(2);
        // for (int i = 0; i < 8 * 25; i++) {
        //   memcpy((uint16_t*)mul_input_ptr + i * 480,
        //          (const uint16_t*)slice_input + i * 475,
        //          475 * sizeof(uint16_t));
        //   memcpy((uint16_t*)mul_input_ptr + 8 * 25 * 480 + i * 480,
        //          (const uint16_t*)slice_1_input + i * 475,
        //          475 * sizeof(uint16_t));
        // }
        // std::vector<char> slice_cache(8 * 25 * 480 * sizeof(uint16_t), 0);
        // memcpy(slice_cache.data(), mul_input_ptr, slice_cache.size());
        // node_cache.emplace("Slice", std::move(slice_cache));

        // std::vector<char> slice_1_cache(8 * 25 * 480 * sizeof(uint16_t), 0);
        // memcpy(slice_1_cache.data(),
        //        mul_input_ptr + 8 * 25 * 480 * sizeof(uint16_t),
        //        slice_1_cache.size());
        // node_cache.emplace("Slice_1", std::move(slice_1_cache));
      } else {
        const std::vector<char>& slice_cache = cache_iter->second;
        memcpy(mul_input_ptr, slice_cache.data(), slice_cache.size());
        const std::vector<char>& slice_1_cache = node_cache.at("Slice_1");
        memcpy(mul_input_ptr + 8 * 25 * 480 * sizeof(uint16_t),
               slice_1_cache.data(), slice_1_cache.size());
      }
    }
  }
  if (node_cache.find("Slice") != node_cache.end()) {
    node_cache.erase("Slice");
    node_cache.erase("Slice_1");
  }
  return 0;
}

int32_t MyCustomOpGT1_3::MainBlockOutputs(Ort::KernelContext& ctx) const {
  /*
   onnx order:
   --- GT
      "oup_lid_QuantizeLinear_Output",
      "oup_cache_v",
      "oup_cache_k",
      "/Add_179_output_0_QuantizeLinear_Output"
  --- GTC
      "oup_cache_v",
      "oup_cache_k",
      "/Add_159_output_0_QuantizeLinear_Output"
   */
  auto get_ort_pointer = [&ctx, this](size_t idx) {
    auto output_shapes = this->ort_output_shapes_;
    auto& output_shape = output_shapes[idx];
    auto ortvalue =
        ctx.GetOutput(idx, output_shape.data(), output_shape.size());
    return ortvalue.GetTensorMutableRawData();
  };

  bool has_oup_lid = (oup_lid_idx_ > 0);
  // for (auto split_idx : {1,2}) {
  //   void* data = get_ort_pointer(split_idx);
  //   bool is_v_split = (split_idx == 1);
  for (auto split_idx : {1, 2}) {
    void* data = get_ort_pointer(split_idx - !has_oup_lid);
    bool is_v_split = (split_idx == 1);
    size_t xrt_offset_kv = is_v_split ? 1561600 : 2048000;
    for (int tf_iter = 0; tf_iter < transformer_block_num_; tf_iter++) {
      size_t xrt_base_ddr = tf_iter * 4300800 + xrt_offset_kv + GT_FRONT_SZ;
      size_t ort_base_ddr = tf_iter * 8 * 450 * 64;
      for (int i = 0; i < 8; i++) {
        bfloat16_to_float_avx512_unrolled(
            (uint16_t*)(ofm_ptr_ + xrt_base_ddr) + i * 475 * 64 + 25 * 64,
            (float*)data + ort_base_ddr + i * 450 * 64, 450 * 64);
      }
    }
  }

  {
    // Add_179_output_0_QuantizeLinear_Output
    void* data = get_ort_pointer(3 - !has_oup_lid);
    memcpy(data, ofm_ptr_ + GT_FRONT_SZ + transformer_block_num_ * 4300800,
           25 * 512 * sizeof(uint16_t));
  }
  if (has_oup_lid) {
    // oup_lid
    void* data = get_ort_pointer(0);
    memcpy(data, ofm_ptr_ + GT_FRONT_SZ + oup_lid_idx_ * 4300800,
           25 * 512 * sizeof(uint16_t));
  }
  return 0;
}

void MyCustomOpGT1_3::Compute(const OrtApi* api,
                              OrtKernelContext* context) const {
  if (Ort::Global<void>::api_ == nullptr) {
    Ort::Global<void>::api_ = api;
  }
  Ort::KernelContext ctx(context);
  // CPU graph, skip hw runner
  if (subgraph_id_ == SUBGRAPH_ID::GT_CACHE_FRAMES_SLICE) {
    Slice144Compute_GT(ctx);
    return;
  }
  const void* input_deq_dup = nullptr;
  // prepare for input
  if (subgraph_id_ == SUBGRAPH_ID::GT_FRONT) {
    {
      auto num_inputs = ctx.GetInputCount();
      VAIML_DEBUG_PRINT("inputs number: ", num_inputs);
      auto inputvalue = ctx.GetInput(0); // 0 is not used
      const void* input = inputvalue.GetTensorRawData();
      auto type_and_shape = inputvalue.GetTensorTypeAndShapeInfo();
      auto num_elements = type_and_shape.GetElementCount();
      auto input_type = type_and_shape.GetElementType();
      memcpy(ifm_ptr_, input, num_elements * sizeof(uint16_t));
    }
  } else if (subgraph_id_ == SUBGRAPH_ID::GT_FRONT_MM) {
    {
      auto num_inputs = ctx.GetInputCount();
      VAIML_DEBUG_PRINT("inputs number: ", num_inputs);
      auto inputvalue = ctx.GetInput(0);
      const void* input = inputvalue.GetTensorRawData();
      auto type_and_shape = inputvalue.GetTensorTypeAndShapeInfo();
      // printf("shape of linear out input");
      // for (auto d : type_and_shape.GetShape()) {
      //   printf("%lld, ", d);
      // }
      // printf("type: %d\n", type_and_shape.GetElementType());
      // 25x512x19->25x512x20
      for (int i = 0; i < 25; ++i) {
        for (int j = 0; j < 512; ++j) {
          memcpy((uint16_t*)(ifm_ptr_ + 16480) + i * 10240 + j * 20,
                 (const uint16_t*)input + i * 9728 + j * 19,
                 19 * sizeof(uint16_t));
        }
      }
      // for (int i = 0; i < 128; ++i) {
      //   std::cout << std::setw(8) << std::hex << std::setfill('0')
      //             << *((uint32_t*)(ifm_ptr_ + 16480) + i) << "    "
      //             << *((uint32_t*)input + i) << std::endl;
      // }
    }
  } else if (subgraph_id_ == SUBGRAPH_ID::GT_LN_MATMUL_ADD_LN) {
    auto num_inputs = ctx.GetInputCount();
    auto inputvalue = ctx.GetInput(0);
    const void* input = inputvalue.GetTensorRawData();
    auto type_and_shape = inputvalue.GetTensorTypeAndShapeInfo();
    memcpy(ifm_ptr_, input, 25 * 512 * sizeof(uint16_t));
  } else if (subgraph_id_ == SUBGRAPH_ID::GT_TRANSFORMER_BLOCK) {
    MainBlockInputs(ctx);
    // to_file("ifm.bin", 4300800 * 2 + 25600, ifm_ptr_);
  }

  // hardware run
  auto output_shapes = ort_output_shapes_;
  VAIML_DEBUG_PRINT("    outputs number: ", output_shapes.size());
  auto err_status =
      runner_->run((void*)ifm_ptr_, (void*)wts_ptr_, (void*)ofm_ptr_);

  if (subgraph_id_ == GT_FRONT) {
    {
      int index = 0; // /Reshape_2_output_0_QuantizeLinear_Output
      auto& output_shape = output_shapes[index];
      auto ortvalue =
          ctx.GetOutput(index, output_shape.data(), output_shape.size());
      auto type_and_shape = ortvalue.GetTensorTypeAndShapeInfo();
      auto tensor_type = type_and_shape.GetElementType();
      auto num_elements = type_and_shape.GetElementCount();
      void* data = ortvalue.GetTensorMutableRawData();
      VAIML_DEBUG_PRINT("conv output num_elements ", num_elements);
      for (int i = 0; i < 25; ++i) {
        for (int j = 0; j < 512; ++j) {
          memcpy((uint16_t*)data + i * 9728 + j * 19,
                 (uint16_t*)(ofm_ptr_ + 16480) + i * 10240 + j * 20,
                 19 * sizeof(uint16_t));
          memset((uint16_t*)(ifm_ptr_ + 16480) + i * 10240 + j * 20 + 19, 0,
                 sizeof(uint16_t));
        }
      }
    }
  } else if (subgraph_id_ == GT_FRONT_MM) {
    int index = 0;
    auto& output_shape = output_shapes[index];
    auto ortvalue =
        ctx.GetOutput(index, output_shape.data(), output_shape.size());
    void* data = ortvalue.GetTensorMutableRawData();
    memcpy(data, ofm_ptr_ + 16480 + 552960, 25600);
  } else if (subgraph_id_ == SUBGRAPH_ID::GT_LN_MATMUL_ADD_LN) {
    int index = 0;
    auto& output_shape = output_shapes[index];
    auto ortvalue =
        ctx.GetOutput(index, output_shape.data(), output_shape.size());
    void* data = ortvalue.GetTensorMutableRawData();
    memcpy(data, ofm_ptr_, 25600);
  } else if (subgraph_id_ == SUBGRAPH_ID::GT_TRANSFORMER_BLOCK) {
    MainBlockOutputs(ctx);
    // to_file("ofm.bin", 4300800*2+25600, ofm_ptr_);
  }

  if (err_status == -2) {
    printf("BOARD CRASHED\n");
  }
}

MyCustomOpGT1_3::~MyCustomOpGT1_3() {}
} // namespace vaip_vaiml_custom_op
