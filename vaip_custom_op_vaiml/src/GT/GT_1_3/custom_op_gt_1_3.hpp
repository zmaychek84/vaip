/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */
#pragma once
#pragma once

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#endif

#include "../../common/hw_elf_runner.h"
#include "../../common/hw_runner.h"
#include "../../common/utils.h"
#include "../../common/vaiml_client.h"
#include "onnxruntime_api.hpp"
#include "vaip/vaip.hpp"
#include "vitis/ai/env_config.hpp"
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>

namespace vaip_vaiml_custom_op {
using namespace vaip_core;

class MyCustomOpGT1_3 : public CustomOpImp {
public:
  MyCustomOpGT1_3(std::shared_ptr<const PassContext> context,
                  const std::shared_ptr<MetaDefProto>& meta_def,
                  onnxruntime::Model* model);

  virtual ~MyCustomOpGT1_3();

private:
  virtual void Compute(const OrtApi* api,
                       OrtKernelContext* context) const override final;
  void LoadConstantsToWts(std::shared_ptr<const PassContext> context,
                          const std::shared_ptr<MetaDefProto>& meta_def);
  void PrepareHwRunner(std::vector<std::string>& v_txn_bins,
                       std::vector<std::string>& v_ctrl_pkt_bins,
                       std::vector<std::stringstream>& v_elf_istream,
                       std::vector<XRTRunOffset>& xrt_offset,
                       std::vector<KERNEL_NM>& v_kernel_indices,
                       std::vector<BO_ORDER>& v_bo_order, size_t& ifm_size,
                       size_t& ofm_size, size_t& wts_size, size_t& tmp_size);
  void InitWeights();
  int32_t Slice144Compute_GT(Ort::KernelContext& ctx) const;
  int32_t MyCustomOpGT1_3::MainBlockInputs(Ort::KernelContext& ctx) const;
  int32_t MyCustomOpGT1_3::MainBlockOutputs(Ort::KernelContext& ctx) const;

  void InitTransformerBlockWeights(
      std::unordered_map<std::string, flexmlrt::client::ErtIoTypeNew>& wts_,
      int8_t* wts_ptr, int tf_idx);
  SUBGRAPH_ID
  IdentifySubgraphVector(const std::shared_ptr<MetaDefProto>& meta_def);
  std::string Alias(const std::string& alias_initializer_name) const {
    return this->initializer_map_.at(alias_initializer_name);
  }

private:
  int oup_lid_idx_ = -1;
  // default to 36 for GT, could be updated from vaiml pass statistics
  int transformer_block_num_ = 36;
  // from developer designed alias to names in model
  std::unordered_map<std::string, std::string> initializer_map_;
  float cache_frame_s_;
  uint16_t cache_frame_zp_;
  std::unordered_map<std::string, flexmlrt::client::ErtIoTypeNew> wts_;
  std::vector<std::vector<char>> wts_buffers_;
  std::string model_version_;
  SUBGRAPH_ID subgraph_id_;
  std::string sg_name_;
  std::vector<std::vector<int64_t>> ort_output_shapes_;
  std::unique_ptr<hw_runner_base> runner_;
  int8_t* wts_ptr_;
  int8_t* ifm_ptr_;
  int8_t* ofm_ptr_;
  static std::map<std::string, std::vector<char>> node_cache;
  const std::map<SUBGRAPH_ID, size_t> gt_global_rtp_offset_ = {
      {SUBGRAPH_ID::GT_QKV, 0},
      {SUBGRAPH_ID::GT_MATMUL_REDUCE, 448},
      {SUBGRAPH_ID::GT_SM_LINEAR_OUT_FEED_FORWARD, 896},
      {SUBGRAPH_ID::GT_LN_MATMUL_ADD_LN, 896},
      {SUBGRAPH_ID::GT_FRONT_MM, 896}};
  int transformer_block_id_;
  const std::string vaiml_model_path_ = "vaiml_par_0";
};
} // namespace vaip_vaiml_custom_op