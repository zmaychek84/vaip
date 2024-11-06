#pragma once
#pragma once

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#endif

namespace vaiml_gt_1_3 {
constexpr unsigned transformer_block_num = 36;
};

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
      int8_t* wts_ptr);
  SUBGRAPH_ID
  IdentifySubgraphVector(const std::shared_ptr<MetaDefProto>& meta_def);

private:
  std::unordered_map<std::string, flexmlrt::client::ErtIoTypeNew> wts_;
  std::vector<std::vector<char>> wts_buffers_;
  std::string model_version_;
  SUBGRAPH_ID subgraph_id_;
  std::string sg_name_;
  std::vector<std::vector<int64_t>> ort_output_shapes_;
  hw_runner g;
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

  const std::map<int, std::string> bmm_scale_wts_prefix_ = {
      {0, ""},
      {1, "/Unsqueeze_411_output_0_scale"},
      {2, "/Slice_10_output_0_scale"},
      {3, "/Concat_138_output_0_scale"},
      {4, "/linear_k_4/Add_output_0_scale"},
      {5, "/linear_k_5/Add_output_0_scale"},
      {6, "/Slice_26_output_0_scale"},
      {7, "/Concat_170_output_0_scale"},
      {8, "/linear_k_8/Add_output_0_scale"},
      {9, "/Unsqueeze_611_output_0_scale"},
      {10, "/Unsqueeze_636_output_0_scale"},
      {11, "/Concat_202_output_0_scale"},
      {12, "/linear_k_12/Add_output_0_scale"},
      {13, "/Concat_218_output_0_scale"},
      {14, "/Slice_58_output_0_scale"},
      {15, "/Concat_234_output_0_scale"},
      {16, "/Concat_242_output_0_scale"},
      {17, "/Concat_250_output_0_scale"},
      {18, "/linear_k_18/Add_output_0_scale"},
      {19, "/linear_k_19/Add_output_0_scale"},
      {20, "/linear_k_20/Add_output_0_scale"},
      {21, "/linear_k_21/Add_output_0_scale"},
      {22, "/Concat_290_output_0_scale"},
      {23, "/Concat_298_output_0_scale"},
      {24, "/Unsqueeze_986_output_0_scale"},
      {25, "/Concat_314_output_0_scale"},
      {26, "/linear_k_26/Add_output_0_scale"},
      {27, "/Slice_110_output_0_scale"},
      {28, "/linear_k_28/Add_output_0_scale"},
      {29, "/Concat_346_output_0_scale"},
      {30, "/Concat_354_output_0_scale"},
      {31, "/Concat_362_output_0_scale"},
      {32, "/Concat_370_output_0_scale"},
      {33, "/Concat_378_output_0_scale"},
      {34, "/Concat_386_output_0_scale"},
      {35, "/linear_k_35/Add_output_0_scale"},
  };

  const std::map<int, std::string> mul_ifm_scale = {
      {0, "/Add_output_0_scale"},      {1, "/Add_5_output_0_scale"},
      {2, "/Add_12_output_0_scale"},   {3, "/Add_17_output_0_scale"},
      {4, "/Add_20_output_0_scale"},   {5, "/Add_25_output_0_scale"},
      {6, "/Add_30_output_0_scale"},   {7, "/Add_35_output_0_scale"},
      {8, "/Add_42_output_0_scale"},   {9, "/Add_45_output_0_scale"},
      {10, "/Add_50_output_0_scale"},  {11, "/Mul_58_output_0_scale"},
      {12, "/Add_60_output_0_scale"},  {13, "/Add_67_output_0_scale"},
      {14, "/Add_70_output_0_scale"},  {15, "/Add_75_output_0_scale"},
      {16, "/Add_82_output_0_scale"},  {17, "/Add_85_output_0_scale"},
      {18, "/Add_92_output_0_scale"},  {19, "/Add_95_output_0_scale"},
      {20, "/Add_100_output_0_scale"}, {21, "/Add_105_output_0_scale"},
      {22, "/Add_110_output_0_scale"}, {23, "/Add_115_output_0_scale"},
      {24, "/Mul_123_output_0_scale"}, {25, "/Add_125_output_0_scale"},
      {26, "/Add_132_output_0_scale"}, {27, "/Add_135_output_0_scale"},
      {28, "/Add_140_output_0_scale"}, {29, "/Add_145_output_0_scale"},
      {30, "/Mul_153_output_0_scale"}, {31, "/Mul_158_output_0_scale"},
      {32, "/Mul_163_output_0_scale"}, {33, "/Add_165_output_0_scale"},
      {34, "/Mul_173_output_0_scale"}, {35, "/Add_175_output_0_scale"},
  };

  const std::map<int, std::string> mul_output_scale = {
      {0, ""},
      {1, "/Add_5_output_0_scale"},
      {2, "/Add_12_output_0_scale"},
      {3, "/Add_17_output_0_scale"},
      {4, "/Mul_23_output_0_scale"},
      {5, "/Add_27_output_0_scale"},
      {6, "/Add_30_output_0_scale"},
      {7, "/Add_37_output_0_scale"},
      {8, "/Add_42_output_0_scale"},
      {9, "/Add_45_output_0_scale"},
      {10, "/Add_50_output_0_scale"},
      {11, "/Mul_58_output_0_scale"},
      {12, "/Add_60_output_0_scale"},
      {13, "/Add_67_output_0_scale"},
      {14, "/Mul_73_output_0_scale"},
      {15, "/Add_75_output_0_scale"},
      {16, "/Add_82_output_0_scale"},
      {17, "/Add_85_output_0_scale"},
      {18, "/Add_92_output_0_scale"},
      {19, "/Add_97_output_0_scale"},
      {20, "/Add_100_output_0_scale"},
      {21, "/Add_105_output_0_scale"},
      {22, "/Add_110_output_0_scale"},
      {23, "/Add_115_output_0_scale"},
      {24, "/Mul_123_output_0_scale"},
      {25, "/Add_125_output_0_scale"},
      {26, "/Add_132_output_0_scale"},
      {27, "/Add_135_output_0_scale"},
      {28, "/Mul_143_output_0_scale"},
      {29, "/Add_145_output_0_scale"},
      {30, "/Mul_153_output_0_scale"},
      {31, "/Mul_158_output_0_scale"},
      {32, "/Mul_163_output_0_scale"},
      {33, "/Add_165_output_0_scale"},
      {34, "/Mul_173_output_0_scale"},
      {35, "/Mul_178_output_0_scale"},
  };

  const std::map<int, std::string> mul2_scale = {
      {0, "/Mul_4_output_0_scale"},
      {1, "/ReduceMin_1_output_0_scale"},
      {2, "/ReduceMin_2_output_0_scale"},
      {3, "/Tile_3_output_0_scale"},
      {4, "/Tile_4_output_0_scale"},
      {5, "/ReduceMin_5_output_0_scale"},
      {6, "/Tile_6_output_0_scale"},
      {7, "/ReduceMin_7_output_0_scale"},
      {8, "/ReduceMin_8_output_0_scale"},
      {9, "/Expand_9_output_0_scale"},
      {10, "/Tile_10_output_0_scale"},
      {11, "/Expand_11_output_0_scale"},
      {12, "/Mul_64_output_0_scale"},
      {13, "/ReduceMin_13_output_0_scale"},
      {14, "/Mul_74_output_0_scale"},
      {15, "/Tile_15_output_0_scale"},
      {16, "/Mul_84_output_0_scale"},
      {17, "/Tile_17_output_0_scale"},
      {18, "/Expand_18_output_0_scale"},
      {19, "/ReduceMin_19_output_0_scale"},
      {20, "/Expand_20_output_0_scale"},
      {21, "/Tile_21_output_0_scale"},
      {22, "/Tile_22_output_0_scale"},
      {23, "/Tile_23_output_0_scale"},
      {24, "/Expand_24_output_0_scale"},
      {25, "/ReduceMin_25_output_0_scale"},
      {26, "/Mul_134_output_0_scale"},
      {27, "/Tile_27_output_0_scale"},
      {28, "/Tile_28_output_0_scale"},
      {29, "/Mul_149_output_0_scale"},
      {30, "/ReduceMin_30_output_0_scale"},
      {31, "/ReduceMin_31_output_0_scale"},
      {32, "/Mul_164_output_0_scale"},
      {33, "/Expand_33_output_0_scale"},
      {34, "/Tile_34_output_0_scale"},
      {35, "/ReduceMin_35_output_0_scale"},
  };

  const std::map<int, std::string> add_scale = {
      {0, "/Mul_3_output_0_scale"},    {1, "/Add_5_output_0_scale"},
      {2, "/Add_12_output_0_scale"},   {3, "/Add_17_output_0_scale"},
      {4, "/Mul_23_output_0_scale"},   {5, "/Add_27_output_0_scale"},
      {6, "/Add_30_output_0_scale"},   {7, "/Add_37_output_0_scale"},
      {8, "/Add_42_output_0_scale"},   {9, "/Add_45_output_0_scale"},
      {10, "/Add_50_output_0_scale"},  {11, "/Mul_58_output_0_scale"},
      {12, "/Add_60_output_0_scale"},  {13, "/Add_67_output_0_scale"},
      {14, "/Mul_73_output_0_scale"},  {15, "/Add_75_output_0_scale"},
      {16, "/Add_82_output_0_scale"},  {17, "/Add_85_output_0_scale"},
      {18, "/Add_92_output_0_scale"},  {19, "/Add_97_output_0_scale"},
      {20, "/Add_100_output_0_scale"}, {21, "/Add_105_output_0_scale"},
      {22, "/Add_110_output_0_scale"}, {23, "/Add_115_output_0_scale"},
      {24, "/Mul_123_output_0_scale"}, {25, "/Add_125_output_0_scale"},
      {26, "/Add_132_output_0_scale"}, {27, "/Add_135_output_0_scale"},
      {28, "/Mul_143_output_0_scale"}, {29, "/Add_145_output_0_scale"},
      {30, "/Mul_153_output_0_scale"}, {31, "/Mul_158_output_0_scale"},
      {32, "/Mul_163_output_0_scale"}, {33, "/Add_165_output_0_scale"},
      {34, "/Mul_173_output_0_scale"}, {35, "/Mul_178_output_0_scale"},
  };

  const std::vector<std::string> k_unsqueeze_scale_ = {
      "/linear_k/Add_output_0_scale",    "/Unsqueeze_411_output_0_scale",
      "/Slice_10_output_0_scale",        "/Concat_138_output_0_scale",
      "/Unsqueeze_486_output_0_scale",   "/Unsqueeze_511_output_0_scale",
      "/Unsqueeze_536_output_0_scale",   "/Unsqueeze_561_output_0_scale",
      "/linear_k_8/Add_output_0_scale",  "/Unsqueeze_611_output_0_scale",
      "/Unsqueeze_636_output_0_scale",   "/Concat_202_output_0_scale",
      "/Unsqueeze_686_output_0_scale",   "/Unsqueeze_711_output_0_scale",
      "/Unsqueeze_736_output_0_scale",   "/Concat_234_output_0_scale",
      "/Concat_242_output_0_scale",      "/linear_k_17/Add_output_0_scale",
      "/Unsqueeze_836_output_0_scale",   "/Unsqueeze_861_output_0_scale",
      "/linear_k_20/Add_output_0_scale", "/Unsqueeze_911_output_0_scale",
      "/Concat_290_output_0_scale",      "/Concat_298_output_0_scale",
      "/Unsqueeze_986_output_0_scale",   "/Concat_314_output_0_scale",
      "/Unsqueeze_1036_output_0_scale",  "/Unsqueeze_1061_output_0_scale",
      "/linear_k_28/Add_output_0_scale", "/Concat_346_output_0_scale",
      "/linear_k_30/Add_output_0_scale", "/Unsqueeze_1161_output_0_scale",
      "/Unsqueeze_1186_output_0_scale",  "/Unsqueeze_1211_output_0_scale",
      "/Concat_386_output_0_scale",      "/Unsqueeze_1261_output_0_scale"};

  const std::vector<std::string> k_unsqueeze_zp_ = {
      "/Unsqueeze_26_output_0_zero_point",
      "/Unsqueeze_411_output_0_zero_point",
      "/Unsqueeze_436_output_0_zero_point",
      "/Unsqueeze_461_output_0_zero_point",
      "/Unsqueeze_486_output_0_zero_point",
      "/Unsqueeze_511_output_0_zero_point",
      "/Unsqueeze_536_output_0_zero_point",
      "/Unsqueeze_561_output_0_zero_point",
      "/Unsqueeze_586_output_0_zero_point",
      "/Unsqueeze_611_output_0_zero_point",
      "/Unsqueeze_636_output_0_zero_point",
      "/Unsqueeze_661_output_0_zero_point",
      "/Unsqueeze_686_output_0_zero_point",
      "/Unsqueeze_711_output_0_zero_point",
      "/Unsqueeze_736_output_0_zero_point",
      "/Unsqueeze_761_output_0_zero_point",
      "/Unsqueeze_786_output_0_zero_point",
      "/Unsqueeze_811_output_0_zero_point",
      "/Unsqueeze_836_output_0_zero_point",
      "/Unsqueeze_861_output_0_zero_point",
      "/Unsqueeze_886_output_0_zero_point",
      "/Unsqueeze_911_output_0_zero_point",
      "/Unsqueeze_936_output_0_zero_point",
      "/Unsqueeze_961_output_0_zero_point",
      "/Unsqueeze_986_output_0_zero_point",
      "/Unsqueeze_1011_output_0_zero_point",
      "/Unsqueeze_1036_output_0_zero_point",
      "/Unsqueeze_1061_output_0_zero_point",
      "/Unsqueeze_1086_output_0_zero_point",
      "/Unsqueeze_1111_output_0_zero_point",
      "/Unsqueeze_1136_output_0_zero_point",
      "/Unsqueeze_1161_output_0_zero_point",
      "/Unsqueeze_1186_output_0_zero_point",
      "/Unsqueeze_1211_output_0_zero_point",
      "/Unsqueeze_1236_output_0_zero_point",
      "/Unsqueeze_1261_output_0_zero_point"};

  const std::vector<std::string> k_concat_slice_scale_ = {
      "/linear_k/Add_output_0_scale",    "/Unsqueeze_411_output_0_scale",
      "/Slice_10_output_0_scale",        "/Concat_138_output_0_scale",
      "/linear_k_4/Add_output_0_scale",  "/linear_k_5/Add_output_0_scale",
      "/Slice_26_output_0_scale",        "/Concat_170_output_0_scale",
      "/linear_k_8/Add_output_0_scale",  "/Unsqueeze_611_output_0_scale",
      "/Unsqueeze_636_output_0_scale",   "/Concat_202_output_0_scale",
      "/linear_k_12/Add_output_0_scale", "/Concat_218_output_0_scale",
      "/Slice_58_output_0_scale",        "/Concat_234_output_0_scale",
      "/Concat_242_output_0_scale",      "/Concat_250_output_0_scale",
      "/linear_k_18/Add_output_0_scale", "/linear_k_19/Add_output_0_scale",
      "/linear_k_20/Add_output_0_scale", "/linear_k_21/Add_output_0_scale",
      "/Concat_290_output_0_scale",      "/Concat_298_output_0_scale",
      "/Unsqueeze_986_output_0_scale",   "/Concat_314_output_0_scale",
      "/linear_k_26/Add_output_0_scale", "/Slice_110_output_0_scale",
      "/linear_k_28/Add_output_0_scale", "/Concat_346_output_0_scale",
      "/Concat_354_output_0_scale",      "/Concat_362_output_0_scale",
      "/Concat_370_output_0_scale",      "/Concat_378_output_0_scale",
      "/Concat_386_output_0_scale",      "/linear_k_35/Add_output_0_scale"};

  const std::vector<std::string> v_unsqueeze_scale_ = {
      "/Unsqueeze_25_output_0_scale",    "/Unsqueeze_410_output_0_scale",
      "/Unsqueeze_435_output_0_scale",   "/Slice_15_output_0_scale",
      "/Unsqueeze_485_output_0_scale",   "/Unsqueeze_510_output_0_scale",
      "/Unsqueeze_535_output_0_scale",   "/Slice_31_output_0_scale",
      "/linear_v_8/Add_output_0_scale",  "/Slice_39_output_0_scale",
      "/linear_v_10/Add_output_0_scale", "/Unsqueeze_660_output_0_scale",
      "/Unsqueeze_685_output_0_scale",   "/Unsqueeze_710_output_0_scale",
      "/Slice_59_output_0_scale",        "/Unsqueeze_760_output_0_scale",
      "/Concat_243_output_0_scale",      "/Concat_251_output_0_scale",
      "/Concat_259_output_0_scale",      "/Concat_267_output_0_scale",
      "/Slice_83_output_0_scale",        "/Unsqueeze_910_output_0_scale",
      "/linear_v_22/Add_output_0_scale", "/Slice_95_output_0_scale",
      "/Unsqueeze_985_output_0_scale",   "/Unsqueeze_1010_output_0_scale",
      "/Unsqueeze_1035_output_0_scale",  "/Unsqueeze_1060_output_0_scale",
      "/Slice_115_output_0_scale",       "/Unsqueeze_1110_output_0_scale",
      "/Unsqueeze_1135_output_0_scale",  "/Concat_363_output_0_scale",
      "/Concat_371_output_0_scale",      "/Unsqueeze_1210_output_0_scale",
      "/Slice_139_output_0_scale",       "/Unsqueeze_1260_output_0_scale"};

  const std::vector<std::string> v_unsqueeze_zp_ = {
      "/Unsqueeze_25_output_0_zero_point",
      "/Unsqueeze_410_output_0_zero_point",
      "/Unsqueeze_435_output_0_zero_point",
      "/Unsqueeze_460_output_0_zero_point",
      "/Unsqueeze_485_output_0_zero_point",
      "/Unsqueeze_510_output_0_zero_point",
      "/Unsqueeze_535_output_0_zero_point",
      "/Unsqueeze_560_output_0_zero_point",
      "/Unsqueeze_585_output_0_zero_point",
      "/Unsqueeze_610_output_0_zero_point",
      "/Unsqueeze_635_output_0_zero_point",
      "/Unsqueeze_660_output_0_zero_point",
      "/Unsqueeze_685_output_0_zero_point",
      "/Unsqueeze_710_output_0_zero_point",
      "/Unsqueeze_735_output_0_zero_point",
      "/Unsqueeze_760_output_0_zero_point",
      "/Unsqueeze_785_output_0_zero_point",
      "/Unsqueeze_810_output_0_zero_point",
      "/Unsqueeze_835_output_0_zero_point",
      "/Unsqueeze_860_output_0_zero_point",
      "/Unsqueeze_885_output_0_zero_point",
      "/Unsqueeze_910_output_0_zero_point",
      "/Unsqueeze_935_output_0_zero_point",
      "/Unsqueeze_960_output_0_zero_point",
      "/Unsqueeze_985_output_0_zero_point",
      "/Unsqueeze_1010_output_0_zero_point",
      "/Unsqueeze_1035_output_0_zero_point",
      "/Unsqueeze_1060_output_0_zero_point",
      "/Unsqueeze_1085_output_0_zero_point",
      "/Unsqueeze_1110_output_0_zero_point",
      "/Unsqueeze_1135_output_0_zero_point",
      "/Unsqueeze_1160_output_0_zero_point",
      "/Unsqueeze_1185_output_0_zero_point",
      "/Unsqueeze_1210_output_0_zero_point",
      "/Unsqueeze_1235_output_0_zero_point",
      "/Unsqueeze_1260_output_0_zero_point"};

  const std::vector<std::string> k_concat_slice_zp_ = {
      "/Slice_2_output_0_zero_point",   "/Slice_6_output_0_zero_point",
      "/Slice_10_output_0_zero_point",  "/Slice_14_output_0_zero_point",
      "/Slice_18_output_0_zero_point",  "/Slice_22_output_0_zero_point",
      "/Slice_26_output_0_zero_point",  "/Slice_30_output_0_zero_point",
      "/Slice_34_output_0_zero_point",  "/Slice_38_output_0_zero_point",
      "/Slice_42_output_0_zero_point",  "/Slice_46_output_0_zero_point",
      "/Slice_50_output_0_zero_point",  "/Slice_54_output_0_zero_point",
      "/Slice_58_output_0_zero_point",  "/Slice_62_output_0_zero_point",
      "/Slice_66_output_0_zero_point",  "/Slice_70_output_0_zero_point",
      "/Slice_74_output_0_zero_point",  "/Slice_78_output_0_zero_point",
      "/Slice_82_output_0_zero_point",  "/Slice_86_output_0_zero_point",
      "/Slice_90_output_0_zero_point",  "/Slice_94_output_0_zero_point",
      "/Slice_98_output_0_zero_point",  "/Slice_102_output_0_zero_point",
      "/Slice_106_output_0_zero_point", "/Slice_110_output_0_zero_point",
      "/Slice_114_output_0_zero_point", "/Slice_118_output_0_zero_point",
      "/Slice_122_output_0_zero_point", "/Slice_126_output_0_zero_point",
      "/Slice_130_output_0_zero_point", "/Slice_134_output_0_zero_point",
      "/Slice_138_output_0_zero_point", "/Slice_142_output_0_zero_point"};

  const std::vector<std::string> v_concat_slice_scale_ = {
      "/linear_v/Add_output_0_scale",    "/Concat_123_output_0_scale",
      "/Concat_131_output_0_scale",      "/Slice_15_output_0_scale",
      "/Concat_147_output_0_scale",      "/Slice_23_output_0_scale",
      "/Unsqueeze_535_output_0_scale",   "/Slice_31_output_0_scale",
      "/Slice_35_output_0_scale",        "/Slice_39_output_0_scale",
      "/linear_v_10/Add_output_0_scale", "/linear_v_11/Add_output_0_scale",
      "/Concat_211_output_0_scale",      "/Concat_219_output_0_scale",
      "/Slice_59_output_0_scale",        "/linear_v_15/Add_output_0_scale",
      "/Concat_243_output_0_scale",      "/Concat_251_output_0_scale",
      "/Concat_259_output_0_scale",      "/Concat_267_output_0_scale",
      "/Slice_83_output_0_scale",        "/Slice_87_output_0_scale",
      "/linear_v_22/Add_output_0_scale", "/Slice_95_output_0_scale",
      "/linear_v_24/Add_output_0_scale", "/Concat_315_output_0_scale",
      "/Concat_323_output_0_scale",      "/Unsqueeze_1060_output_0_scale",
      "/Slice_115_output_0_scale",       "/Concat_347_output_0_scale",
      "/linear_v_30/Add_output_0_scale", "/Concat_363_output_0_scale",
      "/Concat_371_output_0_scale",      "/Concat_379_output_0_scale",
      "/Slice_139_output_0_scale",       "/Concat_395_output_0_scale"};
  const std::vector<std::string> v_concat_slice_zp_ = {
      "/Slice_3_output_0_zero_point",   "/Slice_7_output_0_zero_point",
      "/Slice_11_output_0_zero_point",  "/Slice_15_output_0_zero_point",
      "/Slice_19_output_0_zero_point",  "/Slice_23_output_0_zero_point",
      "/Slice_27_output_0_zero_point",  "/Slice_31_output_0_zero_point",
      "/Slice_35_output_0_zero_point",  "/Slice_39_output_0_zero_point",
      "/Slice_43_output_0_zero_point",  "/Slice_47_output_0_zero_point",
      "/Slice_51_output_0_zero_point",  "/Slice_55_output_0_zero_point",
      "/Slice_59_output_0_zero_point",  "/Slice_63_output_0_zero_point",
      "/Slice_67_output_0_zero_point",  "/Slice_71_output_0_zero_point",
      "/Slice_75_output_0_zero_point",  "/Slice_79_output_0_zero_point",
      "/Slice_83_output_0_zero_point",  "/Slice_87_output_0_zero_point",
      "/Slice_91_output_0_zero_point",  "/Slice_95_output_0_zero_point",
      "/Slice_99_output_0_zero_point",  "/Slice_103_output_0_zero_point",
      "/Slice_107_output_0_zero_point", "/Slice_111_output_0_zero_point",
      "/Slice_115_output_0_zero_point", "/Slice_119_output_0_zero_point",
      "/Slice_123_output_0_zero_point", "/Slice_127_output_0_zero_point",
      "/Slice_131_output_0_zero_point", "/Slice_135_output_0_zero_point",
      "/Slice_139_output_0_zero_point", "/Slice_143_output_0_zero_point"};

  const std::map<int, std::string> softmax_ifm_prefix_ = {
      {0, ""},
      {1, "/Add_5_output_0_scale"},
      {2, "/Add_12_output_0_scale"},
      {3, "/Add_17_output_0_scale"},
      {4, "/Mul_23_output_0_scale"},
      {5, "/Add_27_output_0_scale"},
      {6, "/Add_30_output_0_scale"},
      {7, "/Add_37_output_0_scale"},
      {8, "/Add_42_output_0_scale"},
      {9, "/Add_45_output_0_scale"},
      {10, "/Add_50_output_0_scale"},
      {11, "/Mul_58_output_0_scale"},
      {12, "/Add_60_output_0_scale"},
      {13, "/Add_67_output_0_scale"},
      {14, "/Mul_73_output_0_scale"},
      {15, "/Add_75_output_0_scale"},
      {16, "/Add_82_output_0_scale"},
      {17, "/Add_85_output_0_scale"},
      {18, "/Add_92_output_0_scale"},
      {19, "/Add_97_output_0_scale"},
      {20, "/Add_100_output_0_scale"},
      {21, "/Add_105_output_0_scale"},
      {22, "/Add_110_output_0_scale"},
      {23, "/Add_115_output_0_scale"},
      {24, "/Mul_123_output_0_scale"},
      {25, "/Add_125_output_0_scale"},
      {26, "/Add_132_output_0_scale"},
      {27, "/Add_135_output_0_scale"},
      {28, "/Mul_143_output_0_scale"},
      {29, "/Add_145_output_0_scale"},
      {30, "/Mul_153_output_0_scale"},
      {31, "/Mul_158_output_0_scale"},
      {32, "/Mul_163_output_0_scale"},
      {33, "/Add_167_output_0_scale"},
      {34, "/Mul_173_output_0_scale"},
      {35, "/Mul_178_output_0_scale"},
  };
};
} // namespace vaip_vaiml_custom_op