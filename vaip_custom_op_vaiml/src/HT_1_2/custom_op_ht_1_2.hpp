/*
 *     The Xilinx Vitis AI Vaip in this distribution are provided under the
 * following free and permissive binary-only license, but are not provided in
 * source code form.  While the following free and permissive license is similar
 * to the BSD open source license, it is NOT the BSD open source license nor
 * other OSI-approved open source license.
 *
 *      Copyright (C) 2022 – 2023 Advanced Micro Devices, Inc. All rights
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

/*

**/
#pragma once

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#endif

#define WTS_SIZE_HT 17868672
#define IFM_SIZE_HT 15104
#define OFM_SIZE_HT 9344
#define TMP_SIZE_HT 11904
#define TRANSFORMER_BLOCK_NUM 36
// #include "gt_txn_pkg.hpp"
#include "../common/hw_runner.h"
#include "../common/utils.h"
#include "../common/vaiml_client.h"
#include "load_wts.h"
#include "onnxruntime_api.hpp"
#include "txn_pkg_ht_1_2.hpp"
#include "vaip/vaip.hpp"
#include "vitis/ai/env_config.hpp"
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
using VaimlTensorShape = std::vector<int64_t>;
using VaimlShapeVec = std::vector<VaimlTensorShape>;

namespace vaip_vaiml_custom_op {
using namespace vaip_core;

class MyCustomOpHT1_2 : public CustomOpImp {
public:
  MyCustomOpHT1_2(std::shared_ptr<const PassContext> context,
                  const std::shared_ptr<MetaDefProto>& meta_def,
                  onnxruntime::Model* model);

  virtual ~MyCustomOpHT1_2();

private:
  void real_compute() const;
  void MyCustomOpGraphMode(std::shared_ptr<const PassContext> context,
                           const onnxruntime::Graph* graph,
                           const std::shared_ptr<MetaDefProto>& meta_def);
  void MyCustomOpTransformerMode(std::shared_ptr<const PassContext> context,
                                 const onnxruntime::Graph* graph,
                                 const std::shared_ptr<MetaDefProto>& meta_def);

  void forwardUkernel(std::map<int, int>& datatype_to_size,
                      std::map<int, std::string>& datatype_to_string,
                      const VaimlShapeVec& output_shapes,
                      Ort::KernelContext& ctx) const;

  virtual void Compute(const OrtApi* api,
                       OrtKernelContext* context) const override final;
  int32_t GetInputDataAndSet(Ort::KernelContext& ctx, int index,
                             int8_t* ifm_ptr) const;
  int32_t GetOnputDataAndSet(Ort::KernelContext& ctx, int index,
                             int8_t* ofm_ptr) const;

  bool InitHtWeight(
      std::unordered_map<std::string, flexmlrt::client::ErtIoTypeNew>& wts_,
      int8_t* wts, const vaip_core::PassContext& context);

  int32_t SliceCompute_HT(Ort::KernelContext& ctx) const;
  int32_t ConcatCompute_HT(Ort::KernelContext& ctx) const;

  SUBGRAPH_ID IdentifySubgraph(const std::shared_ptr<MetaDefProto>& meta_def);

  SUBGRAPH_ID subgraph_id_ = SUBGRAPH_ID::UNKNOWN;

  std::string sg_name_;
  static std::map<std::string, std::vector<char>> node_cache;
  static size_t gt_qkv_compute_iter;
  bool debug_ = false;
  std::string vaiml_model_path_ = "vaiml_par_0";
  std::string device_name_ = "phx";
  std::string config_filename_ = "";
  std::shared_ptr<flexmlrt::client::Model> runner_;
  VaimlShapeVec ort_output_shapes_;
  hw_runner g;
  int8_t* wts_ptr_front_;
  int8_t* ifm_ptr_front_;
  int8_t* ofm_ptr_front_;
  int8_t* wts_ptr_;
  int8_t* ifm_ptr_;
  int8_t* ofm_ptr_;
  std::string constants_file_name_ = "wts.bin";
  std::unordered_map<std::string, flexmlrt::client::ErtIoTypeNew> wts_;
  std::vector<std::vector<char>> wts_buffers_;
  std::map<int, int> datatype_to_size;
  std::map<int, std::string> datatype_to_string;
  std::string model_version_;
  std::vector<std::string> ht_lstm_wts_name_ = {
      // scale: float[24]
      "c0_scale", "/decoder/rnn/Slice_13_output_0_scale",
      "/decoder/rnn/Slice_27_output_0_scale", "h0_scale",
      "/decoder/rnn/Slice_12_output_0_scale",
      "/decoder/rnn/Slice_26_output_0_scale",
      "decoder_embedding.embed.weight_scale",
      "decoder_embedding.embed_lnorm.weight_scale",
      "/decoder_embedding/embed_lnorm/Add_1_output_0_scale",
      "/decoder_embedding/sigmoid/Sigmoid_output_0_scale",
      "/decoder/rnn/Unsqueeze_output_0_scale",
      "/decoder/rnn/Unsqueeze_1_output_0_scale",
      "/decoder/rnn/Unsqueeze_2_output_0_scale",
      "/decoder/rnn/LSTM_output_0_scale", "/decoder/rnn/LSTM_output_1_scale",
      "/decoder/rnn/LSTM_output_2_scale",
      "/decoder/rnn/Unsqueeze_3_output_0_scale",
      "/decoder/rnn/Unsqueeze_4_output_0_scale",
      "/decoder/rnn/Unsqueeze_5_output_0_scale",
      "/decoder/rnn/LSTM_1_output_0_scale",
      "/decoder/rnn/LSTM_1_output_1_scale",
      "/decoder/rnn/LSTM_1_output_2_scale", "h1_scale", "c1_scale",
      // zp: int8[24]
      "c0_zero_point", "/decoder/rnn/Slice_13_output_0_zero_point",
      "/decoder/rnn/Slice_27_output_0_zero_point", "h0_zero_point",
      "/decoder/rnn/Slice_12_output_0_zero_point",
      "/decoder/rnn/Slice_26_output_0_zero_point",
      "decoder_embedding.embed.weight_zero_point",
      "decoder_embedding.embed_lnorm.weight_zero_point",
      "/decoder_embedding/embed_lnorm/Add_1_output_0_zero_point",
      "/decoder_embedding/sigmoid/Sigmoid_output_0_zero_point",
      "/decoder/rnn/Unsqueeze_output_0_zero_point",
      "/decoder/rnn/Unsqueeze_1_output_0_zero_point",
      "/decoder/rnn/Unsqueeze_2_output_0_zero_point",
      "/decoder/rnn/LSTM_output_0_zero_point",
      "/decoder/rnn/LSTM_output_1_zero_point",
      "/decoder/rnn/LSTM_output_2_zero_point",
      "/decoder/rnn/Unsqueeze_3_output_0_zero_point",
      "/decoder/rnn/Unsqueeze_4_output_0_zero_point",
      "/decoder/rnn/Unsqueeze_5_output_0_zero_point",
      "/decoder/rnn/LSTM_1_output_0_zero_point",
      "/decoder/rnn/LSTM_1_output_1_zero_point",
      "/decoder/rnn/LSTM_1_output_2_zero_point", "h1_zero_point",
      "c1_zero_point",
      // lstm0_h_wts
      "/decoder/rnn/Unsqueeze_1_output_0_quantized",
      // lstm0_x_wts
      "/decoder/rnn/Unsqueeze_output_0_quantized",
      // lstm0_bias
      "/decoder/rnn/Unsqueeze_2_output_0_quantized",
      // lstm1_h_wts
      "/decoder/rnn/Unsqueeze_4_output_0_quantized",
      // lstm1_x_wts
      "/decoder/rnn/Unsqueeze_3_output_0_quantized",
      // lstm1_bias
      "/decoder/rnn/Unsqueeze_5_output_0_quantized"};
  const std::vector<uint16_t> lstm_lut = {
      1025,  14651, 57420, 14675, 2318,  14704, 63217, 14727, 1667,  14746,
      31387, 14766, 41960, 14789, 56724, 14815, 36512, 14845, 38315, 14863,
      39806, 14882, 8914,  14904, 32539, 14928, 3682,  14956, 40254, 14981,
      15674, 14999, 10753, 15019, 44699, 15041, 7964,  15067, 55702, 15095,
      9100,  15116, 28254, 15134, 3672,  15155, 19368, 15178, 30227, 15204,
      62135, 15232, 32002, 15249, 3691,  15268, 57305, 15288, 11910, 15312,
      15289, 15338, 42534, 15363, 54253, 15379, 52171, 15397, 45647, 15417,
      43734, 15439, 54795, 15463, 10235, 15489, 38595, 15503, 15553, 15519,
      6261,  15536, 8009,  15554, 15131, 15573, 18290, 15593, 3718,  15614,
      41755, 15625, 18066, 15636, 46122, 15646, 41162, 15656, 45992, 15665,
      34584, 15673, 43680, 15679, 42613, 15683, 161,   15685, 17357, 15683,
      1907,  15678, 63769, 15668, 58825, 15655, 49868, 15638, 46218, 15617,
      9408,  15570, 52480, 15514, 42261, 15421, 32822, 15231, 32822, 47999,
      42261, 48189, 52480, 48282, 9408,  48338, 46218, 48385, 49868, 48406,
      58825, 48423, 63769, 48436, 1907,  48446, 17357, 48451, 161,   48453,
      42613, 48451, 43680, 48447, 34584, 48441, 45992, 48433, 41162, 48424,
      46122, 48414, 18066, 48404, 41755, 48393, 3718,  48382, 18290, 48361,
      15131, 48341, 8009,  48322, 6261,  48304, 15553, 48287, 38595, 48271,
      10235, 48257, 54795, 48231, 43734, 48207, 45647, 48185, 52171, 48165,
      54253, 48147, 42534, 48131, 15289, 48106, 11910, 48080, 57305, 48056,
      3691,  48036, 32002, 48017, 62135, 48000, 30227, 47972, 19368, 47946,
      3672,  47923, 28254, 47902, 9100,  47884, 55702, 47863, 7964,  47835,
      44699, 47809, 10753, 47787, 15674, 47767, 40254, 47749, 3682,  47724,
      32539, 47696, 8914,  47672, 39806, 47650, 38315, 47631, 36512, 47613,
      56724, 47583, 41960, 47557, 31387, 47534, 1667,  47514, 63217, 47495,
      2318,  47472, 57420, 47443, 1025,  47419, 62643, 15184, 27846, 15209,
      22961, 15234, 35903, 15249, 31304, 15266, 21676, 15285, 20833, 15306,
      44014, 15329, 42476, 15355, 17352, 15372, 20507, 15388, 9089,  15406,
      60861, 15425, 58128, 15447, 15503, 15472, 39994, 15493, 36147, 15508,
      5604,  15525, 24058, 15543, 36921, 15563, 55944, 15585, 28152, 15610,
      48986, 15626, 41457, 15641, 64430, 15657, 60076, 15675, 36335, 15695,
      1279,  15717, 28524, 15740, 62997, 15754, 52320, 15768, 51234, 15783,
      62733, 15799, 23615, 15817, 858,   15836, 60290, 15855, 34840, 15874,
      45809, 15885, 27736, 15897, 42347, 15909, 18840, 15922, 15855, 15935,
      24717, 15948, 34874, 15961, 33881, 15974, 7495,  15987, 5442,  15999,
      5382,  16005, 2988,  16010, 19721, 16014, 47685, 16017, 14761, 16020,
      47436, 16021, 12730, 16022, 43103, 16021, 12604, 16020, 61343, 16017,
      5487,  16015, 57571, 16011, 39053, 16008, 34292, 16005, 61234, 16002,
      4178,  16001, 5446,  16000, 5446,  16000, 4178,  16001, 61234, 16002,
      34292, 16005, 39053, 16008, 57571, 16011, 5487,  16015, 61343, 16017,
      12604, 16020, 43103, 16021, 12730, 16022, 47436, 16021, 14761, 16020,
      47685, 16017, 19721, 16014, 2988,  16010, 5382,  16005, 5442,  15999,
      7495,  15987, 33881, 15974, 34874, 15961, 24717, 15948, 15855, 15935,
      18840, 15922, 42347, 15909, 27736, 15897, 45809, 15885, 34840, 15874,
      60290, 15855, 858,   15836, 23615, 15817, 62733, 15799, 51234, 15783,
      52320, 15768, 62997, 15754, 28524, 15740, 1279,  15717, 36335, 15695,
      60076, 15675, 64430, 15657, 41457, 15641, 48986, 15626, 28152, 15610,
      55944, 15585, 36921, 15563, 24058, 15543, 5604,  15525, 36147, 15508,
      39994, 15493, 15503, 15472, 58128, 15447, 60861, 15425, 9089,  15406,
      20507, 15388, 17352, 15372, 42476, 15355, 44014, 15329, 20833, 15306,
      21676, 15285, 31304, 15266, 35903, 15249, 22961, 15234, 27846, 15209,
      62643, 15184, 25588, 15468, 15538, 15490, 29305, 15503, 60975, 15517,
      52433, 15533, 11663, 15551, 12780, 15570, 64984, 15590, 47048, 15613,
      17510, 15627, 52813, 15640, 37072, 15655, 42116, 15671, 9045,  15689,
      10369, 15708, 53384, 15728, 40074, 15747, 49258, 15759, 58284, 15772,
      5731,  15787, 26829, 15802, 60191, 15818, 44364, 15836, 48825, 15855,
      5879,  15874, 1044,  15885, 44145, 15896, 5288,  15909, 16347, 15922,
      12125, 15936, 57929, 15950, 21785, 15966, 33064, 15982, 23609, 15999,
      27641, 16008, 28847, 16017, 45117, 16026, 7141,  16036, 41483, 16045,
      11816, 16055, 43223, 16064, 63501, 16073, 65408, 16082, 41282, 16091,
      48805, 16099, 14686, 16107, 62695, 16113, 55329, 16119, 53021, 16124,
      26224, 16128, 59020, 16129, 60081, 16130, 31489, 16131, 42213, 16131,
      31357, 16131, 4479,  16131, 33174, 16130, 57934, 16129, 18644, 16129,
      50642, 16128, 25525, 16128, 9651,  16128, 2004,  16128, 0,     16128,
      0,     16128, 61527, 16127, 46234, 16127, 14486, 16127, 29787, 16126,
      28248, 16125, 15205, 16124, 64724, 16122, 56577, 16121, 2823,  16121,
      46646, 16120, 2559,  16121, 10910, 16122, 13031, 16124, 13087, 16127,
      39026, 16129, 5103,  16132, 1421,  16135, 25425, 16138, 8365,  16142,
      12127, 16146, 32832, 16150, 1017,  16155, 43925, 16159, 26860, 16164,
      12026, 16169, 61966, 16173, 42977, 16178, 18344, 16183, 51715, 16187,
      10482, 16192, 24502, 16196, 27322, 16200, 18286, 16204, 62505, 16207,
      28681, 16211, 47830, 16214, 54500, 16217, 48891, 16220, 31298, 16223,
      2089,  16226, 27223, 16228, 41628, 16230, 45798, 16232, 40244, 16234,
      25482, 16236, 2035,  16238, 35951, 16239, 62200, 16240, 15736, 16242,
      28107, 16243, 34232, 16244, 34547, 16245, 29467, 16246, 19386, 16247,
      4674,  16248, 51217, 16248, 28273, 16249, 1684,  16250, 37273, 16250,
      4239,  16251, 33900, 16251, 60954, 16251, 20080, 16252, 6610,  13577,
      2684,  13616, 2688,  13666, 7941,  13713, 22231, 13754, 17314, 13807,
      40015, 13849, 15713, 13893, 17082, 13949, 39084, 13986, 50960, 14032,
      2466,  14086, 7034,  14124, 64851, 14172, 57512, 14221, 11372, 14262,
      59899, 14313, 11440, 14358, 54132, 14400, 38681, 14455, 62482, 14494,
      6278,  14540, 1865,  14595, 15590, 14632, 777,   14680, 44186, 14730,
      3133,  14770, 39046, 14820, 48705, 14866, 25685, 14908, 55738, 14961,
      15091, 15003, 16537, 15047, 48116, 15103, 6111,  15140, 36006, 15186,
      3371,  15239, 13458, 15277, 4197,  15326, 19102, 15374, 16461, 15414,
      17556, 15465, 9664,  15509, 32964, 15550, 63113, 15602, 41866, 15642,
      23901, 15684, 37270, 15736, 45589, 15772, 35938, 15812, 60757, 15860,
      22083, 15895, 2487,  15929, 13236, 15967, 21259, 16004, 33515, 16025,
      15070, 16045, 46986, 16060, 33130, 16068, 57468, 16064, 45211, 16046,
      30784, 16012, 53136, 15926, 865,   15742, 865,   48510, 53136, 48694,
      30784, 48780, 45211, 48814, 57468, 48832, 33130, 48836, 46986, 48828,
      15070, 48813, 33515, 48793, 21259, 48772, 13236, 48735, 2487,  48697,
      22083, 48663, 60757, 48628, 35938, 48580, 45589, 48540, 37270, 48504,
      23901, 48452, 41866, 48410, 63113, 48370, 32964, 48318, 9664,  48277,
      17556, 48233, 16461, 48182, 19102, 48142, 4197,  48094, 13458, 48045,
      3371,  48007, 36006, 47954, 6111,  47908, 48116, 47871, 16537, 47815,
      15091, 47771, 55738, 47729, 25685, 47676, 48705, 47634, 39046, 47588,
      3133,  47538, 44186, 47498, 777,   47448, 15590, 47400, 1865,  47363,
      6278,  47308, 62482, 47262, 38681, 47223, 54132, 47168, 11440, 47126,
      59899, 47081, 11372, 47030, 57512, 46989, 64851, 46940, 7034,  46892,
      2466,  46854, 50960, 46800, 39084, 46754, 17082, 46717, 15713, 46661,
      40015, 46617, 17314, 46575, 22231, 46522, 7941,  46481, 2688,  46434,
      2684,  46384, 6610,  46345, 39958, 14096, 61072, 14134, 23413, 14183,
      17481, 14226, 58891, 14264, 44297, 14313, 40807, 14355, 30754, 14394,
      31057, 14443, 41714, 14484, 38709, 14523, 44700, 14572, 17261, 14613,
      13401, 14652, 14732, 14701, 29764, 14741, 16181, 14780, 1245,  14829,
      10146, 14869, 42439, 14907, 63771, 14955, 20026, 14996, 21525, 15034,
      64545, 15081, 55019, 15122, 13209, 15160, 61506, 15206, 44564, 15248,
      10822, 15285, 45755, 15330, 48230, 15373, 6311,  15409, 6354,  15453,
      58533, 15497, 54892, 15531, 60027, 15573, 65418, 15620, 11281, 15653,
      55462, 15692, 43198, 15741, 51135, 15772, 27805, 15809, 6534,  15854,
      11148, 15890, 61175, 15922, 18199, 15962, 38741, 16004, 19794, 16032,
      44988, 16064, 1123,  16102, 10266, 16136, 39154, 16159, 56005, 16184,
      5012,  16211, 63669, 16236, 26530, 16258, 17997, 16268, 4789,  16275,
      2044,  16278, 62179, 16276, 30369, 16272, 7187,  16266, 1867,  16260,
      21608, 16256, 21608, 16256, 1867,  16260, 7187,  16266, 30369, 16272,
      62179, 16276, 2044,  16278, 4789,  16275, 17997, 16268, 26530, 16258,
      63669, 16236, 5012,  16211, 56005, 16184, 39154, 16159, 10266, 16136,
      1123,  16102, 44988, 16064, 19794, 16032, 38741, 16004, 18199, 15962,
      61175, 15922, 11148, 15890, 6534,  15854, 27805, 15809, 51135, 15772,
      43198, 15741, 55462, 15692, 11281, 15653, 65418, 15620, 60027, 15573,
      54892, 15531, 58533, 15497, 6354,  15453, 6311,  15409, 48230, 15373,
      45755, 15330, 10822, 15285, 44564, 15248, 61506, 15206, 13209, 15160,
      55019, 15122, 64545, 15081, 21525, 15034, 20026, 14996, 63771, 14955,
      42439, 14907, 10146, 14869, 1245,  14829, 16181, 14780, 29764, 14741,
      14732, 14701, 13401, 14652, 17261, 14613, 44700, 14572, 38709, 14523,
      41714, 14484, 31057, 14443, 30754, 14394, 40807, 14355, 44297, 14313,
      58891, 14264, 17481, 14226, 23413, 14183, 61072, 14134, 39958, 14096,
      64924, 49023, 64773, 49023, 64585, 49023, 64352, 49023, 64062, 49023,
      63703, 49023, 63257, 49023, 62703, 49023, 62017, 49023, 61168, 49023,
      60116, 49023, 58816, 49023, 57208, 49023, 55223, 49023, 52773, 49023,
      49751, 49023, 46028, 49023, 41445, 49023, 35808, 49023, 28882, 49023,
      20380, 49023, 9954,  49023, 62719, 49022, 47094, 49022, 27999, 49022,
      4694,  49022, 41826, 49021, 7255,  49021, 30778, 49020, 45340, 49019,
      49135, 49018, 40025, 49017, 15491, 49016, 38114, 49014, 38907, 49012,
      13868, 49010, 23923, 49007, 63808, 49003, 62037, 48999, 11971, 48995,
      37388, 48989, 64905, 48982, 20779, 48975, 27824, 48966, 12686, 48956,
      34221, 48944, 22326, 48931, 41366, 48916, 29649, 48900, 61383, 48869,
      62804, 48832, 58828, 48794, 5229,  48746, 861,   48674, 39140, 48578,
      41085, 48430, 21767, 48004, 59943, 15517, 52882, 15589, 12414, 15569,
      26582, 15498, 42246, 15355, 54315, 15082, 0,     32768, 0,     32768,
      54315, 47850, 42246, 48123, 26582, 48266, 12414, 48337, 52882, 48357,
      59943, 48285, 21767, 15236, 41085, 15662, 39140, 15810, 861,   15906,
      5229,  15978, 58828, 16026, 62804, 16064, 61383, 16101, 29649, 16132,
      41366, 16148, 22326, 16163, 34221, 16176, 12686, 16188, 27824, 16198,
      20779, 16207, 64905, 16214, 37388, 16221, 11971, 16227, 62037, 16231,
      63808, 16235, 23923, 16239, 13868, 16242, 38907, 16244, 38114, 16246,
      15491, 16248, 40025, 16249, 49135, 16250, 45340, 16251, 30778, 16252,
      7255,  16253, 41826, 16253, 4694,  16254, 27999, 16254, 47094, 16254,
      62719, 16254, 9954,  16255, 20380, 16255, 28882, 16255, 35808, 16255,
      41445, 16255, 46028, 16255, 49751, 16255, 52773, 16255, 55223, 16255,
      57208, 16255, 58816, 16255, 60116, 16255, 61168, 16255, 62017, 16255,
      62703, 16255, 63257, 16255, 63703, 16255, 64062, 16255, 64352, 16255,
      64585, 16255, 64773, 16255, 64924, 16255};
};

} // namespace vaip_vaiml_custom_op
