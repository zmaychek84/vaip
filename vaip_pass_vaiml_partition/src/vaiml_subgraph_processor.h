// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "vaip/vaip.hpp"
#include "vitis/ai/env_config.hpp"
#include <ctime>
#include <nlohmann/json.hpp>

#include "pass_imp.hpp"
#include "vaiml_partition_load_wts.h"

using json = nlohmann::json;
using VaimlTensorShape = std::vector<int64_t>;
using VaimlShapeDict = std::map<std::string, VaimlTensorShape>;
using VaimlShapeVec = std::vector<VaimlTensorShape>;

using namespace vaip_core;
DEF_ENV_PARAM(DEBUG_VAIML_PARTITION, "0")
// n: DEBUG verbosity level
// 1: Key DEBUG messages
// 2: Verbose DEBUG messages
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_VAIML_PARTITION) >= n)

#include <iostream>
template <typename... Args> inline void vaiml_print_func(Args&&... args) {
  (std::cout << ... << args);
}

// This is to work around an issue that lit deos not work with sterr
#define VAIML_DEBUG_PRINT(...)                                                 \
  if (ENV_PARAM(DEBUG_VAIML_PARTITION) >= 1) {                                 \
    vaiml_print_func(__VA_ARGS__);                                             \
    std::cout << std::endl;                                                    \
  }

#define VAIML_DEBUG_PRINT2(...)                                                \
  if (ENV_PARAM(DEBUG_VAIML_PARTITION) >= 2) {                                 \
    vaiml_print_func(__VA_ARGS__);                                             \
    std::cout << std::endl;                                                    \
  }

// #define VAIP_PASS_VAIML_PARTITION_PROFILING

namespace vaip_vaiml_subgraph_processor {

struct ConstantInfo {
  size_t offset;
  size_t size;
  int type;
  std::vector<int> shape;
};

struct IndexedSubGraph {
  std::vector<size_t> all_nodes;
  std::vector<std::string> all_inputs;
  std::vector<std::string> all_inits;
  std::vector<std::string> all_outputs;
  std::filesystem::path save_path;
  std::string name;
  size_t name_id;
  VaimlShapeVec input_shapes;
  VaimlShapeVec output_shapes;
};

struct NodeWithNodeArg {
  const onnxruntime::Node* node;
  const onnxruntime::NodeArg* nodeArg;
  NodeWithNodeArg(const onnxruntime::Node* n, const onnxruntime::NodeArg* na)
      : node(n), nodeArg(na) {}
};

class VaimlSubgraphProcessor {
public:
  VaimlSubgraphProcessor(Graph& graph, IPass& self);
  VaimlSubgraphProcessor() = default;

  ~VaimlSubgraphProcessor(){

  };
  std::vector<std::vector<size_t>> GetSupportedNodes(const Graph& graph) const;
  // partition function: generate subgraphs supported VAIML
  std::vector<std::unique_ptr<IndexedSubGraph>>
  GetPartitions(const Graph& graph) const;
  const Node* fuseGraph(IndexedSubGraph& subgraph) const;

  std::map<std::string, std::vector<std::string>>
  getOpsFromJson(const Graph& graph, const std::string& opsFileName) const;

  // top flow
  void process() const;
  bool isFusableNode(std::string& nodeName) const;
  bool isNodeSupported(const Graph& graph,
                       std::unordered_set<std::string>& deviceOps,
                       const Node* node, bool supported_node_mode) const;

  bool
  isNodeTypeSupported(const Graph& graph,
                      std::unordered_map<std::string, int>& supportedOpTypes,
                      const Node* node) const;

  bool getNodeShapesAndAttributes(const Node* node, int64_t& B, int64_t& M,
                                  int64_t& K, int64_t& N,
                                  int64_t& weightQuantBits,
                                  int64_t& blockSize) const;

  void setVaimlConfigOptions();
  void dumpConstants(const Graph& graph);
  void setMsigOpsMap();

  int saveMemoryToCache(const char* mem, size_t mem_size,
                        std::filesystem::path cache_dir, std::string filename);
  void loadAdd128(std::vector<uint8_t>& dst, int8_t* src, int size);
  int htGenerateLstmInput(const LstmSettings& s,
                          const struct lstm_init_wts& lstm_in, uint8_t* result,
                          std::filesystem::path cache_dir);
  std::vector<uint8_t> ht_wts_gen_lstm_b2b(const lstm_init_wts& param,
                                           std::filesystem::path cache_dir);
  void InitHtLstmWeights();

private:
  Graph& graph_;
  IPass& self_;

  std::unordered_set<std::string> vaiml_supported_ops_;

  //  dump subgraphs to onnx format for debugging purpose
  std::string model_name_ = "";
  bool debug_ = false;
  bool binary_module_ = true;
#ifdef _WIN32
  std::string vaiml_model_path_ = "C:\\amd\\voe\\binary-modules\\ResNet.flexml";
#else
  std::string vaiml_model_path_ = "./vaiml_partition.flexml";
#endif
  std::string device_name_ = "phx";
  int max_num_inputs_ = -1;
  int max_num_outputs_ = -1;
  int max_num_partitions_ = 1;
  std::string output_type_ = "aie-exe";
  std::string config_filename_ = "";
  bool run_vaiml_compile_ = false;
  bool supported_node_mode_ = true;
  bool ai_analyzer_profiling_ = false;
  bool ai_analyzer_visualization_ = false;
  std::string aie_unsupported_ops_override_;
  bool force_ = false;
  std::string vitisai_root_dir_ = "";
  std::string init_m_values_ = "1,8";
  std::vector<int64_t> initMValues_;
  std::string custom_ops_repo_;
  std::string overlay_json_ = "";
  std::string overlay_json_signature_ = "";
  std::string tilingEngine = "microkernel";

  std::unordered_map<std::string,
                     std::unordered_map<std::string, std::vector<std::string>>>
      subgraph_map_;
  std::vector<std::string> fusableOps_;

  std::string constants_file_name_ = "wts.bin";
  std::unordered_map<std::string, ConstantInfo> constants_map_;
  std::string full_model_hash_;
  std::map<std::string, std::map<std::string, std::vector<std::string>>>
      msig_ops_map_;

  std::filesystem::path cache_dir_;
  std::unordered_map<std::string, char*> ht_weights_preformat_;
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
      // lstm0_h_wts 48
      "/decoder/rnn/Unsqueeze_1_output_0_quantized",
      // lstm0_x_wts 49
      "/decoder/rnn/Unsqueeze_output_0_quantized",
      // lstm0_bias 50
      "/decoder/rnn/Unsqueeze_2_output_0_quantized",
      // lstm1_h_wts 51
      "/decoder/rnn/Unsqueeze_4_output_0_quantized",
      // lstm1_x_wts 52
      "/decoder/rnn/Unsqueeze_3_output_0_quantized",
      // lstm1_bias 53
      "/decoder/rnn/Unsqueeze_5_output_0_quantized"};
};

} // namespace vaip_vaiml_subgraph_processor
