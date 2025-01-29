// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
#include <cctype>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <regex>

#include <xir/util/tool_function.hpp>

#include <cassert>
#include <codecvt>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <istream>
#include <memory>
#include <nlohmann/json.hpp>

#include "gt_initializer_mapping_subpass.h"
#include "ht_initializer_mapping_subpass.h"
#include "vaiml_subgraph_processor.h"
#include "vaip/vaip.hpp"
#ifdef _WIN32
#  pragma warning(disable : 4244)
// #pragma warning(disable : 4840)
#endif

using namespace ONNX_NAMESPACE;

namespace vaip_vaiml_subgraph_processor {

constexpr const char* VAIML = "VAIML";

std::string GetOriginalModelHash(std::string sig_file) {
  std::ifstream sig_fs(sig_file);
  if (!sig_fs.is_open()) {
    return "NONE";
  }
  // read from flexml_bm.signature and compare with current input graph hash
  // value
  std::string hash_value;
  getline(sig_fs, hash_value);

  return hash_value;
}

std::map<std::string, std::vector<std::string>>
VaimlSubgraphProcessor::getOpsFromJson(const Graph& graph,
                                       const std::string& opsFileName) const {
  std::ifstream opsFile(opsFileName);

  nlohmann::json jsonData;
  try {
    jsonData = nlohmann::json::parse(opsFile);
  } catch (const std::exception& e) {
    LOG(FATAL) << "Error: Failed to parse JSON from file: " << opsFileName
               << ": " << e.what() << std::endl;
  }
  opsFile.close();
  std::map<std::string, std::vector<std::string>> subgraphNodesMap;
  for (auto& kv : jsonData.items()) {
    std::string key = kv.key();
    std::vector<std::string> values = kv.value();
    subgraphNodesMap[key] = values;
  }

  return subgraphNodesMap;
}

std::vector<const Node*> getConsumerNodes(const Graph& graph,
                                          const Node* node) {
  std::vector<const Node*> ret;
  auto outputs = node_get_output_node_args(*node);
  for (auto output : outputs) {
    auto consumers =
        graph_get_consumer_nodes(graph, node_arg_get_name(*output));

    for (auto c : consumers) {
      ret.push_back(c);
    }
  }
  return ret;
}

std::vector<NodeWithNodeArg> getConsumerNodesWithSrcNodeArgs(const Graph& graph,
                                                             const Node* node) {
  std::vector<NodeWithNodeArg> ret;
  auto outputs = node_get_output_node_args(*node);
  for (auto output : outputs) {
    auto consumers =
        graph_get_consumer_nodes(graph, node_arg_get_name(*output));
    for (auto c : consumers) {
      NodeWithNodeArg result(c, output);
      ret.push_back(result);
    }
  }
  return ret;
}

bool isConsumer(const Graph& graph, std::vector<const Node*>& cur_group_nodes,
                const Node* node) {
  bool res = false;
  for (const auto* cur_node : cur_group_nodes) {
    // get cur_node cosnumer and match with node name
    auto consumers = getConsumerNodes(graph, cur_node);
    for (auto c : consumers) {
      auto consumerName = VAIP_ORT_API(node_get_name)(*c);
      // for (auto it = cur_node->OutputEdgesBegin(), end =
      // cur_node->OutputEdgesEnd(); it != end; ++it) { auto consumerName =
      // it->GetNode().Name();
      auto nodeName = VAIP_ORT_API(node_get_name)(*node);
      if (consumerName == nodeName) {
        res = true;
        break;
      }
    }
    if (res) {
      break;
    }
  }
  return res;
}

/*
  Get Producers for the node
*/
std::vector<const Node*> getProducerNodes(const Graph& graph,
                                          const Node* node) {
  std::vector<const Node*> ret;
  auto inputs = node_get_input_node_args(*node);
  for (auto input : inputs) {
    auto producer =
        VAIP_ORT_API(graph_producer_node)(graph, node_arg_get_name(*input));

    if (producer != nullptr)
      ret.push_back(producer);
  }
  return ret;
}

bool isIsolated(size_t idx, std::vector<size_t>& node_idx_vec,
                const Graph& graph) {
  bool res = true;
  const auto* cur_node = VAIP_ORT_API(graph_get_node)(graph, idx);
  for (size_t i : node_idx_vec) {
    const auto* node = VAIP_ORT_API(graph_get_node)(graph, i);
    for (const auto* input : node_get_input_node_args(*node)) {
      // if the node input was not produced by this subgraph, add it to the
      // subgraph inputs.

      if (!node_arg_exists(*input)) {
        continue;
      }
      std::string inputArgName = node_arg_get_name(*input);
      const Node* producer =
          VAIP_ORT_API(graph_producer_node)(graph, inputArgName);
      if ((producer != nullptr) && (VAIP_ORT_API(node_get_name)(*producer) ==
                                    VAIP_ORT_API(node_get_name)(*cur_node))) {
        res = false;
        break;
      }
    }
    if (!res) {
      break;
    }
    auto consumers = getConsumerNodes(graph, node);
    for (auto c : consumers) {
      auto consumerName = VAIP_ORT_API(node_get_name)(*c);
      auto nodeName = VAIP_ORT_API(node_get_name)(*cur_node);
      if (consumerName == nodeName) {
        res = false;
        break;
      }
    }
    if (!res) {
      break;
    }
  }
  if (res) {
    // std::cout << "    Remove orphan node "
    //           << VAIP_ORT_API(node_get_name)(*cur_node)
    //           << " from current subgraph" << std::endl;
  }
  return res;
}

// #endif

void VaimlSubgraphProcessor::setVaimlConfigOptions() {
  auto& vaiml_proto = self_.get_pass_proto().vaiml_config();
  auto& config_proto = self_.get_context()->get_config_proto();

  // get config filename and parse json
  // change to use single json file:
  // config_filename_ = vaiml_proto.config_file();
  auto all_session_options = config_proto.provider_options();

  // auto it = all_session_options.find("config_file");
  // if (it != all_session_options.end() && !it->second.empty()) {
  //   config_filename_ = it->second;
  // } else {
  //   LOG(FATAL) << "Error: Key 'config_file' not found in session options ";
  //   return;
  // }

  // std::ifstream f(config_filename_);
  // if (!f.is_open()) {
  //   LOG(FATAL) << "Error: Failed to open file: " << config_filename_;
  //   return;
  // }

  // get AI Analyzer user settings
  // NOTE: visualization and profiling options can be set in config_file as
  // well,
  //       but the settings in provider_options will take precedence
  if (config_proto.has_ai_analyzer_visualization()) {
    ai_analyzer_visualization_ = config_proto.ai_analyzer_visualization();
  }
  // DL Analyzer creating dpu_timestamp_info.json, or not.
  if (config_proto.has_ai_analyzer_profiling()) {
    ai_analyzer_profiling_ = config_proto.ai_analyzer_profiling();
  }

  if (config_proto.provider_options().find("model_name") !=
      config_proto.provider_options().end()) {
    model_name_ = config_proto.provider_options().at("model_name");
    VAIML_DEBUG_PRINT2("model recognized as: ", model_name_);
  }
  // Get rest options from json
  if (vaiml_proto.has_vaiml_model_path()) {
    vaiml_model_path_ = vaiml_proto.vaiml_model_path();
  }

  if (vaiml_proto.has_device_name()) {
    device_name_ = vaiml_proto.device_name();
  }

  if (device_name_ != "v70" && device_name_ != "phx" &&
      device_name_ != "ryzen-ai" && device_name_ != "stx" &&
      device_name_ != "ryzen-ai" && device_name_ != "cpu")
    LOG(FATAL) << "Supported device names are v70, phx, stx and cpu";

  if (vaiml_proto.has_force()) {
    force_ = vaiml_proto.force();
  }

  if (vaiml_proto.has_output_type()) {
    output_type_ = vaiml_proto.output_type();
  }

  if (vaiml_proto.has_debug()) {
    // debug_ = vaiml_proto.debug();
    debug_ = 0;
  }

  if (vaiml_proto.has_binary_module()) {
    binary_module_ = vaiml_proto.binary_module();
  }

  if (vaiml_proto.has_run_vaiml_compile()) {
    run_vaiml_compile_ = vaiml_proto.run_vaiml_compile();
  }

  if (vaiml_proto.has_max_num_inputs()) {
    max_num_inputs_ = vaiml_proto.max_num_inputs();
  }

  if (vaiml_proto.has_supported_node_mode()) {
    supported_node_mode_ = vaiml_proto.supported_node_mode();
  }

  if (vaiml_proto.has_max_num_outputs()) {
    max_num_outputs_ = vaiml_proto.max_num_outputs();
  }

  if (vaiml_proto.has_max_num_partitions()) {
    max_num_partitions_ = vaiml_proto.max_num_partitions();
  }

  if (vaiml_proto.has_aie_unsupported_ops_override()) {
    aie_unsupported_ops_override_ = vaiml_proto.aie_unsupported_ops_override();
  }

  if (vaiml_proto.has_init_m_values()) {
    init_m_values_ = vaiml_proto.init_m_values();
  }

  if (vaiml_proto.has_overlay_json()) {
    overlay_json_ = vaiml_proto.overlay_json();
    overlay_json_signature_ = xir::get_md5_of_file(overlay_json_);
  }

  // Set initMValues_ from init_m_values_ string
  // Use stringstream to split the line by commas
  std::stringstream ss(init_m_values_);
  std::string token;
  while (std::getline(ss, token, ',')) {
    initMValues_.push_back(std::stoi(token));
  }

  // std::string infos =
  //     device_name_ + output_type_ + std::to_string(max_num_inputs_) +
  //     std::to_string(max_num_outputs_) + std::to_string(max_num_partitions_);

  // auto config_hash = xir::get_md5_of_buffer(infos.c_str(), infos.length());

  // std::ofstream sig_file("original-info-signature.txt");
  // if (sig_file.is_open()) {
  //   sig_file << config_hash;
  //   sig_file.close();
  //   // std::cout << "Original info signature saved to
  //   // original-info-signature.txt."
  //   //           << std::endl;
  // }
}

void VaimlSubgraphProcessor::setMsigOpsMap() {
  // subgraph mappings are versioned so that partitions can work with all
  // future versions of VAIP

  // Use separate header files for each model and version to make it more
  // managable
#include "gt_inner_map_1_2.h"
  msig_ops_map_.insert(std::make_pair("GT_v1.2", gt_inner_map_1_2));
}

VaimlSubgraphProcessor::VaimlSubgraphProcessor(Graph& graph, IPass& self)
    : graph_(graph), self_(self) {

  // check config_file has vaiml key option and update the corresponding value
  // for vaiml option
  setVaimlConfigOptions();
  dumpConstants(graph);
  fusableOps_.push_back("QuantizeLinear");
  full_model_hash_ = self_.get_context()->get_config_proto().cache_key();
  // Cache xclbin during context creation
  auto& session_option =
      self_.get_context()->get_config_proto().provider_options();
  (void)self_.get_context()->xclbin_path_to_cache_files(
      session_option.at("xclbin"));

  if (debug_) {
    std::cout << "INFO:: Print VAIML config options: " << std::endl;
    std::cout << " vaiml_model_path: " << vaiml_model_path_ << "\n"
              << " device_name: " << device_name_ << "\n"
              << " output_type: " << output_type_ << "\n"
              << " config_filename: " << config_filename_ << "\n"
              << " debug: " << debug_ << "\n"
              << " supported_node_mode: " << supported_node_mode_ << "\n"
              << " max_num_inputs: " << max_num_inputs_ << "\n"
              << " max_num_outputs: " << max_num_outputs_ << "\n"
              << " max_num_partitions: " << max_num_partitions_ << "\n"
              << " ai_analyzer_visualization: " << ai_analyzer_visualization_
              << "\n"
              << " ai_analyzer_profiling: " << ai_analyzer_profiling_ << "\n"
              << " init_token_lengths: " << init_m_values_ << "\n"
              << " aie_unsupported_ops_override: "
              << aie_unsupported_ops_override_ << "\n"
              << " custom_ops_repo: " << custom_ops_repo_ << std::endl;
  }

  // save model to cache dir for debug
  if (debug_) {
    std::filesystem::path save_path =
        self.get_context()->get_log_dir() / "cloned_graph.onnx";
    VAIP_ORT_API(graph_save)
    (graph, save_path.string(), "", std::numeric_limits<size_t>::max());
    MY_LOG(1) << "model saved to " << save_path;
  }

  setMsigOpsMap();
}

/*
    Check if the op given by nodeName is a fusable op
*/
bool VaimlSubgraphProcessor::isFusableNode(std::string& nodeName) const {
  bool res = false;
  for (auto i = 0; i < (int)(fusableOps_.size()); ++i) {
    auto opName = fusableOps_[i];
    if (nodeName.find(opName) != std::string::npos) {
      res = true;
      break;
    }
  }
  return res;
}

/*
   check if the node is supported on the device
*/
bool VaimlSubgraphProcessor::isNodeSupported(
    // const Graph& graph, std::unordered_map<std::string, int>& deviceOps,
    const Graph& graph, std::unordered_set<std::string>& deviceOps,
    const Node* node, bool supported_node_mode) const {

  auto nodeName = VAIP_ORT_API(node_get_name)(*node);

  // To avoid collisions between existing node names, we do the following
  // 1) If the node name is empty,
  //    we assign "=<op_type>-><name of first output>"
  // 2) else if the node name starts with "=",
  //    we replace that part with "=="
  // 3) else stays as is
  // Keep in sync with assign_unique_names() in vaiml.py!
  if (nodeName.empty()) {
    auto outputs = node_get_output_node_args(*node);
    // Need an output to guarantee uniqueness; but right now ONNX doesn't
    // contain any op without outputs, so this should never happen.
    if (!outputs.empty())
      nodeName =
          "=" + node_op_type(*node) + "->" + node_arg_get_name(*outputs[0]);
  } else if (nodeName[0] == '=') {
    nodeName = "=" + nodeName;
  }

  bool supported;
  if (supported_node_mode) {
    supported = deviceOps.find(nodeName) != deviceOps.end();
  } else {
    // a node is supported if it is not on the unsupported ops list
    supported = deviceOps.find(nodeName) == deviceOps.end();
  }
  // if (supported) {
  //   std::cout << "    supported: ";
  // } else {
  //   std::cout << "    not supported: ";
  // }
  // std::cout << nodeName
  //           << " (original name: " << VAIP_ORT_API(node_get_name)(*node)
  //           << ")\n";
  return supported;
}

size_t getInputEdgesCount(const Node* node) {
  auto inputs = node_get_inputs(*node);
  size_t ret = 0;
  for (auto input : inputs) {
    if (input.node != nullptr && input.node_arg != nullptr) {
      ret += 1;
    }
  }
  return ret;
}

std::vector<PartitionInfo>
VaimlSubgraphProcessor::GetSupportedNodes(const Graph& graph) const {
  // check if the partition needs to be recompiled based on original model
  // signature
  // std::string currentHash =
  //    GetOriginalModelHash("original-model-signature.txt");
  // std::string vaiml_bm_file_path =
  //    vaiml_model_path_ + "/original-model-signature.txt";
  // std::string compiledHash = GetOriginalModelHash(vaiml_bm_file_path);

  std::string opsFileName;
  if (supported_node_mode_) {
    opsFileName = vaiml_model_path_ + "/aie_supported_ops.json";
  } else {
    opsFileName = vaiml_model_path_ + "/aie_unsupported_ops.json";
  }
  // if (currentHash != compiledHash) {
  //   LOG(FATAL)
  //       << "Input onnx model signature doesn't match compiled model
  //       signature";
  // }

  std::vector<std::vector<size_t>> supported_node_vecs;
  std::map<std::string, size_t> nodename_index_map;
  // json get node names => need to get node idx from node name
  // traverse the graph and get node name. assign node name if it doesn't have.
  const auto& node_indices = graph_get_node_in_topoligical_order(graph);
  // name the node if nodename is empty
  for (auto node_indice : node_indices) {
    auto node = VAIP_ORT_API(graph_get_node)(graph, node_indice);
    auto nodeName = VAIP_ORT_API(node_get_name)(*node);
    if (nodeName.empty()) {
      auto outputs = node_get_output_node_args(*node);
      // Need an output to guarantee uniqueness; but right now ONNX doesn't
      // contain any op without outputs, so this should never happen.
      if (!outputs.empty())
        nodeName =
            "=" + node_op_type(*node) + "->" + node_arg_get_name(*outputs[0]);
    } else if (nodeName[0] == '=') {
      nodeName = "=" + nodeName;
    }
    nodename_index_map[nodeName] = node_indice;
    // VAIML_DEBUG_PRINT("nodeName: ", nodeName, " node indice: ", node_indice);
  }
  // form the results according to the user provided input
  std::map<std::string, std::vector<std::string>> subgraphNodesMap;
  // auto deviceUnsupportedOps = getUnsupportedOps(graph);
  std::ifstream opsFile(opsFileName);
  if (!opsFile) {
    LOG(INFO) << "INFO: Reading ops from default";
    if (msig_ops_map_.find(model_name_) != msig_ops_map_.end()) {
      subgraphNodesMap = msig_ops_map_.at(model_name_);
    } else {
      LOG(INFO) << "ERROR: model name " << model_name_
                << "is not set for DeviceOps";
    }
  } else {
    LOG(INFO) << "INFO: Reading ops from " << opsFileName;
    subgraphNodesMap = getOpsFromJson(graph, opsFileName);
  }
  opsFile.close();

  std::vector<PartitionInfo> node_groups;

  for (auto kv : subgraphNodesMap) {
    std::vector<size_t> supported_node_vec;
    std::string partition_name = kv.first;
    for (auto nodeName : kv.second) {
      if (nodename_index_map.find(nodeName) != nodename_index_map.end()) {
        supported_node_vec.push_back(nodename_index_map[nodeName]);
      } else {
        LOG(INFO) << "WARNING:: nodeName " << nodeName
                  << " can not be found in graph, skipping";
      }
    }
    node_groups.push_back({supported_node_vec, partition_name});
  }

  return node_groups;
}

int VaimlSubgraphProcessor::saveMemoryToCache(const char* mem, size_t mem_size,
                                              std::filesystem::path cache_dir,
                                              std::string filename) const {
  self_.get_context()->write_file(filename + ".bin",
                                  gsl::span<const char>(mem, mem_size));
  VAIML_DEBUG_PRINT(mem_size, " bytes of memory saved to cache ", filename);
  return 0;
}

void VaimlSubgraphProcessor::loadAdd128(std::vector<uint8_t>& dst, int8_t* src,
                                        int size) const {
  for (int i = 0; i < size; i++) {
    dst.push_back(static_cast<uint8_t>(src[i] + 128));
  }
  return;
}

int VaimlSubgraphProcessor::htGenerateLstmInput(
    const LstmSettings& s, const struct lstm_init_wts& lstm_in, uint8_t* result,
    std::filesystem::path cache_dir) const {
  VAIML_DEBUG_PRINT("htGenerateLstmInput layer ", s.layer_name);
  auto cachFilepath = cache_dir / (s.layer_name + ".bin");
  if (std::filesystem::exists(cachFilepath)) {
    auto wts_size = std::filesystem::file_size(cachFilepath);
    VAIML_DEBUG_PRINT("    Using cached weights from ", cachFilepath,
                      ". Weight size: ", wts_size);
    return 0;
  }

  double Sx, Sw, Sr, Sb, Sh, Sc, Sy1, Sy2;
  int Zx, Zw, Zr, Zb, Zh, Zc, Zy1, Zy2;
  int Sg = 1;
  int Zg = 0;
  char* lstm_rpt_ptr{};
  uint8_t nonlinear_in_shift{};
  if (s.layer_id == 320) {
    Sx = lstm_in.scale[9];
    Zx = lstm_in.zp[9];
    Sw = lstm_in.scale[10];
    Zw = lstm_in.zp[10];
    Sr = lstm_in.scale[11];
    Zr = lstm_in.zp[11];
    Sb = lstm_in.scale[12];
    Zb = lstm_in.zp[12];
    Sh = lstm_in.scale[4];
    Zh = lstm_in.zp[4];
    Sc = lstm_in.scale[1];
    Zc = lstm_in.zp[1];
    Sy1 = lstm_in.scale[14];
    Zy1 = lstm_in.zp[14];
    Sy2 = lstm_in.scale[15];
    Zy2 = lstm_in.zp[15];
    lstm_rpt_ptr = (char*)(lstm_in.lstm_320_rtp);
    nonlinear_in_shift = *((uint8_t*)(lstm_rpt_ptr + 16));
  } else if (s.layer_id == 1024) {
    Sx = lstm_in.scale[13];
    Zx = lstm_in.zp[13];
    Sw = lstm_in.scale[16];
    Zw = lstm_in.zp[16];
    Sr = lstm_in.scale[17];
    Zr = lstm_in.zp[17];
    Sb = lstm_in.scale[18];
    Zb = lstm_in.zp[18];
    Sh = lstm_in.scale[5];
    Zh = lstm_in.zp[5];
    Sc = lstm_in.scale[2];
    Zc = lstm_in.zp[2];
    Sy1 = lstm_in.scale[20];
    Zy1 = lstm_in.zp[20];
    Sy2 = lstm_in.scale[21];
    Zy2 = lstm_in.zp[21];
    lstm_rpt_ptr = (char*)(lstm_in.lstm_1024_rtp);
    nonlinear_in_shift = *((uint8_t*)(lstm_rpt_ptr + 16));
  }

  int8_t* x_wts_p;
  int8_t* h_wts_p;
  int8_t* b_wts_p;
  if (s.layer_id == 320) {
    x_wts_p = (lstm_in.lstm0_x_wts);
    h_wts_p = (lstm_in.lstm0_h_wts);
    b_wts_p = (lstm_in.lstm0_bias);
  } else {
    x_wts_p = (lstm_in.lstm1_x_wts);
    h_wts_p = (lstm_in.lstm1_h_wts);
    b_wts_p = (lstm_in.lstm1_bias);
  }
  std::vector<uint8_t> w_u8; // (x_wts_p, x_wts_p + 4096 * s.len_x);
  std::vector<uint8_t> r_u8; // (h_wts_p, h_wts_p + 4096 * s.len_h);
  std::vector<uint8_t> b_u8; // (b_wts_p, b_wts_p + 8192);
  loadAdd128(w_u8, x_wts_p, 4096 * s.len_x);
  loadAdd128(r_u8, h_wts_p, 4096 * s.len_h);
  loadAdd128(b_u8, b_wts_p, 8192);

  WTensor<int32_t> w_i32 = WTensor<int32_t>::createFromVector(w_u8);
  WTensor<int32_t> r_i32 = WTensor<int32_t>::createFromVector(r_u8);
  WTensor<int32_t> b_i32 = WTensor<int32_t>::createFromVector(b_u8);
  // w_i32.print("### wi32: ", PARTIAL_DATA);
  // r_i32.print("### ri32: ", PARTIAL_DATA);
  // b_i32.print("### bi32: ", PARTIAL_DATA);

  // b_i32.print("b_i32: ", PARTIAL_DATA);
  auto bw_i32 = b_i32.slice({0}, {4096}).reshape({4096});
  // bw_i32.print("bw_i32: ", PARTIAL_DATA);
  auto br_i32 = b_i32.slice({4096}, {8192}).reshape({4096});
  // br_i32.print("br_i32: ", PARTIAL_DATA);

  // printf("###--- len_x:%d, len_h:%d\n", s.len_x, s.len_h);
  auto SWk_i32 = w_i32.reshape({4096, s.len_x})
                     .transpose({1, 0})
                     .reduceSum(0)
                     .reshape({4096});
  // SWk_i32.print("SWk_i32: ", PARTIAL_DATA);
  auto SRk_i32 = r_i32.reshape({4096, s.len_h})
                     .transpose({1, 0})
                     .reduceSum(0)
                     .reshape({4096});
  // SRk_i32.print("SRk_i32: ", PARTIAL_DATA);

  // Calculate QDQ values
  double Qx = Sx * Sw / Sg;
  double Qh = Sh * Sr / Sg;
  double Qa = Qx * Zw;
  double Qb = Qh * Zr;
  // printf("--- Qx:%.6f, Qh:%.6f, Qa:%.6f, Qb:%.6f\n", Qx, Qh, Qa, Qb);
  WTensor<double> bw_dbl = WTensor<double>::createFromVector(bw_i32.data());
  WTensor<double> br_dbl = WTensor<double>::createFromVector(br_i32.data());
  WTensor<double> SWk_dbl = WTensor<double>::createFromVector(SWk_i32.data());
  WTensor<double> SRk_dbl = WTensor<double>::createFromVector(SRk_i32.data());
  // SWk_dbl.print("SWk_dbl: ", PARTIAL_DATA);
  // SRk_dbl.print("SRk_dbl: ", PARTIAL_DATA);
  double Qc_0 = Qx * s.len_x * Zx * Zw + Qh * s.len_h * Zh * Zr + Zg;
  auto Qc_1 = SWk_dbl.mul((-1) * Qx * Zx);
  auto Qc_2 = SRk_dbl.mul((-1) * Qh * Zh);
  // Qc_2.print("Qc_2: ", PARTIAL_DATA);
  // bw_dbl.print("bw_dbl: ", PARTIAL_DATA);
  // br_dbl.print("br_dbl: ", PARTIAL_DATA);
  auto Qc_3 = bw_dbl.add(br_dbl);
  // Qc_3.print("Qc_3: ", PARTIAL_DATA);
  auto Qc_4 = Qc_3.sub(2 * Zb);
  auto Qc_5 = Qc_4.mul(Sb / Sg);
  // Qc_5.print("Qc_5: ", PARTIAL_DATA);
  auto Qc_6 = Qc_1.add(Qc_2);
  // Qc_6.print("Qc_6: ", PARTIAL_DATA);
  auto Qc_7 = Qc_6.add(Qc_5);
  // Qc_7.print("Qc_7: ", PARTIAL_DATA);
  auto Qc_8 = Qc_7.add(Qc_0);
  // Qc_8.print("Qc_8: ", PARTIAL_DATA);
  auto Qc_9 = Qc_8.mul(std::pow(2, nonlinear_in_shift));
  // Qc_9.print("Qc_9: ", PARTIAL_DATA);
  WTensor<int32_t> NB_i32 = WTensor<int32_t>::createFromVector(Qc_9.data());
  // NB_i32.print("--- Qc int(NB_i32): ", PARTIAL_DATA);

  dims_t order = {2, 3, 0, 1};
  WTensor<uint8_t> w0_u8 = WTensor<uint8_t>::createFromVector(w_u8);
  WTensor<uint8_t> r0_u8 = WTensor<uint8_t>::createFromVector(r_u8);
  WTensor<uint8_t> w1_u8 =
      w0_u8.reshape({4, 1024, s.len_x}).reorder(order).reshape({4096, s.len_x});
  WTensor<uint8_t> r1_u8 =
      r0_u8.reshape({4, 1024, s.len_h}).reorder(order).reshape({4096, s.len_h});
  WTensor<int32_t> nb_i32 =
      NB_i32.reshape({4, 1024}).reorder(order).reshape({1, 4096});

  // w1_u8.print("--- post reorder w1_u8: ", PARTIAL_DATA);
  // r1_u8.print("--- post reorder r1_u8: ", PARTIAL_DATA);
  // nb_i32.print("--- post reorder nb_i32: ", PARTIAL_DATA);

  // ----------------- write_tvs -----------------
  int max_K = std::max(s.len_x, s.len_h);
  auto w1_u8_t = w1_u8.reshape({4096, s.len_x}).transpose({1, 0});
  auto r1_u8_t = r1_u8.reshape({4096, s.len_h}).transpose({1, 0});
  auto W_pad_u8 = w1_u8_t.pad(
      {{0, max_K - s.len_x},
       {0, 0}}); // .reshape({max_K / s.sv_K, s.sv_K, s.n_iter * s.sv_N});
  auto R_pad_u8 = r1_u8_t.pad(
      {{0, max_K - s.len_h},
       {0, 0}}); // .reshape({max_K / s.sv_K, s.sv_K, s.n_iter * s.sv_N});
  // W_pad_u16.print("W_pad_u16: ", PARTIAL_DATA);
  // R_pad_u16.print("R_pad_u16: ", PARTIAL_DATA);
  // ----------------- write_wts -----------------
  int ssv_N = s.sv_N / 4;
  int K = s.len_kp;
  int sv_K = s.sv_K;

  // auto W_pad_u8 = WTensor<uint8_t>::createFromVector(W_pad_u16.data());
  // auto R_pad_u8 = WTensor<uint8_t>::createFromVector(R_pad_u16.data());

  auto W_reshaped_ = W_pad_u8.reshape(
      {K / sv_K, sv_K, 4, s.n_iter / s.num_row, s.num_row, ssv_N});
  auto W_transposed_u8 = W_reshaped_.transpose({3, 0, 4, 1, 2, 5});
  auto R_reshaped_ = R_pad_u8.reshape(
      {K / sv_K, sv_K, 4, s.n_iter / s.num_row, s.num_row, ssv_N});
  auto R_transposed_u8 = R_reshaped_.transpose({3, 0, 4, 1, 2, 5});
  auto B_reshaped_i32 = nb_i32.reshape({4, s.len_h});
  // W_transposed_u8.print("### W_transposed_u8: ", TYPE_SHAPE);
  // R_transposed_u8.print("### R_transposed_u8: ", TYPE_SHAPE);

  dims_t sv_W_start = {0, 0, 0, 0, 0, 0};
  dims_t sv_W_end = W_transposed_u8.shape();
  unsigned char* wts = result;
  unsigned char* p = wts;
  for (int idx_n_col = 0; idx_n_col < (s.n_iter / s.num_row); idx_n_col++) {
    for (int idx_K = 0; idx_K < (max_K / s.sv_K); idx_K++) {
      for (int idx_row = 0; idx_row < s.num_row; idx_row++) {
        sv_W_start[0] = idx_n_col;
        sv_W_start[1] = idx_K;
        sv_W_start[2] = idx_row;
        sv_W_end[0] = idx_n_col + 1;
        sv_W_end[1] = idx_K + 1;
        sv_W_end[2] = idx_row + 1;
        auto sv_W = W_transposed_u8.slice(sv_W_start, sv_W_end)
                        .reshape({sv_K, 4 * ssv_N});
        auto sv_R = R_transposed_u8.slice(sv_W_start, sv_W_end)
                        .reshape({sv_K, 4 * ssv_N});
        sv_W.reshape({s.sv_K / 8, 8, s.sv_N});
        sv_R.reshape({s.sv_K / 8, 8, s.sv_N});
        dims_t svm_s0 = {0, 0, 0};
        dims_t svm_e0 = sv_W.shape();
        svm_e0[1] = 4;
        dims_t svm_s1 = {0, 0, 0};
        dims_t svm_e1 = sv_W.shape();
        svm_s1[1] = 4;
        svm_e1[1] = 8;
        auto svm0_W = sv_W.slice(svm_s0, svm_e0)
                          .reshape({s.sv_K / 8, 4, s.sv_N / 8, 8})
                          .transpose({2, 0, 1, 3});
        auto svm1_W = sv_W.slice(svm_s1, svm_e1)
                          .reshape({s.sv_K / 8, 4, s.sv_N / 8, 8})
                          .transpose({2, 0, 1, 3});
        auto svm0_R = sv_R.slice(svm_s0, svm_e0)
                          .reshape({s.sv_K / 8, 4, s.sv_N / 8, 8})
                          .transpose({2, 0, 1, 3});
        auto svm1_R = sv_R.slice(svm_s1, svm_e1)
                          .reshape({s.sv_K / 8, 4, s.sv_N / 8, 8})
                          .transpose({2, 0, 1, 3});
        int idx_n_real = idx_n_col * s.num_row + idx_row;
        dims_t svb_s = {0, 0};
        dims_t svb_e = B_reshaped_i32.shape();
        svb_s[1] = idx_n_real * ssv_N;
        svb_e[1] = (idx_n_real + 1) * ssv_N;
        auto sv_B = B_reshaped_i32.slice(svb_s, svb_e);
        memcpy(p, svm0_W.data().data(), svm0_W.size());
        p += svm0_W.size();
        memcpy(p, svm0_R.data().data(), svm0_R.size());
        p += svm0_R.size();
        memcpy(p, svm1_W.data().data(), svm1_W.size());
        p += svm1_W.size();
        memcpy(p, svm1_R.data().data(), svm1_R.size());
        p += svm1_R.size();
        memcpy(p, sv_B.data().data(), sv_B.size() * sizeof(int32_t));
        p += sv_B.size() * sizeof(int32_t);
      }
    }
  }
  memcpy(p, lstm_rpt_ptr, 16 * sizeof(uint32_t));
  p += 16 * sizeof(uint32_t);
  size_t wts_size = p - wts;
  saveMemoryToCache((const char*)wts, wts_size, cache_dir, s.layer_name);
  return 0;
}

std::vector<uint8_t> VaimlSubgraphProcessor::ht_wts_gen_lstm_b2b(
    const lstm_init_wts& param, std::filesystem::path cache_dir) const {
  uint8_t* result = new uint8_t[8448 * 1024 * 2 + 64 * 2 +
                                64 * 2 /* lstm_320_rtp lstm_1024_rtp*/];
#ifdef VAIP_PASS_VAIML_PARTITION_PROFILING
  auto start2 = std::chrono::steady_clock::now();
#endif

  LstmSettings lstm_s320 = LstmSettings(320);
  htGenerateLstmInput(lstm_s320, param, result + 64 * 2, cache_dir);

#ifdef VAIP_PASS_VAIML_PARTITION_PROFILING
  auto end2 = std::chrono::steady_clock::now();
  double time_sec2 = std::chrono::duration<double>(end2 - start2).count();
  std::cout << "    htGenerateLstmInput lstm_s320 time (sec): " << time_sec2
            << std::endl;
#endif

#ifdef VAIP_PASS_VAIML_PARTITION_PROFILING
  auto start3 = std::chrono::steady_clock::now();
#endif
  LstmSettings lstm_s1024 = LstmSettings(1024);
  htGenerateLstmInput(lstm_s1024, param, result + 8448 * 1024 + 64 * 2,
                      cache_dir);

#ifdef VAIP_PASS_VAIML_PARTITION_PROFILING
  auto end3 = std::chrono::steady_clock::now();
  double time_sec3 = std::chrono::duration<double>(end3 - start3).count();
  std::cout << "    htGenerateLstmInput lstm_s1024 time (sec): " << time_sec3
            << std::endl;
#endif

  std::vector<uint8_t> wts(result, result + 8448 * 1024 * 2 + 64 * 2);

  delete[] result;
  return wts;
}

static void InitLstmQdqParams(lstm_init_wts& lstm_in) {
  struct lstm_params_st {
    uint16_t k_iter_cnt_x;
    uint16_t k_iter_cnt_h;
    uint16_t c_sv_idx;
    uint16_t total_iter_cnt;
    uint16_t n_iter_cnt;

    uint8_t xw_shift;
    uint8_t hr_shift;
    uint8_t xs_shift;
    uint8_t hs_shift;
    uint8_t xwq_shift;
    uint8_t hrq_shift;
    uint8_t nonlinear_in_shift;
    uint8_t nonlinear_out_h_shift;
    uint8_t nonlinear_out_c_shift;
    uint8_t output_h_shift;
    uint8_t output_c_shift;
    uint8_t c_shift0;
    uint8_t c_shift1;
    uint8_t c_shift2;

    uint32_t qa;
    uint32_t qb;
    uint32_t qx;
    uint32_t qh;
    uint16_t output_h_scale;
    uint16_t output_h_zp;
    uint16_t output_c_scale;
    uint16_t output_c_zp;
    uint16_t c_scale;
    uint16_t c_zp;
  };

  auto calc_lstm_params = [&](uint32_t lstm_params_rpt[16], int lstm_len_x) {
    double lstm_sx{};
    double lstm_sw{};
    double lstm_sh{};
    double lstm_sr{};
    double lstm_sc{};
    double lstm_sy1{};
    double lstm_sy2{};

    int32_t lstm_zw{};
    int32_t lstm_zr{};
    int32_t lstm_c_zp{};
    int32_t lstm_output_c_zp{};
    int32_t lstm_output_h_zp{};

    uint32_t lstm_len_h = 1024;
    uint32_t lstm_sv_k = 64;
    uint32_t lstm_sv_n = 64;

    if (lstm_len_x == 320) {
      lstm_sx = lstm_in.scale[9];
      lstm_sw = lstm_in.scale[10];
      lstm_sh = lstm_in.scale[4];
      lstm_sr = lstm_in.scale[11];
      lstm_sc = lstm_in.scale[1];
      lstm_sy1 = lstm_in.scale[14]; // h
      lstm_sy2 = lstm_in.scale[15]; // c

      lstm_zw = lstm_in.zp[10];
      lstm_zr = lstm_in.zp[11];
      lstm_c_zp = lstm_in.zp[1];
      lstm_output_c_zp = lstm_in.zp[15];
      lstm_output_h_zp = lstm_in.zp[14];
    } else if (lstm_len_x == 1024) {
      lstm_sx = lstm_in.scale[13];
      lstm_sw = lstm_in.scale[16];
      lstm_sh = lstm_in.scale[5];
      lstm_sr = lstm_in.scale[17];
      lstm_sc = lstm_in.scale[2];
      lstm_sy1 = lstm_in.scale[20]; // h
      lstm_sy2 = lstm_in.scale[21]; // c

      lstm_zw = lstm_in.zp[16];
      lstm_zr = lstm_in.zp[17];
      lstm_c_zp = lstm_in.zp[2];
      lstm_output_c_zp = lstm_in.zp[21];
      lstm_output_h_zp = lstm_in.zp[20];
    }

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

    // init qx_shift qh_shift qa_shift qb_shift -> currently make it fixed, can
    // be smaller
    uint32_t uint32_max_value = (1ll << 32) - 1;
    uint32_t qx_shift =
        MIN(std::floor(std::log2(uint32_max_value / lstm_sx / lstm_sw)), 30);
    uint32_t qh_shift =
        MIN(std::floor(std::log2(uint32_max_value / lstm_sh / lstm_sr)), 30);
    uint32_t qa_shift = MIN(
        std::floor(std::log2(uint32_max_value / lstm_sx / lstm_sw / lstm_zw)),
        30);
    uint32_t qb_shift = MIN(
        std::floor(std::log2(uint32_max_value / lstm_sh / lstm_sr / lstm_zr)),
        30);

    uint32_t lstm_qx = std::round(lstm_sx * lstm_sw * (1 << qx_shift));
    uint32_t lstm_qh = std::round(lstm_sh * lstm_sr * (1 << qh_shift));
    uint32_t lstm_qa =
        std::round(lstm_sx * lstm_sw * lstm_zw * (1 << qa_shift));
    uint32_t lstm_qb =
        std::round(lstm_sh * lstm_sr * lstm_zr * (1 << qb_shift));

    // x/h_bit = 8, len_x/h_bit = log2(len_x/h) -> 9/10, wts_bit = 8,
    // mul_bit = x/h_bit + wts_bit + 10 [log2(1024)] = 26
    // sum_x/h_bit = x/h_bit + len_x/h_bit
    // sum_x/h_bit + qa/qb_bit - 29 <= xs/hs_shift
    // mul_bit + qx/qh_bit - 30 <= xwq/hrq_shift
    // qx_shift - xwq_shift = qa_shift - xs_shift = qh_shift - hrq_shift =
    // qb_shift - hs_shift = nonlinear_in_shift

    // init min shift below
    uint32_t lstm_xs_shift = MAX(8 + std::ceil(std::log2(lstm_len_x)) +
                                     std::ceil(std::log2(lstm_qa)) - 29,
                                 0);
    uint32_t lstm_xwq_shift = MAX(26 + std::ceil(std::log2(lstm_qx)) - 30, 0);
    uint32_t lstm_hs_shift = MAX(8 + std::ceil(std::log2(lstm_len_h)) +
                                     std::ceil(std::log2(lstm_qb)) - 29,
                                 0);
    uint32_t lstm_hrq_shift = MAX(26 + std::ceil(std::log2(lstm_qh)) - 30, 0);

    uint32_t lstm_nonlinear_in_shift =
        MAX(MAX(qx_shift - lstm_xwq_shift, qa_shift - lstm_xs_shift),
            MAX(qh_shift - lstm_hrq_shift, qb_shift - lstm_hs_shift));

    lstm_xs_shift = qa_shift - lstm_nonlinear_in_shift;
    lstm_xwq_shift = qx_shift - lstm_nonlinear_in_shift;
    lstm_hs_shift = qb_shift - lstm_nonlinear_in_shift;
    lstm_hrq_shift = qh_shift - lstm_nonlinear_in_shift;

    uint32_t lstm_c_shift2 = 20;
    uint32_t lstm_nonlinear_out_h_shift = 20;
    uint32_t lstm_nonlinear_out_c_shift = 20;

    uint32_t lstm_output_h_shift =
        std::floor(std::log2(lstm_sy1)) + lstm_nonlinear_out_h_shift + 16;
    uint32_t lstm_output_c_shift =
        std::floor(std::log2(lstm_sy2)) + lstm_nonlinear_out_c_shift + 16;
    uint32_t lstm_output_h_scale = std::round(
        std::pow(2, lstm_output_h_shift - lstm_nonlinear_out_h_shift) /
        lstm_sy1);
    uint32_t lstm_output_c_scale = std::round(
        std::pow(2, lstm_output_c_shift - lstm_nonlinear_out_c_shift) /
        lstm_sy2);
    uint32_t lstm_c_scale = lstm_sc * (1 << lstm_c_shift2);

    struct lstm_params_st* lstm_param = (struct lstm_params_st*)lstm_params_rpt;
    lstm_param->k_iter_cnt_x = lstm_len_x / lstm_sv_k;
    lstm_param->k_iter_cnt_h = lstm_len_h / lstm_sv_k;
    lstm_param->total_iter_cnt =
        MAX(lstm_param->k_iter_cnt_x, lstm_param->k_iter_cnt_h);
    lstm_param->n_iter_cnt = 4;
    lstm_param->xs_shift = lstm_xs_shift;
    lstm_param->hs_shift = lstm_hs_shift;
    lstm_param->xwq_shift = lstm_xwq_shift;
    lstm_param->hrq_shift = lstm_hrq_shift;
    lstm_param->nonlinear_in_shift = lstm_nonlinear_in_shift;
    lstm_param->nonlinear_out_h_shift = lstm_nonlinear_out_h_shift;
    lstm_param->nonlinear_out_c_shift = lstm_nonlinear_out_c_shift;
    lstm_param->output_h_shift = lstm_output_h_shift;
    lstm_param->output_c_shift = lstm_output_c_shift;
    lstm_param->c_shift2 = lstm_c_shift2;
    lstm_param->qa = lstm_qa;
    lstm_param->qb = lstm_qb;
    lstm_param->qx = lstm_qx;
    lstm_param->qh = lstm_qh;
    lstm_param->output_h_scale = lstm_output_h_scale;
    lstm_param->output_c_scale = lstm_output_c_scale;
    lstm_param->output_h_zp = lstm_output_h_zp;
    lstm_param->output_c_zp = lstm_output_c_zp;
    lstm_param->c_scale = lstm_c_scale;
    lstm_param->c_zp = lstm_c_zp;

#undef MIN
#undef MAX
  };

  calc_lstm_params(lstm_in.lstm_320_rtp, 320);
  calc_lstm_params(lstm_in.lstm_1024_rtp, 1024);
}

void VaimlSubgraphProcessor::InitHtLstmWeights(
    const std::unordered_map<std::string, std::string>& initializer_map) const {

  const auto cxx_graph = vaip_cxx::GraphConstRef(graph_);
  auto getRawData = [&](const std::string& name) -> void* {
    return (void*)(cxx_graph.find_node_arg(initializer_map.at(name))
                       ->const_data_as_raw()
                       .data());
  };

  lstm_init_wts lstm_in{};
  lstm_in.scale[9] = /*Sx;*/ *((float*)(getRawData("lstm320_x_s")));
  lstm_in.zp[9] = /*Zx;*/ *((int8_t*)(getRawData("lstm320_x_zp"))) + 128;
  lstm_in.scale[10] = /*Sw;*/ *((float*)(getRawData("lstm320_x_wts_s")));
  lstm_in.zp[10] = /*Zw;*/ *((int8_t*)(getRawData("lstm320_x_wts_zp"))) + 128;
  lstm_in.scale[11] = /*Sr;*/ *((float*)(getRawData("lstm320_h_wts_s")));
  lstm_in.zp[11] = /*Zr;*/ *((int8_t*)(getRawData("lstm320_h_wts_zp"))) + 128;
  lstm_in.scale[12] = /*Sb;*/ *((float*)(getRawData("lstm320_bias_s")));
  lstm_in.zp[12] = /*Zb;*/ *((int8_t*)(getRawData("lstm320_bias_zp"))) + 128;
  lstm_in.scale[4] = /*Sh;*/ *((float*)(getRawData("lstm320_init_h_s")));
  lstm_in.zp[4] = /*Zh;*/ *((int8_t*)(getRawData("lstm320_init_h_zp"))) + 128;
  lstm_in.scale[1] = /*Sc;*/ *((float*)(getRawData("lstm320_init_c_s")));
  lstm_in.zp[1] = /*Zc;*/ *((int8_t*)(getRawData("lstm320_init_c_zp"))) + 128;
  lstm_in.scale[14] = /*Sy1;*/ *((float*)(getRawData("lstm320_output_1_s")));
  lstm_in.zp[14] =
      /*Zy1;*/ *((int8_t*)(getRawData("lstm320_output_1_zp"))) + 128;
  lstm_in.scale[15] = /*Sy2;*/ *((float*)(getRawData("lstm320_output_2_s")));
  lstm_in.zp[15] =
      /*Zy2;*/ *((int8_t*)(getRawData("lstm320_output_2_zp"))) + 128;

  lstm_in.scale[13] = /*Sx;*/ *((float*)(getRawData("lstm1024_x_s")));
  lstm_in.zp[13] = /*Zx;*/ *((int8_t*)(getRawData("lstm1024_x_zp"))) + 128;
  lstm_in.scale[16] = /*Sw;*/ *((float*)(getRawData("lstm1024_x_wts_s")));
  lstm_in.zp[16] = /*Zw;*/ *((int8_t*)(getRawData("lstm1024_x_wts_zp"))) + 128;
  lstm_in.scale[17] = /*Sr;*/ *((float*)(getRawData("lstm1024_h_wts_s")));
  lstm_in.zp[17] = /*Zr;*/ *((int8_t*)(getRawData("lstm1024_h_wts_zp"))) + 128;
  lstm_in.scale[18] = /*Sb;*/ *((float*)(getRawData("lstm1024_bias_s")));
  lstm_in.zp[18] = /*Zb;*/ *((int8_t*)(getRawData("lstm1024_bias_zp"))) + 128;
  lstm_in.scale[5] = /*Sh;*/ *((float*)(getRawData("lstm1024_init_h_s")));
  lstm_in.zp[5] = /*Zh;*/ *((int8_t*)(getRawData("lstm1024_init_h_zp"))) + 128;
  lstm_in.scale[2] = /*Sc;*/ *((float*)(getRawData("lstm1024_init_c_s")));
  lstm_in.zp[2] = /*Zc;*/ *((int8_t*)(getRawData("lstm1024_init_c_zp"))) + 128;
  lstm_in.scale[20] = /*Sy1;*/ *((float*)(getRawData("lstm1024_output_1_s")));
  lstm_in.zp[20] =
      /*Zy1;*/ *((int8_t*)(getRawData("lstm1024_output_1_zp"))) + 128;
  lstm_in.scale[21] = /*Sy2;*/ *((float*)(getRawData("lstm1024_output_2_s")));
  lstm_in.zp[21] =
      /*Zy2;*/ *((int8_t*)(getRawData("lstm1024_output_2_zp"))) + 128;

  InitLstmQdqParams(lstm_in);

  lstm_in.lstm0_h_wts = (int8_t*)(getRawData("lstm320_h_wts"));
  lstm_in.lstm0_x_wts = (int8_t*)(getRawData("lstm320_x_wts"));
  lstm_in.lstm0_bias = (int8_t*)(getRawData("lstm320_bias"));
  lstm_in.lstm1_h_wts = (int8_t*)(getRawData("lstm1024_h_wts"));
  lstm_in.lstm1_x_wts = (int8_t*)(getRawData("lstm1024_x_wts"));
  lstm_in.lstm1_bias = (int8_t*)(getRawData("lstm1024_bias"));

#ifdef VAIP_PASS_VAIML_PARTITION_PROFILING
  auto start1 = std::chrono::steady_clock::now();
#endif

  std::vector<uint8_t> wts = ht_wts_gen_lstm_b2b(lstm_in, cache_dir_);

#ifdef VAIP_PASS_VAIML_PARTITION_PROFILING
  auto end1 = std::chrono::steady_clock::now();
  double time_sec1 = std::chrono::duration<double>(end1 - start1).count();
  VAIML_DEBUG_PRINT("ht_wts_gen_lstm_b2b time (sec): ", time_sec1);
#endif
}

void VaimlSubgraphProcessor::dumpConstants(const Graph& graph) {
  // save constants to a file in the cache directory
  cache_dir_ = self_.get_context()->get_log_dir();
  std::filesystem::path fullCntsFileName = cache_dir_ / constants_file_name_;
  VAIML_DEBUG_PRINT("VaimlSubgraphProcessor::dumpConstants to :",
                    fullCntsFileName);

  std::unique_ptr<std::ostream> cnts_file =
      std::make_unique<std::ostringstream>();
  size_t cnt_offset = 0;
  auto all_constants = VAIP_ORT_API(graph_get_all_initialized_tensors)(graph_);
  auto cxx_graph = vaip_cxx::GraphConstRef(graph_);

  // Dump weights to a binary file
  bool weight_preformated;
  for (const auto& constant : all_constants) {
    weight_preformated = false;
    auto const_name = constant.first;
    ConstantInfo cnt_info;
    cnt_info.offset = cnt_offset;
    auto tensor_proto_ptr = constant.second;
    if (tensor_proto_ptr == nullptr) {
      continue;
    }
    auto& tensor_proto = *tensor_proto_ptr;
    cnt_info.type = VAIP_ORT_API(tensor_proto_data_type)(tensor_proto);
    auto tensor_proto_shape = tensor_proto_get_shape(tensor_proto);
    for (auto s : tensor_proto_shape) {
      cnt_info.shape.push_back((int)s);
    }
    if (cnt_info.shape.empty()) {
      // use [1] to mimic scalar
      cnt_info.shape.push_back(1);
    }
    // for some unknown reason, some shape are <=0.
    for (auto& s : cnt_info.shape) {
      if (s <= 0) {
        s = 1;
      }
    }
    auto cxx_graph = vaip_cxx::GraphConstRef(graph);
    auto raw_values =
        cxx_graph.find_node_arg(constant.first)->const_data_as_raw();
    // For preformateed weights, create an entry in constant map, but do not
    // dump the data to wts.bin
    if (!weight_preformated) {
      cnts_file->write(raw_values.data(), raw_values.size());
      cnt_offset += raw_values.size();
      cnt_info.size = raw_values.size();
    } else {
      cnt_info.size = 0;
    }

    constants_map_[constant.first] = cnt_info;
    // VAIML_DEBUG_PRINT2("DEBUG:: constant ", constant.first, " has ",
    //                    constants_map_[constant.first].size,
    //                    " bytes and saved to offset ",
    //                    constants_map_[constant.first].offset);
  }

  auto charStream = dynamic_cast<std::ostringstream*>(cnts_file.get());
  self_.get_context()->write_file(constants_file_name_, charStream->str());
}

std::vector<std::unique_ptr<IndexedSubGraph>>
VaimlSubgraphProcessor::GetPartitions(const Graph& graph) const {
  VAIML_DEBUG_PRINT("DEBUG: In GetPartitions...");
  std::vector<std::unique_ptr<IndexedSubGraph>> result;
  std::vector<PartitionInfo> node_groups;

  // Graph mode: run VAIML frontend to get all supported nodes in the full
  // graph
  node_groups = GetSupportedNodes(graph);

  auto graph_outputs_vec = graph_get_outputs(graph);
  std::unordered_set<const NodeArg*> graph_outputs(graph_outputs_vec.cbegin(),
                                                   graph_outputs_vec.cend());
  auto graph_inputs_vec = graph_get_inputs(graph);
  std::unordered_set<const NodeArg*> graph_inputs(graph_inputs_vec.cbegin(),
                                                  graph_inputs_vec.cend());

  size_t num_of_supported_nodes = 0;

  for (const auto& partition : node_groups) {
    const std::vector<size_t>& group = partition.partition_nodes;
    if (group.empty())
      continue;
    // Dont modify std::cout for now. This message is needed for lit tests
    VAIML_DEBUG_PRINT("DEBUG: number of nodes in the current group is ",
                      group.size());
    std::unordered_set<size_t> node_set;
    node_set.reserve(group.size());
    for (const auto& index : group) {
      node_set.insert(index);
    }
    // auto sub_graph = onnxruntime::IndexedSubGraph::Create();
    auto sub_graph = std::make_unique<IndexedSubGraph>();
    std::unordered_set<const NodeArg*> node_outputs;
    std::unordered_set<const NodeArg*> subgraph_inputs;
    std::unordered_set<const NodeArg*> subgraph_outputs;
    std::vector<const NodeArg*> ordered_subgraph_inputs;
    std::vector<const NodeArg*> ordered_subgraph_outputs;
    for (const auto& index : group) {
      // sub_graph->Nodes().push_back(index);
      sub_graph->all_nodes.push_back(index);
      // const auto* node = graph.GetNode(index);
      const auto* node = VAIP_ORT_API(graph_get_node)(graph, index);
      // for (const auto* output_def : node->OutputDefs()) {
      for (const auto* output_def : node_get_output_node_args(*node)) {
        // if (!output_def->Exists()) {
        if (!node_arg_exists(*output_def)) {
          continue;
        }
        node_outputs.insert(output_def);
        // if output is overall graph output we need to produce it.
        if (graph_outputs.count(output_def) != 0) {
          ordered_subgraph_outputs.push_back(output_def);
        }
      }
    }
    for (const auto& index : group) {
      // const auto* node = graph_viewer.GetNode(index);
      const auto* node = VAIP_ORT_API(graph_get_node)(graph, index);
      // for (const auto* input : node->InputDefs()) {
      for (const auto* input : node_get_input_node_args(*node)) {
        // if the node input was not produced by this subgraph, add it to the
        // subgraph inputs.
        // if (!input->Exists()) {
        if (!node_arg_exists(*input)) {
          continue;
        }
        if (node_outputs.count(input) == 0) {
          // std::string inputArgName = input->Name();
          std::string inputArgName = node_arg_get_name(*input);
          // const Node* producer = graph_viewer.GetProducerNode(input->Name());
          const Node* producer =
              VAIP_ORT_API(graph_producer_node)(graph, inputArgName);

          if (subgraph_inputs.count(input)) {
            continue; // already added
          }

          if ((graph_inputs.count(input) != 0) || (producer != nullptr)) {
            subgraph_inputs.insert(input);
            ordered_subgraph_inputs.push_back(input);

            // VAIML_DEBUG_PRINT uses stdout, so the messages below can be used
            // for lit filecheck
            VAIML_DEBUG_PRINT("DEBUG: subgraph input ",
                              ordered_subgraph_inputs.size(),
                              " name is: ", inputArgName,
                              " type is: ", node_arg_get_element_type(*input));
          }
        }
      }
      for (auto output_def : node_get_output_node_args(*node)) {
        auto consumers =
            graph_get_consumer_nodes(graph, node_arg_get_name(*output_def));

        bool is_consumer_outside_subgraph = false;
        for (auto c : consumers) {
          if (node_set.count(VAIP_ORT_API(node_get_index)(*c)))
            continue; // Consumed within this subgraph
          is_consumer_outside_subgraph = true;
        }

        // Declare this NodeArg as subgraph output if it is consumed by an
        // external node or when it is an overall graph output.
        if (is_consumer_outside_subgraph || graph_outputs.count(output_def)) {
          // VAIML_DEBUG_PRINT uses stdout, so the messages below can be used
          // for lit filecheck
          VAIML_DEBUG_PRINT(
              "DEBUG: subgraph output ", ordered_subgraph_outputs.size(),
              " name is: ", node_arg_get_name(*output_def),
              " type is: ", node_arg_get_element_type(*output_def));
          subgraph_outputs.insert(output_def);
          ordered_subgraph_outputs.push_back(output_def);
        }
      }
    }
    // auto cur_subgraph_size = sub_graph->Nodes().size();
    auto cur_subgraph_size = sub_graph->all_nodes.size();
    num_of_supported_nodes += cur_subgraph_size;
    // generate current partition info
    if (ordered_subgraph_inputs.empty()) {
      // std::cout << "INFO: subgraph has no inputs (constant foldable), drop
      // it."
      //           << std::endl;
      if (model_name_ == "GT_v1.2") {
        // allow customized constand folding
      } else {
        continue;
      }
    }
    if (max_num_inputs_ > 0 &&
        (int)(ordered_subgraph_inputs.size()) > max_num_inputs_) {
      MY_LOG(2) << "INFO: subgraph has " << ordered_subgraph_inputs.size()
                << " inputs, drop it.";
      continue;
    }
    if (max_num_outputs_ > 0 &&
        (int)(ordered_subgraph_outputs.size()) > max_num_outputs_) {
      MY_LOG(2) << "INFO: subgraph has " << ordered_subgraph_outputs.size()
                << " outputs, drop it.";
      continue;
    }

    auto compareName = [](const NodeArg* a, const NodeArg* b) {
      return node_arg_get_name(*a) < node_arg_get_name(*b);
    };
    // try_fuse requires the inputs to be sorted (see their use of
    // std::includes). We also sort the outputs for consistency.
    std::sort(ordered_subgraph_inputs.begin(), ordered_subgraph_inputs.end(),
              compareName);
    std::sort(ordered_subgraph_outputs.begin(), ordered_subgraph_outputs.end(),
              compareName);
    for (const auto& input : ordered_subgraph_inputs) {
      // meta_def->inputs().push_back(input->Name());
      sub_graph->all_inputs.push_back(node_arg_get_name(*input));
      auto shapePtr = node_arg_get_shape_i64(*input);
      VAIML_DEBUG_PRINT("ordered_subgraph_inputs")
      // shape can be dynamic, which means not shape information
      if (shapePtr != nullptr) {
        sub_graph->input_shapes.push_back(*(shapePtr.get()));
      }
    }
    for (const auto& output : ordered_subgraph_outputs) {
      // meta_def->outputs().push_back(output->Name());
      sub_graph->all_outputs.push_back(node_arg_get_name(*output));
      auto shapePtr = node_arg_get_shape_i64(*output);
      VAIML_DEBUG_PRINT("ordered_subgraph_outputs")
      // shape can be dynamic, so the shape pointer can be nullptr
      if (shapePtr != nullptr) {
        sub_graph->output_shapes.push_back(*(shapePtr.get()));
      }
    }
    // sub_graph->SetMetaDef(std::move(meta_def));
    // result.push_back(ComputeCapability::Create(std::move(sub_graph)));
    sub_graph->name = partition.partition_name;
    sub_graph->name_id = result.size();

    MY_LOG(2) << "VaimlSubgraphProcessor::Partition, " << sub_graph->name
              << " current supported subgraph size: " << cur_subgraph_size;

    // We try fusing here because the partition algorithm sometimes produces
    // partitions where its inputs depend on its outputs e.g. the AIE subgraph
    // in the diamond pattern
    //        / -> AIE `
    // -> AIE -         AIE ->
    //         \ -> CPU /
    // try_fuse checks for such loops, so we can reject them.
    auto [meta_def, fuse_error] =
        self_.try_fuse(graph_, sub_graph->name, sub_graph->all_inputs,
                       sub_graph->all_outputs, {}, "VAIML");
    if (!meta_def) {
      if (fuse_error.comments == "loop detected") {
        // Dont modify std::cout for now. This message is needed for lit tests
        // std::cout << "INFO: subgraph has a loop, drop it (see CR-1201729)."
        //          << std::endl;
      } else {
        MY_LOG(2) << "WARNING: subgraph cannot be fused: "
                  << fuse_error.comments;
      }
      continue;
    }
    result.push_back(std::move(sub_graph));
  }
  // Sort graphs by largest first.
  std::sort(result.begin(), result.end(),
            [](const std::unique_ptr<IndexedSubGraph>& a,
               const std::unique_ptr<IndexedSubGraph>& b) {
              return a->all_nodes.size() == b->all_nodes.size()
                         ? a->name_id < b->name_id
                         : (a->all_nodes.size() > b->all_nodes.size());
            });
  MY_LOG(2) << "sorted subgraph:";
  for (auto& res : result) {
    MY_LOG(2) << "subgraph name " << res->name
              << " node size: " << res->all_nodes.size();
  }
  int supported_size = (int)result.size();
  if (max_num_partitions_ > 0) {
    // Erase the smallest partitions.
    result.erase(result.begin() + std::min(supported_size, max_num_partitions_),
                 result.end());
  }
  float percent_flexml =
      100.0f * (static_cast<float>(num_of_supported_nodes) /
                static_cast<float>(graph_nodes(graph).size()));
  // Using std::cout for summary report.
  // std::cout << "------------ VAIML Passes Summary ---------------" <<
  // std::endl; std::cout
  //    << "    Number of nodes in the graph: " << graph_nodes(graph).size()
  //    << "\n    Number of nodes supported by VAIML: " <<
  //    num_of_supported_nodes
  //    << "\n    Percentage of supported nodes by VAIML: " << std::fixed
  //    << std::setprecision(2) << " (" << percent_flexml << "%)" << std::endl
  //    << "\n    Number of partitions supported by VAIML: " << supported_size
  //    << "\n    Number of partitions deployed to device: " << result.size();

  // only print out stats of each partition in graph mode
  for (auto i = 0; i < (int)(result.size()); ++i) {
    auto num_nodes = result[i]->all_nodes.size();
    // std::cout << "\n    Number of nodes in partition " << result[i]->name
    //           << ": " << num_nodes;
    float percent_partition =
        100.0f * (static_cast<float>(num_nodes) /
                  static_cast<float>(graph_nodes(graph).size()));
    // std::cout << "\n    Percentage of supported nodes in partition " << i
    //           << ": " << std::fixed << std::setprecision(2) <<
    //           percent_partition
    //           << "%";
  }
  // fuse the final partition results

  // std::cout << std::endl;
  return result;
}

void VaimlSubgraphProcessor::process() const {
  auto start = std::chrono::steady_clock::now();
  std::vector<std::unique_ptr<IndexedSubGraph>> result;
  bool is_model_name_matched = false;
  if (model_name_ == "GT_v1.2" || model_name_ == "HT_v1.2" ||
      model_name_ == "GT_v1.3" || model_name_ == "GTC_v1.0") {
    is_model_name_matched = true;
  } else {
    std::cout << "WARNING: not supported model\n";
  }

  if (model_name_ == "GT_v1.3" || model_name_ == "GTC_v1.0") {
    auto initializer_map = vaip_core::MetaDefProto();
    GT_initializer_mapping_pass pass(self_, initializer_map);
    pass.process(graph_);
    std::string buf;
    initializer_map.SerializeToString(&buf);
    self_.get_context()->write_file(
        "gt_init_map.proto.bin", gsl::span<const char>(buf.data(), buf.size()));
    msig_ops_map_.insert(
        std::make_pair(model_name_, pass.get_partitioned_nodes()));
  } else if (model_name_ == "HT_v1.2") {
    auto initializer_map = vaip_core::MetaDefProto();
    HT_initializer_mapping_pass pass(self_, initializer_map);
    pass.process(graph_);
    std::unordered_map<std::string, std::string> initializer_unordered_map(
        initializer_map.generic_param().begin(),
        initializer_map.generic_param().end());
    InitHtLstmWeights(initializer_unordered_map);

    std::string buf;
    initializer_map.SerializeToString(&buf);
    self_.get_context()->write_file(
        "ht_init_map.proto.bin", gsl::span<const char>(buf.data(), buf.size()));
    msig_ops_map_.insert(
        std::make_pair("HT_v1.2", pass.get_partitioned_nodes()));
  }
  if (is_model_name_matched) {
    result = GetPartitions(graph_);
    for (auto& res : result) {
      fuseGraph(*res);
    }
  }

  auto end = std::chrono::steady_clock::now();
  double time_sec = std::chrono::duration<double>(end - start).count();
  VAIML_DEBUG_PRINT2("VaimlSubgraphProcessor::process time (sec): ", time_sec);
}

/**
 * @brief Fuse a subgraph
 * This function is common to graph mode and transformer mode
 */
const Node* VaimlSubgraphProcessor::fuseGraph(IndexedSubGraph& subgraph) const {
  MY_LOG(2) << "DEBUG: Running fuseGraph...";
  MY_LOG(2) << "DEBUG: subgraph all_inputs size: "
            << subgraph.all_inputs.size();
  MY_LOG(2) << "DEBUG: subgraph input_shapes size: "
            << subgraph.input_shapes.size();
  auto [meta_def, fuse_error] =
      self_.try_fuse(graph_, subgraph.name, subgraph.all_inputs,
                     subgraph.all_outputs, {}, "VAIML");

  if (meta_def == nullptr) {
    LOG(ERROR) << "Failed to fuse subgraph " << subgraph.name << ": "
               << fuse_error.comments << "\n";
    return nullptr;
  }

  auto vaiml_param = meta_def->mutable_vaiml_param();
  vaiml_param->set_vaiml_model_path(subgraph.name);
  vaiml_param->set_device_name(device_name_);
  auto constant_initializers = meta_def->constant_initializers();
  for (auto const_name : constant_initializers) {
    auto const_info = constants_map_.at(const_name);
    auto const_data = ConstInfo();
    const_data.set_offset(const_info.offset);
    const_data.set_size(const_info.size);
    const_data.mutable_shape()->Assign(const_info.shape.begin(),
                                       const_info.shape.end());
    const_data.set_type(const_info.type);
    vaiml_param->mutable_const_data_info()->insert(
        google::protobuf::MapPair{const_name, const_data});
  }

  MY_LOG(1) << "vaiml model path:"
            << meta_def->vaiml_param().vaiml_model_path();
  auto* ret = &self_.fuse(graph_, std::move(*meta_def));
  for (auto& shape : subgraph.input_shapes) {
    (*(vaiml_param->add_input_shapes()->mutable_shapes())) = {shape.begin(),
                                                              shape.end()};
  }
  auto node_inputs = node_get_input_node_args(*ret);
  for (size_t i = 0; i < node_inputs.size(); ++i) {
    auto node_name = node_arg_get_name(*node_inputs[i]);
    VAIML_DEBUG_PRINT("Input node name dump:", node_name);
  }

  auto outputSize = node_get_output_node_args(*ret).size();
  auto node_outputs = node_get_output_node_args(*ret);
  for (size_t i = 0; i < outputSize; i++) {
    VAIML_DEBUG_PRINT("    Adding node output ", i);
    auto shapePtr = node_arg_get_shape_i64(*node_outputs[i]);
    auto node_name = node_arg_get_name(*node_outputs[i]);
    VAIML_DEBUG_PRINT("Output node name dump:", node_name);

    if (shapePtr != nullptr) {
      auto shape = *(shapePtr.get());

      VAIML_DEBUG_PRINT("output shape dump: ");
      std::string str = "==>";
      // for (auto item : shape) {
      for (size_t i = 0; i < shape.size(); ++i) {
        shape[i] = (shape[i] < 0) ? 1 : shape[i];
        str += " ";
        str += std::to_string(shape[i]);
      }
      VAIML_DEBUG_PRINT(str);
      (*(vaiml_param->add_output_shapes()->mutable_shapes())) = {shape.begin(),
                                                                 shape.end()};
    }
  }

  VAIML_DEBUG_PRINT("finish fuseGraph");
  return ret;
}

} // namespace vaip_vaiml_subgraph_processor
