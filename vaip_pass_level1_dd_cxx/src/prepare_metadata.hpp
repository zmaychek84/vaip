/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

#pragma once

#include "vaip/vaip.hpp"
#include <map>
#include <op_fuser/fusion_rt.hpp>
#include <string>
#include <vector>

namespace dd {

struct AuxTensorInfo {
  std::string dtype;
  std::vector<int64_t> shape;
  int32_t size_in_bytes;
  int32_t offset;
};

struct NewTensorInfo {
  int32_t buffer_size;
  int xrt_arg_id;
  std::vector<std::string> packed_tensors;
};

using NewTensors = std::map<std::string, NewTensorInfo>;

struct NewTensorMapItem {
  std::string packed_buffer_label;
  int32_t xrt_arg_id;
  AuxTensorInfo aux_info;
  std::string file_name;
  int32_t file_size;
};

using NewTensorInfoMap = std::map<std::string, NewTensorMapItem>;

using OPAttrs =
    std::map<std::string, std::pair<std::string, std::vector<std::string>>>;

struct OPInfo {
  std::string name;
  std::string type;
  std::vector<std::string> in_args;
  std::vector<std::string> const_args;
  std::vector<std::string> out_args;
  OPAttrs attrs;
};

struct ConstData {
  std::string name;      // Original name as per the model
  std::string file_name; // Translated filename
  std::vector<uint8_t> data;
};

// using ConstDB = std::map<std::string, ConstData>;
using ConstDB = std::map<std::string, std::vector<char>>;
using LeanConstDB = std::map<std::string, OpsFusion::SimpleSpan>;

std::tuple<std::vector<OPInfo>, NewTensors, NewTensorInfoMap, LeanConstDB>
graph_prepare_metadata(const vaip_cxx::Subgraph& graph,
                       const std::filesystem::path& dir_path);

std::string save_tensors_to_json(const std::vector<OPInfo>& op_list,
                                 const NewTensors& new_tensors,
                                 const NewTensorInfoMap& new_tensors_map);

} // namespace dd
