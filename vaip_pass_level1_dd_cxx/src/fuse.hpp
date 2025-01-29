/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

#pragma once
#include "dd.hpp"
#include "nlohmann/json.hpp"
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#include <vaip/vaip.hpp>

#include "vaip/graph.hpp"
#include "vaip/node.hpp"
#include "vaip/node_arg.hpp"
// #include "onnxruntime_vitisai_ep.hpp"
// #include "partitioner.hpp"
#include "prepare_metadata.hpp"
#include "vaip/vaip_ort.hpp"

namespace dd {
// using node_ind_t = int32_t;
// using node_list = std::vector<node_ind_t>;
// using node_set = std::set<node_ind_t>;
// using Graph = std::map<node_ind_t, node_list>;
// using label_map = std::map<node_ind_t, node_ind_t>;
// using property_map = std::map<node_ind_t, std::string>;

std::tuple<label_map, Graph, property_map>
partition_onnx_model(const std::filesystem::path& model,
                     const property_map& idx_node_map);
nlohmann::json prepare_metadef_context_json_from_subgraph(
    const std::string& model, const std::filesystem::path& meta_json_path,
    node_ind_t subgraph_label, const node_list, const std::string& model_);

std::tuple<dd::label_map, dd::Graph, std::map<int, std::string>,
           std::map<dd::node_ind_t, std::string>>
partition_onnx_model(onnxruntime::Graph& graph);

std::tuple<std::vector<float>, std::vector<float>, std::vector<int>>
prepare_metadef_context_json_from_subgraph2(
    const onnxruntime::Graph& graph, const node_list& subgraph_node_ids,
    const std::map<int, std::string>& idx_node_map,
    const std::filesystem::path& meta_json_path, node_ind_t subgraph_label);

void prepare_metadef_context_json_from_subgraph3(
    const onnxruntime::Graph& graph, vaip_core::MetaDefProto* metadef,
    const std::string& model_category, std::string qos_priority = "",
    std::string is_preemptible = "false");

using ret_type = std::pair<std::vector<std::string>, std::vector<std::string>>;
ret_type get_subgraph_input_outputs(
    const onnxruntime::Graph& graph, const node_list& subgraph_node_ids,
    const std::map<int, std::string>& idx_node_map,
    const std::filesystem::path& meta_json_path, node_ind_t subgraph_label);

std::vector<std::vector<const onnxruntime::Node*>>
graph_get_input_nodes(const onnxruntime::Graph& graph);

} // namespace dd
