/*
 *     The Xilinx Vitis AI Vaip in this distribution are provided under the
 * following free and permissive binary-only license, but are not provided in
 * source code form.  While the following free and permissive license is similar
 * to the BSD open source license, it is NOT the BSD open source license nor
 * other OSI-approved open source license.
 *
 *      Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights
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

#include "./fuse.hpp"
#include "nlohmann/json.hpp"
#include <fstream>
#include <glog/logging.h>
#include <google/protobuf/util/json_util.h>
#include <unordered_set>

using namespace std::literals;

using namespace vaip_core;

// Utility to join elements of a vector to string
// Eg : {1, 2, 3, 4} --> "1,2,3,4"
template <typename T>
std::string join(const std::vector<T>& vec, const std::string& sep = ",") {
  std::ostringstream oss;
  for (auto iter = vec.begin(); iter != vec.end(); iter++) {
    oss << *iter;
    if (iter != std::prev(vec.end())) {
      oss << sep;
    }
  }
  return oss.str();
}

// Utilty to join vector of vector
// Inner vector is joined with space
// Outer vector is joined with comma
// Eg : {{1,2}, {}, {3, 4, 5}} --> "1 2,,3 4 5"
template <typename T>
std::string join(const std::vector<std::vector<T>>& vec,
                 const std::string& sep = ",") {
  std::ostringstream oss;
  for (auto iter = vec.begin(); iter != vec.end(); iter++) {
    oss << join(*iter, " ");
    if (iter != std::prev(vec.end())) {
      oss << sep;
    }
  }
  return oss.str();
}

static std::vector<std::string> get_strings_attrs(const onnxruntime::Node& node,
                                                  const std::string& name) {
  auto& attrs = node_get_attributes_ref(node);
  // auto attr_keys = VAIP_ORT_API(node_attributes_get_keys)(
  //     const_cast<NodeAttributes&>(attrs));
  auto attr_proto = node_attributes_get(attrs, name);
  auto strs_value = VAIP_ORT_API(attr_proto_get_strings)(*attr_proto);
  return strs_value;
}

std::tuple<dd::label_map, dd::Graph, dd::property_map>
dd::partition_onnx_model(const std::filesystem::path& model,
                         const dd::property_map& idx_node_map) {
  auto filename = model;
  filename.replace_extension(std::filesystem::path(".onnx.partion.json"));
  std::ifstream f(filename);
  CHECK(f.is_open()) << "failed to open config file: " << filename;
  auto json = nlohmann::json::parse(f);
  dd::label_map label;
  dd::Graph g;
  dd::property_map p;
  auto node_subgraph_labels = json["node_subgraph_labels"];
  for (auto it = node_subgraph_labels.begin(), end = node_subgraph_labels.end();
       it != end; ++it) {
    label[std::stoi(it.key())] = it.value();
  }
  auto subgraph_node_cluster = json["subgraph_node_cluster"];
  for (auto it = subgraph_node_cluster.begin(),
            end = subgraph_node_cluster.end();
       it != end; ++it) {
    auto& v = g[std::stoi(it.key())];
    v.clear();
    for (auto jt = it.value().begin(), jend = it.value().end(); jt != jend;
         ++jt) {
      v.push_back(*jt);
    }
  }
  auto target_label = json["target_label"];
  for (auto it = target_label.begin(), end = target_label.end(); it != end;
       ++it) {
    p[std::stoi(it.key())] = it.value();
  }

  return std::make_tuple(label, g, p);
}

nlohmann::json dd::prepare_metadef_context_json_from_subgraph(
    const std::string& model, const std::filesystem::path& meta_json_path,
    node_ind_t subgraph_label, node_list, const std::string& model_) {
  auto ret = nlohmann::json();
  auto filename = meta_json_path;
  filename.replace_extension(std::filesystem::path(".json.metadef"));
  // LOG(DEBUG) << " dd::prepare_metadef_context_json_from_subgraph open file: "
  //            << filename;
  std::ifstream f(filename);
  CHECK(f.is_open()) << "failed to open file: " << filename;
  ret = nlohmann::json::parse(f);
  return ret["metadef_data"];
}

template <typename T>
static std::vector<T> stable_unique(const std::vector<T>& src) {
  std::unordered_set<T> visited;
  std::vector<T> res;
  for (const auto& item : src) {
    if (visited.find(item) == visited.end()) {
      visited.insert(item);
      res.push_back(item);
    }
  }
  return res;
}

static std::vector<std::string> extract_subgraph_inputs(
    const onnxruntime::Graph& graph,
    const std::unordered_set<const Node*>& subgraph_nodes,
    const std::unordered_set<std::string>& subgraph_node_names) {
  // Get graph input tensors
  auto graph_input_args = graph_get_inputs(graph);
  std::unordered_set<std::string> graph_input_tensors;
  for (const auto* graph_input_arg : graph_input_args) {
    graph_input_tensors.insert(node_arg_get_name(*graph_input_arg));
  }

  // Get subgraph input tensors
  std::vector<std::string> subgraph_input_tensors;
  for (const auto* node : subgraph_nodes) {
    for (const auto* input_arg : node_get_input_node_args(*node)) {
      // Skip constant inputs
      if (node_arg_is_constant(graph, *input_arg)) {
        continue;
      }

      // If it is graph input, add it to subgraph inputs
      auto input_arg_name = node_arg_get_name(*input_arg);
      if (graph_input_tensors.find(input_arg_name) !=
          graph_input_tensors.end()) {
        subgraph_input_tensors.push_back(input_arg_name);
        continue;
      }

      // If the producer of this input lies outside the subgraph, it is an input
      // Short cut : Node's name is same as its output.
      auto producer = VAIP_ORT_API(graph_producer_node)(graph, input_arg_name);
      const auto& producer_name = VAIP_ORT_API(node_get_name)(*producer);
      if (subgraph_node_names.find(producer_name) ==
          subgraph_node_names.end()) {
        subgraph_input_tensors.push_back(input_arg_name);
      }
    }
  }

  auto unique_subgraph_input_tensors = stable_unique(subgraph_input_tensors);

  // for (const auto& name : unique_subgraph_input_tensors) {
  // LOG(DEBUG) << "Subgraph Input : " << name;
  // }

  return unique_subgraph_input_tensors;
}

static std::vector<std::string> extract_subgraph_outputs(
    const onnxruntime::Graph& graph,
    const std::unordered_set<const Node*>& subgraph_nodes,
    const std::unordered_set<std::string>& subgraph_node_names) {
  // Get graph input tensors
  auto graph_output_args = graph_get_outputs(graph);
  std::unordered_set<std::string> graph_output_tensors;
  for (const auto* graph_output_arg : graph_output_args) {
    graph_output_tensors.insert(node_arg_get_name(*graph_output_arg));
  }

  // Get subgraph input tensors
  std::vector<std::string> subgraph_output_tensors;
  for (const auto* node : subgraph_nodes) {
    for (const auto* output_arg : node_get_output_node_args(*node)) {

      // If it is graph output, add it to subgraph outputs
      auto output_arg_name = node_arg_get_name(*output_arg);
      if (graph_output_tensors.find(output_arg_name) !=
          graph_output_tensors.end()) {
        subgraph_output_tensors.push_back(output_arg_name);
        continue;
      }

      // If any of consumer of this input lies outside the subgraph, it is an
      // output Short cut : Node's name is same as its output.
      auto consumers = graph_get_consumer_nodes(graph, output_arg_name);
      for (const auto* consumer : consumers) {
        const auto& consumer_name = VAIP_ORT_API(node_get_name)(*consumer);
        if (subgraph_node_names.find(consumer_name) ==
            subgraph_node_names.end()) {
          subgraph_output_tensors.push_back(output_arg_name);
          break;
        }
      }
    }
  }

  auto unique_subgraph_output_tensors = stable_unique(subgraph_output_tensors);

  // for (const auto& name : unique_subgraph_output_tensors) {
  // LOG(DEBUG) << "Subgraph output : " << name;
  // }

  return unique_subgraph_output_tensors;
}

std::vector<std::vector<const onnxruntime::Node*>>
dd::graph_get_input_nodes(const onnxruntime::Graph& graph) {
  auto graph_inputs = graph_get_inputs(graph);
  // std::vector<std::vector<const Node*>> input_nodes;
  auto input_nodes = std::vector<std::vector<const Node*>>();
  input_nodes.reserve(graph_inputs.size());
  for (auto& i : graph_inputs) {
    if (i) {
      // auto consumer_nodes = VAIP_ORT_API(graph_get_consumer_nodes_unsafe)(
      //     graph, node_arg_get_name(*i));
      auto consumers_nodes =
          graph_get_consumer_nodes(graph, node_arg_get_name(*i));
      // if (consumer_nodes) {
      input_nodes.push_back(consumers_nodes);
      // }
    }
  }
  return input_nodes;
}

dd::ret_type dd::get_subgraph_input_outputs(
    const onnxruntime::Graph& graph, const node_list& subgraph_node_ids,
    const std::map<int, std::string>& idx_node_map,
    const std::filesystem::path& meta_json_path, node_ind_t subgraph_label) {

  // Get all nodes & node names in the subgraph
  std::unordered_set<const Node*> subgraph_nodes;
  std::unordered_set<std::string> subgraph_node_names;
  for (auto node_id : subgraph_node_ids) {
    auto node = VAIP_ORT_API(graph_get_node)(graph, node_id);
    std::string node_name = VAIP_ORT_API(node_get_name)(*node);
    subgraph_node_names.insert(node_name);
    subgraph_nodes.insert(node);
  }

  auto subgraph_input_tensors =
      extract_subgraph_inputs(graph, subgraph_nodes, subgraph_node_names);
  auto subgraph_output_tensors =
      extract_subgraph_outputs(graph, subgraph_nodes, subgraph_node_names);

  return std::make_pair(subgraph_input_tensors, subgraph_output_tensors);
}

// save_tensors_to_json
std::tuple<std::vector<float>, std::vector<float>, std::vector<int>>
dd::prepare_metadef_context_json_from_subgraph2(
    const onnxruntime::Graph& subgraph, const node_list& subgraph_node_ids,
    const std::map<int, std::string>& idx_node_map,
    const std::filesystem::path& meta_json_path, node_ind_t subgraph_label) {

  nlohmann::json meta_def_output_names;

  // std::vector<float> input_q_params;
  std::vector<float> output_q_params_;
  std::vector<int> original_output_shapes;

  auto nodes_in_subgraph = graph_get_node_in_topoligical_order(subgraph);

  auto subgraph_input_tensors = graph_get_inputs(subgraph);
  auto subgraph_output_tensors = graph_get_outputs(subgraph);

  // using Tuple = std::tuple<std::vector<float>, std::vector<float>>;

  // Extract Input Q Params
  std::vector<float> input_q_params;

  for (auto subgraph_input : subgraph_input_tensors) {
    // LOG(DEBUG) << "input arg name : " << node_arg_get_name(*subgraph_input);
    auto consumers =
        graph_get_consumer_nodes(subgraph, node_arg_get_name(*subgraph_input));

    if (consumers.empty()) {
      continue;
    }
    std::vector<float> current_in_qparams;
    std::vector<int> input_idx;
    for (auto consumer : consumers) {
      auto consumer_inputs = node_get_input_node_args(*consumer);
      for (std::size_t idx = 0; idx < consumer_inputs.size(); ++idx) {
        if (node_arg_get_name(*consumer_inputs[idx]) ==
            node_arg_get_name(*subgraph_input)) {
          input_idx.push_back((int32_t)idx);
        }
      }

      bool node_has_input_q = node_has_attr(*consumer, "input_q_params");
      if (node_has_input_q) {
        auto span_input_q_params =
            node_get_attr_floats(*consumer, "input_q_params");

        if (!span_input_q_params.empty()) {
          std::vector<float> scale(input_idx.size());
          std::vector<float> zp(input_idx.size());

          std::transform(input_idx.begin(), input_idx.end(), scale.begin(),
                         [&span_input_q_params](int i) {
                           int paramIdx = 2 * i;
                           if (paramIdx < (int32_t)span_input_q_params.size()) {
                             return span_input_q_params[paramIdx];
                           } else
                             return 0.0f;
                         });

          std::transform(input_idx.begin(), input_idx.end(), zp.begin(),
                         [&span_input_q_params](int i) {
                           int paramIdx = 2 * i + 1;
                           if (paramIdx < (int32_t)span_input_q_params.size()) {
                             return span_input_q_params[paramIdx];
                           } else
                             return 0.0f;
                         });
          // current_in_qparams = current_in_qparamsstd::make_tuple(scale, zp);
          current_in_qparams.insert(current_in_qparams.end(), scale.begin(),
                                    scale.end());

          current_in_qparams.insert(current_in_qparams.end(), zp.begin(),
                                    zp.end());
        }
        break;
      }

      // input_q_params.push_back(current_in_qparams);
      input_q_params.insert(input_q_params.end(), current_in_qparams.begin(),
                            current_in_qparams.end());
    }
  }

  // out_q_params
  std::vector<float> output_q_params;
  std::vector<int> meta_def_output_shapes;

  for (auto subgraph_out : subgraph_output_tensors) {
    auto producer = VAIP_ORT_API(graph_producer_node)(
        subgraph, node_arg_get_name(*subgraph_out));

    if (!producer) {
      continue;
    }

    std::vector<int> output_idx;
    auto producer_output = node_get_output_node_args(*producer);
    for (std::size_t idx = 0; idx < producer_output.size(); ++idx) {
      if (node_arg_get_name(*producer_output[idx]) ==
          node_arg_get_name(*subgraph_out)) {
        output_idx.push_back((int32_t)idx);
      }
    }

    std::vector<float> node_outq_params;
    if (node_has_attr(*producer, "output_q_params")) {
      auto span_output_q_params =
          node_get_attr_floats(*producer, "output_q_params");

      if (!span_output_q_params.empty()) {
        std::vector<float> scale(output_idx.size());
        std::vector<float> zp(output_idx.size());

        std::transform(output_idx.begin(), output_idx.end(), scale.begin(),
                       [&span_output_q_params](int i) {
                         int paramIdx = 2 * i;
                         if (paramIdx < (int32_t)span_output_q_params.size()) {
                           return span_output_q_params[paramIdx];
                         } else
                           return 0.0f;
                       });

        std::transform(output_idx.begin(), output_idx.end(), zp.begin(),
                       [&span_output_q_params](int i) {
                         int paramIdx = 2 * i + 1;
                         if (paramIdx < (int32_t)span_output_q_params.size()) {
                           return span_output_q_params[paramIdx];
                         } else
                           return 0.0f;
                       });
        // node_outq_params = std::make_tuple(scale, zp);
        node_outq_params.insert(node_outq_params.end(), scale.begin(),
                                scale.end());
        node_outq_params.insert(node_outq_params.end(), zp.begin(), zp.end());
      }
    }

    if (node_has_attr(*producer, "orig_output_shape")) {
      auto span_original_output_shapes =
          node_get_attr_ints(*producer, "orig_output_shape");
      original_output_shapes.reserve(
          span_original_output_shapes.size()); // Reserve space for efficiency
      for (const auto& elem : span_original_output_shapes) {
        original_output_shapes.push_back(static_cast<int>(elem));
      }
    }

    // output_q_params.push_back(node_outq_params);
    output_q_params.insert(output_q_params.end(), node_outq_params.begin(),
                           node_outq_params.end());
    // meta_def_output_shapes.push_back(original_output_shapes);

    meta_def_output_shapes.insert(meta_def_output_shapes.end(),
                                  original_output_shapes.begin(),
                                  original_output_shapes.end());
  }

  // LOG(DEBUG) << "# Nodes in subgraph : " << nodes_in_subgraph.size();
  // for (auto node_idx : nodes_in_subgraph) {

  //   nlohmann::json output_names;
  //   auto subg_node = VAIP_ORT_API(graph_get_node)(subgraph, node_idx);
  //   auto input_args = node_get_input_node_args(*subg_node);
  //   auto output_args = node_get_output_node_args(*subg_node);
  //   auto node_attrs = node_get_attributes(*subg_node);
  // }

  return std::make_tuple(output_q_params, output_q_params,
                         meta_def_output_shapes);
}

// save_tensors_to_json
void dd::prepare_metadef_context_json_from_subgraph3(
    const onnxruntime::Graph& graph, vaip_core::MetaDefProto* metadef,
    const std::string& model_category, std::string qos_priority) {

  nlohmann::json meta_def_output_names;

  std::vector<std::string> subgraph_input_tensors(metadef->inputs().begin(),
                                                  metadef->inputs().end());
  std::vector<std::string> subgraph_output_tensors(metadef->outputs().begin(),
                                                   metadef->outputs().end());

  // Extract Input Q Params
  // Stored as { {T1.zp, T1.scale}, {T2.zp, T2.scale}, ...}
  std::vector<std::vector<float>> input_q_params;

  for (auto subgraph_input : subgraph_input_tensors) {
    auto consumers = graph_get_consumer_nodes(graph, subgraph_input);

    if (consumers.empty()) {
      throw std::runtime_error("No consumer found for tensor: "s +
                               subgraph_input);
    }

    std::vector<float> current_in_qparams;
    for (auto consumer : consumers) {
      bool node_has_input_q = node_has_attr(*consumer, "input_q_params");
      if (!node_has_input_q) {
        continue;
      }

      // Find index of tensor
      size_t input_idx = -1;
      auto consumer_inputs = node_get_input_node_args(*consumer);
      for (std::size_t idx = 0; idx < consumer_inputs.size(); ++idx) {
        if (node_arg_get_name(*consumer_inputs[idx]) == subgraph_input) {
          input_idx = idx;
          break;
        }
      }
      if ((int32_t)input_idx == -1) {
        throw std::runtime_error(
            "Couldn't find tensor in list of inputs of consumer");
      }

      // Extract input q params
      size_t paramIdx = input_idx * 2;
      auto span_input_q_params =
          node_get_attr_floats(*consumer, "input_q_params");
      if (span_input_q_params.size() < paramIdx + 2) {
        throw std::runtime_error(
            "Insufficient input_q_params for node [Todo : getnodename]");
      }
      float scale = span_input_q_params[paramIdx];
      float zp = span_input_q_params[paramIdx + 1];
      current_in_qparams = {scale, zp};
      break;
    }
    input_q_params.push_back(current_in_qparams);
  }

  // Extract Input Dtypes
  // Stored as { "type1", "type2", ...}
  std::vector<std::string> input_dtypes;
  for (auto subgraph_input : subgraph_input_tensors) {
    auto consumers = graph_get_consumer_nodes(graph, subgraph_input);

    if (consumers.empty()) {
      throw std::runtime_error("No consumer found for tensor: "s +
                               subgraph_input);
    }

    for (auto consumer : consumers) {
      if (!node_has_attr(*consumer, "in_dtypes")) {
        continue;
      }

      // Find index of tensor
      size_t input_idx = -1;
      auto consumer_inputs = node_get_input_node_args(*consumer);
      for (std::size_t idx = 0; idx < consumer_inputs.size(); ++idx) {
        if (node_arg_get_name(*consumer_inputs[idx]) == subgraph_input) {
          input_idx = idx;
          break;
        }
      }
      if ((int32_t)input_idx == -1) {
        throw std::runtime_error(
            "Couldn't find tensor in list of inputs of consumer");
      }

      if (node_has_attr(*consumer, "in_dtypes")) {
        auto span_in_dtypes = get_strings_attrs(*consumer, "in_dtypes");
        auto in_dtype = span_in_dtypes[input_idx];
        input_dtypes.push_back(in_dtype);
        break;
      } else {
        auto consumer_name = VAIP_ORT_API(node_get_name)(*consumer);
        throw std::runtime_error("in_dtypes attr not found in node: "s +
                                 consumer_name);
      }
    }
  }

  // Extract Output Q Params
  // Stored as { {T1.zp, T1.scale}, {T2.zp, T2.scale}, ...}
  std::vector<std::vector<float>> output_q_params;

  // Extract Output Shapes
  // Stored as { {d11, d12, d13}, {d21, d22}, ...}
  std::vector<std::vector<int64_t>> original_output_shapes;

  // Extract Out Dtypes
  // Stored as { "type1", "type2", ....}
  std::vector<std::string> out_dtypes;

  for (auto subgraph_out : subgraph_output_tensors) {
    auto producer = VAIP_ORT_API(graph_producer_node)(graph, subgraph_out);

    if (!producer) {
      throw std::runtime_error("NO producer found for tensor: "s +
                               subgraph_out);
    }

    size_t output_idx = -1;
    auto producer_output = node_get_output_node_args(*producer);
    for (std::size_t idx = 0; idx < producer_output.size(); ++idx) {
      if (node_arg_get_name(*producer_output[idx]) == subgraph_out) {
        output_idx = idx;
        break;
      }
    }

    if ((int32_t)output_idx == -1) {
      throw std::runtime_error(
          "Couldn't find tensor name in producer output list");
    }

    // Output Q params
    std::vector<float> curr_outq_params;
    bool node_has_output_q = node_has_attr(*producer, "output_q_params");
    if (node_has_output_q) {
      auto span_output_q_params =
          node_get_attr_floats(*producer, "output_q_params");

      int paramIdx = 2 * int(output_idx);
      if ((int32_t)span_output_q_params.size() < paramIdx + 2) {
        throw std::runtime_error(
            "Insufficient input_q_params for node [Todo : getnodename]");
      }

      float scale = span_output_q_params[paramIdx];
      float zp = span_output_q_params[paramIdx + 1];
      curr_outq_params = {scale, zp};
    }
    output_q_params.push_back(curr_outq_params);

    // Output shape
    std::vector<int64_t> curr_output_shapes;
    // static const std::string out_shape_attr_name = "orig_output_shape";
    static const std::string out_shape_attr_name = "shape";
    if (node_has_attr(*producer, out_shape_attr_name)) {
      auto span_original_output_shapes =
          node_get_attr_ints(*producer, out_shape_attr_name);
      for (const auto& elem : span_original_output_shapes) {
        curr_output_shapes.push_back(static_cast<int>(elem));
      }
    }
    original_output_shapes.push_back(std::move(curr_output_shapes));

    // Output Dtype
    if (node_has_attr(*producer, "out_dtypes")) {
      auto span_out_dtypes = get_strings_attrs(*producer, "out_dtypes");
      auto out_dtype = span_out_dtypes[output_idx];
      out_dtypes.push_back(out_dtype);
    } else {
      throw std::runtime_error("out_dtypes attr not found");
    }
  }

  // for (auto node_idx : nodes_in_subgraph) {
  //   nlohmann::json output_names;
  //   auto subg_node = VAIP_ORT_API(graph_get_node)(graph, node_idx);
  //   auto input_args = node_get_input_node_args(*subg_node);
  //   auto output_args = node_get_output_node_args(*subg_node);
  //   auto node_attrs = node_get_attributes(*subg_node);
  // }

  auto& generic_param = *metadef->mutable_generic_param();
  generic_param["input_q_params"] = join(input_q_params);
  generic_param["output_q_params"] = join(output_q_params);
  generic_param["original_output_shapes"] = join(original_output_shapes);
  generic_param["in_dtypes"] = join(input_dtypes);
  generic_param["out_dtypes"] = join(out_dtypes);
  generic_param["model_category"] = model_category;
  if (qos_priority != "")
    generic_param["qos_priority"] = qos_priority;
}
