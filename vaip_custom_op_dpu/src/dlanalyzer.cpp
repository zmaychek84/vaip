/*
 *     The Xilinx Vitis AI Vaip in this distribution are provided under the
 * following free and permissive binary-only license, but are not provided in
 * source code form.  While the following free and permissive license is similar
 * to the BSD open source license, it is NOT the BSD open source license nor
 * other OSI-approved open source license.
 *
 *      Copyright (C) 2022 Xilinx, Inc. All rights reserved.
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

#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include <xir/graph/graph.hpp>

using json = nlohmann::json;

std::string gen_dpu_timestamp_info(const xir::Graph* graph) {
  constexpr uint32_t PDI_ID_BIT_OFFSET = 10;
  constexpr uint32_t GRAPH_TYPE_BIT_OFFSET = 20;
  constexpr uint32_t TYPE_WIDTH = 4;
  constexpr uint32_t TYPE_MASK = ((1 << TYPE_WIDTH) - 1);
  constexpr uint32_t PDI_ID_MASK = ((1 << PDI_ID_BIT_OFFSET) - 1);
  constexpr uint32_t MAX_LAYER_ID = ((1 << PDI_ID_BIT_OFFSET) - 1);
  enum RecordTimerType {
    PDI_START = 1,
    PDI_END = 0,
    SUPER_LAYER_START = 3,
    SUPER_LAYER_END = 2
  };

  json j;
  j = R"({"events": []})"_json;

  auto simple_hash = [](const std::string& str) {
    uint32_t hash_value = 0;
    for (char ch : str) {
      hash_value = hash_value * 11 + ch;
    }
    return hash_value;
  };

  auto root = graph->get_root_subgraph();
  for (auto dev_subg : root->children_topological_sort()) {
    for (auto pdi_subg : dev_subg->children_topological_sort()) {
      uint32_t pdi_uid = 0u;
      for (auto& o_tensor : pdi_subg->get_output_tensors()) {
        auto o_tensor_name = ((o_tensor))->get_name();
        pdi_uid = pdi_uid ^ simple_hash(o_tensor_name);
      }
      auto uid = 0u;
      pdi_uid &= PDI_ID_MASK;
      uid = pdi_uid << PDI_ID_BIT_OFFSET;
      uid |= PDI_START << GRAPH_TYPE_BIT_OFFSET;

      // make start event for pdi graphs
      json event_obj = json::object();
      event_obj["start"] = true;
      event_obj["name"] = pdi_subg->get_name();
      event_obj["id"] = uid;
      event_obj["type"] = "pdi_graph";
      if (pdi_subg->has_attr("type")) {
        event_obj["op_type"] = pdi_subg->get_attr<std::string>("type");
      }
      j["events"].push_back(event_obj);

      uid = pdi_uid << PDI_ID_BIT_OFFSET;
      uid |= PDI_END << GRAPH_TYPE_BIT_OFFSET;
      // make end event for pdi graphs
      event_obj["start"] = false;
      event_obj["id"] = uid;
      j["events"].push_back(event_obj);

      uint32_t layer_id = 0;
      for (auto superlayer_subg : pdi_subg->children_topological_sort()) {
        uid = 0;
        std::vector<char> mc_code;
        if (superlayer_subg->has_attr("mc_code")) {
          mc_code = superlayer_subg->get_attr<decltype(mc_code)>("mc_code");
        }

        // make start event for superlayer graphs
        if (mc_code.size() > 4) {
          // a valid superlayer
          if (mc_code[3] != 10)
            uid = (pdi_uid << PDI_ID_BIT_OFFSET) | layer_id++;
          else
            uid = (pdi_uid << PDI_ID_BIT_OFFSET) | MAX_LAYER_ID;
        } else {
          uid = (pdi_uid << PDI_ID_BIT_OFFSET) | MAX_LAYER_ID;
        }

        json event_obj = json::object();
        event_obj["start"] = true;
        event_obj["name"] = superlayer_subg->get_name();
        uid &= ~(TYPE_MASK << GRAPH_TYPE_BIT_OFFSET);
        event_obj["id"] = uid | (SUPER_LAYER_START << GRAPH_TYPE_BIT_OFFSET);
        event_obj["type"] = "layer";
        if (superlayer_subg->has_attr("type")) {
          event_obj["op_type"] = superlayer_subg->get_attr<std::string>("type");
        }
        j["events"].push_back(event_obj);

        // make start event for superlayer graphs
        event_obj["start"] = false;
        uid &= ~(TYPE_MASK << GRAPH_TYPE_BIT_OFFSET);
        event_obj["id"] = uid | (SUPER_LAYER_END << GRAPH_TYPE_BIT_OFFSET);
        j["events"].push_back(event_obj);
      }
    }
  }

  return j.dump(2);
}

std::vector<std::pair<std::string, std::string>> input_ports;
std::vector<std::pair<std::string, std::string>> output_ports;

json tensor_to_json(const xir::Tensor* t, const std::string op_name,
                    const char* direction) {
  json port_obj = {};

  std::string port_name = t->get_name();
  std::string port_id = t->get_name() + "@" + op_name;

  port_obj["name"] = port_name;
  port_obj["id"] = port_id;
  port_obj["direction"] = direction;
  port_obj["properties"] = json::array();

  if (strncmp(direction, "in", 2) == 0)
    input_ports.push_back(std::make_pair(port_name, port_id));
  else if (strncmp(direction, "out", 3) == 0)
    output_ports.push_back(std::make_pair(port_name, port_id));
  else
    assert(true);

  return port_obj;
}

std::string gen_fused_viz(const xir::Graph* graph) {
  json j = R"({
        "flexml_graph_metadata": {
            "schema_version" : {
                "major": "0",
                "minor": "0",
                "patch": "0"
                },
            "design": {
                "name": "graph"
                },
            "processors": [
                {
                  "name": "cpu",
                  "properties": []
                },
                {
                  "name": "NPU",
                  "properties": []
                }
            ],
            "blocks": [],
            "operators": [],
            "connections": []
        }
    })"_json;

  auto root = graph->get_root_subgraph();

  for (auto dev_subg : root->children_topological_sort()) {
    std::string dev;
    if (dev_subg->has_attr("device")) {
      dev = dev_subg->get_attr<std::string>("device");
    } else {
      continue;
    }

    std::string dev_subg_name = dev_subg->get_name();

    // CPU OPs
    if ((dev == "CPU") || (dev == "USER")) {
      for (auto& op : dev_subg->get_ops()) {
        json op_obj = json::object();

        op_obj["name"] = op->get_name();
        op_obj["id"] = op->get_name();
        op_obj["processor"] = "cpu";
        op_obj["operator_type"] = op->get_type();
        op_obj["ports"] = json::array();

        // process input tensors
        for (auto& i_port : op->get_input_tensors()) {
          op_obj["ports"].push_back(
              tensor_to_json(i_port, op->get_name(), "in"));
        }

        // process output tensors
        auto o_port = op->get_output_tensor();
        op_obj["ports"].push_back(
            tensor_to_json(o_port, op->get_name(), "out"));

        j["flexml_graph_metadata"]["operators"].push_back(op_obj);
      }
    }

    // IPU OPs
    if (dev == "DPU") {
      // process pdi subgraphs
      for (auto& pdi_subg : dev_subg->children_topological_sort()) {
        json pdi_obj = json::object();

        pdi_obj["name"] = pdi_subg->get_name();
        pdi_obj["id"] = "ipu_0";
        pdi_obj["design_name"] = "design1", pdi_obj["children"] = json::array();
        pdi_obj["properties"] = json::array();
        pdi_obj["data_transfers"] = json::array();
        pdi_obj["processor"] = "NPU";
        pdi_obj["operators"] = json::array();

        // process superlayer subgraphs
        for (auto& op : pdi_subg->children_topological_sort()) {
          json op_obj = json::object();

          op_obj["name"] = op->get_name();
          op_obj["id"] = op->get_name();
          op_obj["processor"] = "NPU";
          if (op->has_attr("type")) {
            op_obj["operator_type"] = op->get_attr<std::string>("type");
          }
          op_obj["ports"] = json::array();

          // process input tensors
          for (auto& i_port : op->get_input_tensors()) {
            op_obj["ports"].push_back(
                tensor_to_json(i_port, op->get_name(), "in"));
          }

          // process output tensors
          for (auto& o_port : op->get_output_tensors()) {
            op_obj["ports"].push_back(
                tensor_to_json(o_port, op->get_name(), "out"));
            pdi_obj["operators"].push_back(op_obj);
          }
        }

        j["flexml_graph_metadata"]["blocks"].push_back(pdi_obj);
      }
    }
  }

  // Generate connections
  for (auto i_port : input_ports) {
    static auto c_id = 0u;

    json conn_obj = json::object();

    auto i_tensor_name = std::get<0>(i_port);
    auto i_tensor_id = std::get<1>(i_port);

    for (auto o_port : output_ports) {
      auto o_tensor_name = std::get<0>(o_port);
      auto o_tensor_id = std::get<1>(o_port);

      if (o_tensor_name == i_tensor_name) {
        conn_obj["id"] = std::string("C_") + std::to_string(c_id++);
        conn_obj["from_port_id"] = o_tensor_id;
        conn_obj["to_port_id"] = i_tensor_id;
        conn_obj["properties"] = json::array();

        j["flexml_graph_metadata"]["connections"].push_back(conn_obj);
      }
    }
  }

  return j.dump(2);
}
