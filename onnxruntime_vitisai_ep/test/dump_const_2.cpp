/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

#include <glog/logging.h>

#include <fstream>
#include <iostream>
#include <string>

#ifdef _WIN32
#  pragma warning(push)
#  pragma warning(disable : 4251)
#  pragma warning(disable : 4275)
#endif
#include <xir/graph/graph.hpp>
#ifdef _WIN32
#  pragma warning(pop)
#endif

static std::string layer_name(const std::string& name) {
  auto name_remove_xfix = xir::remove_xfix(name);
  std::string ret;
  ret.reserve(name_remove_xfix.size());
  std::transform(name_remove_xfix.begin(), name_remove_xfix.end(),
                 std::back_inserter(ret), [](char c) {
                   bool ok = c >= '0' && c <= '9';
                   ok = ok || (c >= 'a' && c <= 'z');
                   ok = ok || (c >= 'A' && c <= 'Z');
                   // ok = ok || (c ==
                   // std::filesystem::path::preferred_separator);
                   ok = ok || (c == '_');
                   return ok ? c : '_';
                 });
  return ret;
}
int main(int argc, char* argv[]) {
  auto model = argv[1];
  auto graph = xir::Graph::deserialize(model);
  auto root = graph->get_root_subgraph();
  auto subgraphs = root->children_topological_sort();
  for (auto sg : subgraphs) {
    if (sg->has_attr("device") &&
        "DPU" == sg->get_attr<std::string>("device") &&
        sg->has_attr("reg_id_to_parameter_value")) {
      // reg_id_to_parameter_value
      auto reg_id_to_parameter_value =
          sg->get_attr<std::map<std::string, std::vector<char>>>(
              "reg_id_to_parameter_value");
      for (auto op : sg->get_ops()) {
        if ("const-fix" == op->get_type()) {
          auto reg_id = op->get_output_tensor()->get_attr<int>("reg_id");
          auto ddr_addr =
              op->get_output_tensor()->get_attr<std::int32_t>("ddr_addr");
          auto data_size = op->get_output_tensor()->get_data_size();

          auto it_value =
              reg_id_to_parameter_value.find("REG_" + std::to_string(reg_id));
          auto filename = "./const/" +
                          layer_name(op->get_output_tensor()->get_name()) +
                          ".bin";
          auto mode =
              std::ios_base::out | std::ios_base::binary | std::ios_base::trunc;
          CHECK(std::ofstream(filename, mode)
                    .write(&it_value->second[ddr_addr], data_size)
                    .good())
              << " faild to dump code to " << filename;
        }
      }
    }
  }

  return 0;
}
