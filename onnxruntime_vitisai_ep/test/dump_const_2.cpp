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
