/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

#include <glog/logging.h>
#include <google/protobuf/message.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <utility>
#include <xir/graph/graph.hpp>
#include <xir/tensor/tensor.hpp>
#include <xir/util/tool_function.hpp>

#include "vart/runner_ext.hpp"

using namespace std;

std::vector<std::int32_t> get_index_zeros(const xir::Tensor* tensor) {
  auto ret = tensor->get_shape();
  std::fill(ret.begin(), ret.end(), 0);
  return ret;
}
struct xmodel_result {
  std::map<std::string, std::string> md5_to_tensor_name;
};

xmodel_result run_xmodel_result(const xir::Subgraph* subgraph) {
  auto ret = xmodel_result();
  auto attrs = xir::Attrs::create();
  auto r = vart::RunnerExt::create_runner(subgraph, attrs.get());
  auto inputs = r->get_inputs();
  for (auto input : inputs) {
    size_t batch_size = input->get_tensor()->get_shape()[0];
    auto size_per_batch = input->get_tensor()->get_data_size() / batch_size;
    for (auto i = 0u; i < batch_size; ++i) {
      uint64_t input_data_u64 = 0;
      size_t input_size;
      auto idx = get_index_zeros(input->get_tensor());
      idx[0] = i;
      std::tie(input_data_u64, input_size) = input->data(idx);
      char* input_data = (char*)input_data_u64;
      for (auto d = 0u; d < size_per_batch; ++d) {
        input_data[d] = (char)(d & 0xFF);
      }
    }
  }
  for (auto in : inputs) {
    in->sync_for_write(0, in->get_tensor()->get_data_size() /
                              in->get_tensor()->get_shape()[0]);
  }
  auto outputs = r->get_outputs();
  r->execute_async(inputs, outputs);
  r->wait(0, 0);
  for (auto out : outputs) {
    out->sync_for_read(0, out->get_tensor()->get_data_size() /
                              out->get_tensor()->get_shape()[0]);
  }
  std::cout << "result of subgraph: " << subgraph->get_name() << "\n";
  for (auto i = 0u; i < outputs.size(); ++i) {
    auto output = outputs[i];
    uint64_t output_data_u64 = 0;
    size_t output_size;
    auto idx = get_index_zeros(output->get_tensor());
    std::tie(output_data_u64, output_size) = output->data(idx);
    void* output_data = (char*)output_data_u64;
    auto md5 = xir::get_md5_of_buffer(output_data, output_size);
    std::cout << "\t" << md5 << "\t" << output->get_tensor()->get_name()
              << "\n";
    ret.md5_to_tensor_name.insert(
        std::make_pair(md5, output->get_tensor()->to_string()));
  }
  return ret;
}

std::vector<xmodel_result> run_xmodel_results(const xir::Graph* graph) {
  auto ret = std::vector<xmodel_result>();
  for (auto subgraph :
       graph->get_root_subgraph()->children_topological_sort()) {
    if (subgraph->has_attr("device") &&
        subgraph->get_attr<std::string>("device") == "DPU") {
      ret.emplace_back(run_xmodel_result(subgraph));
    }
  }
  return ret;
}

std::pair<bool, std::string>
compare_results(const std::vector<xmodel_result>& r1,
                const std::vector<xmodel_result>& r2) {
  std::ostringstream str;
  if (r1.size() != r2.size()) {
    str << "num of subgraphs is not same. " << r1.size() << " vs " << r2.size();
    return std::make_pair(false, str.str());
  }
  auto size = r1.size();
  auto ret = true;
  for (auto i = 0u; i < size; ++i) {
    auto& rs1 = r1[i];
    auto& rs2 = r2[i];
    if (rs1.md5_to_tensor_name.size() > rs2.md5_to_tensor_name.size()) {
      str << "num of result for " << i << " DPU subgraph is not right. "
          << rs1.md5_to_tensor_name.size() << " vs "
          << rs2.md5_to_tensor_name.size();
      return std::make_pair(false, str.str());
    }
    // auto size2 = rs1.md5_to_tensor_name.size();
    auto b1 = rs1.md5_to_tensor_name.begin();
    auto e1 = rs1.md5_to_tensor_name.end();
    auto b2 = rs2.md5_to_tensor_name.begin();
    auto e2 = rs2.md5_to_tensor_name.end();
    for (auto i1 = b1; i1 != e1; ++i1) {
      auto res = rs2.md5_to_tensor_name.find(i1->first);
      if (res == e2) {
        ret = false;
        for (auto i1 = b1; i1 != e1; ++i1) {
          str << "ref xmodel: subgraph[" << i << "] result is not same." //
              << i1->first << ";"                                        //
              << i1->second << ";";
        }
        for (auto i2 = b2; i2 != e2; ++i2) {
          str << "tested xmodel: subgraph[" << i << "] result is not same." //
              << i2->first << ";"                                           //
              << i2->second << ";";
        }
      }
    }
  }
  return std::make_pair(ret, str.str());
}

int main(int argc, char* argv[]) {
  auto g1 = xir::Graph::deserialize(argv[1]);
  auto g2 = xir::Graph::deserialize(argv[2]);
  auto r1 = run_xmodel_results(g1.get());
  auto r2 = run_xmodel_results(g2.get());
  auto ret = compare_results(r1, r2);
  if (ret.first) {
    return 0;
  }
  std::cerr << ret.second << std::endl;
  return 1;
}
