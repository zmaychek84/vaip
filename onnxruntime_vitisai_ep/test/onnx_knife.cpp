/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

#include <exception>
#include <glog/logging.h>

#include "initialize_vaip.hpp"
#include <filesystem>
#include <limits>
#include <vector>

#include "vaip/vaip.hpp"
#include "vaip/vaip_ort.hpp"
extern "C" {
#include "./getopt.h"
}
using namespace vaip_core;
using namespace std;
using namespace vaip_cxx;
auto onnx_input_file = std::string("");
auto onnx_output_file = std::string("");
auto input_names = std::vector<std::string>();
auto output_names = std::vector<std::string>();

struct KnifePass {
  KnifePass(IPass& self) : self_{self} {}
  void process(IPass& self, Graph& graph) {
    auto graph_ref = GraphRef(graph);
    auto [meta_def, error] = graph_ref.try_fuse("op_name", input_names,
                                                output_names, {}, "ONNX_KNIFE");
    if (meta_def == nullptr) {
      LOG(FATAL) << "try fuse failed. " << error.comments;
    }
    std::cout << " meta_def=" << meta_def->DebugString() << std::endl;

    auto node = graph_ref.fuse(std::move(*meta_def));
    auto subgraph = node.get_function_body();
    subgraph.save(onnx_output_file, onnx_output_file + ".dat", 128u);
  }
  ~KnifePass() {}
  IPass& self_;
};

static void usage() {
  std::cerr
      << "Usage: onnx_knife -I <onnxfile> -O <onnxfile> -i <input_node> -o "
         "<output_node>\n"
      << "Options: \n"
      << "  -I <onnxfile> : Path to the input ONNX model file\n"
      << "  -O <onnxfile> : Output path where ONNX subgraph will be saved\n"
      << "  -i <input_node> : input node_arg name\n"
      << "  -o <output_node> : output node_arg name\n"
      << "  -h : Display the help message and exit.\n";
}
// extract subgraph from onnx model

int main(int argc, char* argv[]) {
  try {
    int opt = 0;
    while ((opt = getopt(argc, argv, "i:o:I:O:h")) != -1) {
      switch (opt) {
      case 'i': {
        input_names.push_back(std::string(optarg));
        break;
      }
      case 'o': {
        output_names.push_back(std::string(optarg));
        break;
      }
      case 'I': {
        onnx_input_file = std::string(optarg);
        break;
      }
      case 'O': {
        onnx_output_file = std::string(optarg);
        break;
      }
      case 'h': {
        usage();
        return 0;
      }
      }
    }

    Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "onnx_knife");
    initialize_vaip();
    auto removed_nodes = std::vector<const Node*>{};
    CHECK(!onnx_input_file.empty()) << "-I <onnxfile> is required;";
    CHECK(!onnx_output_file.empty()) << "-O <onnxfile> is required;";
    CHECK(!input_names.empty()) << "-i <input_node> is required;";
    CHECK(!output_names.empty()) << "-o <output_node> is required;";

    auto model = vaip_cxx::Model::load(onnx_input_file);
    auto graph = model->main_graph();
    graph.resolve();

    auto input_nodes = std::vector<NodeConstRef>{};
    auto output_nodes = std::vector<NodeConstRef>{};
    for (auto& node_arg_name : input_names) {
      auto node_arg = graph.find_node_arg(node_arg_name);
      if (!node_arg) {
        LOG(WARNING) << "cannot find input node arg: " << node_arg_name;
        continue;
      }
      auto node = node_arg.value().find_producer();
      CHECK(node.has_value()) << "cannot find producer: " << node_arg.value();
      input_nodes.push_back(node.value());
    }
    for (auto& node_arg_name : output_names) {
      auto node_arg = graph.find_node_arg(node_arg_name);
      if (!node_arg) {
        LOG(WARNING) << "cannot find output node arg: " << node_arg_name;
        continue;
      }
      auto node = node_arg.value().find_producer();
      CHECK(node.has_value()) << "cannot find producer: " << node_arg.value();
      output_nodes.push_back(node.value());
    }

    CHECK(!input_nodes.empty()) << "-i <input_node> is required;";
    CHECK(!output_nodes.empty()) << "-o <output_node> is required;";

    LOG(INFO) << "inputs :";
    for (auto& input : input_nodes) {
      LOG(INFO) << "\t " << input;
    }
    LOG(INFO) << "outputs :";
    for (auto& output : output_nodes) {
      LOG(INFO) << "\t " << output;
    }
    auto pass_info = ProcessorPassInfo<KnifePass>::pass_info();
    auto context = vaip_core::load_context(std::filesystem::path("."));
    auto passes = std::vector<std::shared_ptr<vaip_core::IPass>>{};
    passes.push_back(vaip_core::IPass::create_pass(context, *pass_info));
    vaip_core::IPass::run_passes(passes, graph);
    /*graph_resolve(graph, true);
    VAIP_ORT_API(graph_save)
        (graph, file, file + ".dat", 128u);*/
  } catch (const std::exception& e) {
    std::cerr << "exception occurs : " << e.what() << "\n";
  }
  return 0;
}
#include "./getopt.c"
