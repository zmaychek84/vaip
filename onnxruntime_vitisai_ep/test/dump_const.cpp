/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

#include <glog/logging.h>

#include <fstream>
#include <iostream>

#ifdef _WIN32
#  pragma warning(push)
#  pragma warning(disable : 4251)
#  pragma warning(disable : 4275)
#endif
#include <xir/graph/graph.hpp>
#ifdef _WIN32
#  pragma warning(pop)
#endif

int main(int argc, char* argv[]) {
  auto model = argv[1];
  auto graph = xir::Graph::deserialize(model);
  auto ops = graph->get_ops();
  for (auto op : ops) {
    if ("const" == op->get_type()) {
      auto data = op->get_attr<std::vector<char>>("data");
      auto filename = std::string("dump/" + op->get_name() + ".bin");
      LOG(INFO) << "DUMP const datat to " << filename;
      std::ofstream(filename).write((char*)&data[0], data.size());
    }
  }
  return 0;
}
