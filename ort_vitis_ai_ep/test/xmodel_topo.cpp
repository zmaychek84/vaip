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

#include <iostream>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <xir/graph/graph.hpp>
#include <xir/op/op_def.hpp>
using namespace std;

static void topological_sort(
    const std::vector<const xir::Op*>& from,
    const std::function<void(const xir::Op*)>& enter,
    const std::function<void(const xir::Op*)>& leave,
    const std::function<bool(const xir::Op*, const xir::Op*)>& comp,
    const std::function<bool(const xir::Op* from, const xir::Op* to)>& stop) {
  using WorkEntry =
      std::pair<const xir::Op*, bool>; // bool represents leave or not
  std::vector<WorkEntry> stack;
  stack.reserve(from.size());
  for (auto node : from) {
    stack.emplace_back(node, false);
  }

  std::unordered_set<const xir::Op*> visited{};
  while (!stack.empty()) {
    const WorkEntry last_entry = stack.back();
    stack.pop_back();

    if (last_entry.first == nullptr) {
      continue;
    }
    auto n = last_entry.first;

    if (last_entry.second) {
      // leave node
      leave(n);
      continue;
    }

    auto it = visited.find(n);
    if (it != visited.end()) {
      continue;
    }
    visited.insert(n);

    if (enter) {
      enter(n);
    }

    if (leave) {
      stack.emplace_back(n, true);
    }

    if (comp) {
      std::cerr << "not supported" << std::endl;
      exit(1);
    } else {
      auto opdef = n->get_opdef();
      auto inputs = n->get_input_ops();
      for (auto& input_args : opdef->input_args()) {
        auto input_name = input_args.name;
        for (auto& input2 : n->get_input_ops(input_name)) {
          if (stop && stop(n, input2)) {
            continue;
          }
          if (visited.find(input2) == visited.end()) {
            stack.emplace_back(input2, false);
          }
        }
      }
    }
  }
}

static int get_op_index(const xir::Op* n, const xir::Op* input_arg) {
  auto opdef = n->get_opdef();
  auto inputs = n->get_input_ops();
  int c = 0;
  for (auto& input_args : opdef->input_args()) {
    auto input_name = input_args.name;
    for (auto& input2 : n->get_input_ops(input_name)) {
      if (input_arg == input2) {
        return c;
      }
      c = c + 1;
    }
  }
  LOG(FATAL) << "op " << input_arg->get_name() << " is not an input of op "
             << n->get_name();
  return -1;
}
static std::string to_name(const std::vector<int>& name) {
  std::ostringstream str;
  str << "b";
  for (auto& i : name) {
    str << "_" << i;
  }
  str << "_e";
  return str.str();
}
int main(int argc, char* argv[]) {
  auto graph = xir::Graph::deserialize(std::string(argv[1]));
  auto ops = std::vector<const xir::Op*>();
  ops.reserve((size_t)argc);
  for (auto i = 2; i < argc; ++i) {
    auto op = graph->get_op(std::string(argv[i]));
    if (op == nullptr) {
      std::cerr << "cannot find op: name=" << argv[i] << std::endl;
      continue;
    }
    ops.push_back(op);
  }
  if (ops.empty()) {
    auto graph_output = graph->get_tail_ops();
    ops.assign(graph_output.begin(), graph_output.end());
  }
  int g_counter = 0;
  auto ops_name_mapping =
      std::unordered_map<const xir::Op*, std::vector<int>>();
  topological_sort(
      ops,
      [&g_counter, &ops_name_mapping](const xir::Op* op) {
        auto& m = ops_name_mapping[op];
        auto fanout_ops = op->get_fanout_ops();
        if (fanout_ops.empty()) {
          m.push_back(g_counter);
          g_counter = g_counter + 1;
        } else {
          auto name = std::vector<int>();
          auto arg_index = -1;
          for (auto fanout_op : fanout_ops) {
            auto it = ops_name_mapping.find(fanout_op);
            if (it != ops_name_mapping.end()) {
              name = it->second;
              arg_index = get_op_index(fanout_op, op);
              name.push_back(arg_index);
              break;
            }
          }
          CHECK(!name.empty()) << "cannot find the first fanout op";
          m = name;
        }
      },
      [&ops_name_mapping](const xir::Op* op) {
        auto it = ops_name_mapping.find(op);
        CHECK(it != ops_name_mapping.end());
        std::cout << "\t" << op->get_name() << "\t" << to_name(it->second)
                  << " " << op->get_type() << " "
                  << op->get_output_tensor()->to_string() << std::endl;
      },
      nullptr, nullptr);
  return 0;
}
