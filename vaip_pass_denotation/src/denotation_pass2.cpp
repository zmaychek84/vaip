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

#include "./denotation_pass2.hpp"

#include <glog/logging.h>

#include <unordered_map>

#include "vaip/util.hpp"
#include "vaip/vaip.hpp"
#include "vitis/ai/env_config.hpp"
DEF_ENV_PARAM(DEBUG_DENOTATION_PASS, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_DENOTATION_PASS) >= n)

namespace vaip_pass_denotation {

bool NodeDenotation::apply(Graph& graph, const Node& node) {
  NodeActionState state{node};
  auto layout_check =
      [](std::vector<std::unique_ptr<layout_t>>& layout) -> bool {
    return std::all_of(
        layout.begin(), layout.end(),
        [](const std::unique_ptr<layout_t>& l) { return nullptr != l; });
  };
  if (!layout_check(state.input_layouts_) ||
      !layout_check(state.output_layouts_)) {
    return false;
  }
  for (auto& action : actions_) {
    action(state, graph, node);
  }
  state.check_is_modified();
  if (state.is_modified()) {
    MY_LOG(1) << "before update " << node_as_string(node);
  }
  state.update_layout();
  if (state.is_modified()) {
    MY_LOG(1) << "after update " << node_as_string(node);
  }
  return state.is_modified();
}

NodeDenotation& NodeDenotation::action(const node_action_t& action) {
  this->actions_.push_back(action);
  return *this;
}

static std::vector<std::unique_ptr<layout_t>>
get_node_args_layouts(const std::vector<const NodeArg*>& node_args) {
  auto ret = std::vector<std::unique_ptr<layout_t>>(node_args.size());
  auto i = 0u;
  for (auto& node_arg : node_args) {
    if (node_arg_exists(*node_arg)) {
      ret[i] = node_arg_get_denotation(*node_arg);
    }
    i = i + 1;
  }
  return ret;
}

NodeActionState::NodeActionState(const Node& node)
    : modified_{false},                                                    //
      node_{&node},                                                        //
      inputs_{node_get_input_node_args(node)},                             //
      input_layouts_{get_node_args_layouts(inputs_)},                      //
      outputs_{
          std::vector<const NodeArg*>{&(node_get_output_node_arg(node))}}, //
      output_layouts_{get_node_args_layouts(outputs_)} {}

void NodeActionState::negotiate_layout(std::vector<denotation_view_t> views) {
  auto c = denotation_t{};
  for (auto& v : views) {
    if (c.empty()) {
      c = v.denotation;
    }
    if (!c.empty() && !v.denotation.empty()) {
      if (c != v.denotation) {
        LOG(WARNING) << " layout conflicts:" << c << " != " << v.denotation
                     << " " << node_as_string(*node_);
        return;
      }
    }
  }
  if (c.empty()) {
    return;
  }
  for (auto& v : views) {
    v.denotation = c;
  }
}

denotation_t& NodeActionState::get_denotation(size_t input_or_output,
                                              size_t arg_idx, size_t dim_idx) {
  auto& layout = get_layout(input_or_output, arg_idx);
  CHECK_LT(dim_idx, layout.size())
      << node_as_string(*node_) << "arg_index " << arg_idx << " " //
      ;
  return layout[dim_idx];
}

layout_t& NodeActionState::get_layout(size_t input_or_output, size_t arg_idx) {
  CHECK(node_ != nullptr);
  CHECK_LT(input_or_output, 2u) << node_as_string(*node_);
  auto layout = input_or_output == 0 ? input_layouts_[arg_idx].get()
                                     : output_layouts_[arg_idx].get();
  return *layout;
}

void NodeActionState::set_input_layout(size_t arg_idx,
                                       const layout_t& layout1) {
  auto layout = layout1;
  if (arg_idx < input_layouts_.size()) {
    auto input_layout = input_layouts_[arg_idx].get();
    if (input_layout->size() != layout.size()) {
      LOG(WARNING) << container_as_string(*input_layout) << " "
                   << container_as_string(layout) << " "
                   << node_as_string(*node_);
      error_counter_ = error_counter_ + 1;
    } else {
      for (auto i = 0u; i < input_layout->size(); ++i) {
        negotiate_layout(
            std::vector<denotation_view_t>{{(*input_layout)[i]}, {layout[i]}});
      }
    }
  }
}

void NodeActionState::set_output_layout(const layout_t& layout1) {
  auto layout = layout1;
  auto& output_layout = output_layouts_[0];
  if (output_layout->size() != layout.size()) {
    LOG(WARNING) << container_as_string(*output_layout) << " "
                 << container_as_string(layout) << " "
                 << node_as_string(*node_);
    error_counter_ = error_counter_ + 1;
  } else {
    for (auto i = 0u; i < output_layout->size(); ++i) {
      negotiate_layout(
          std::vector<denotation_view_t>{{(*output_layout)[i]}, {layout[i]}});
    }
  }
}

void NodeActionState::copy_input_layout(size_t input_index) {
  if (input_index >= input_layouts_.size()) {
    return;
  }
  auto input_layout = input_layouts_[input_index].get();
  auto output_layout = output_layouts_[0].get();
  if (input_layout->size() != output_layout->size()) {
    LOG(WARNING) << " not same dim size: "
                 << "input_index " << input_index << " " //
                 << node_as_string(*node_);
    return;
  }
  for (auto i = 0u; i < input_layout->size(); ++i) {
    negotiate_layout(std::vector<denotation_view_t>{{(*input_layout)[i]},
                                                    {(*output_layout)[i]}});
  }
}

void NodeActionState::copy_all_input() {
  for (auto i = 0u; i < input_layouts_.size(); ++i) {
    copy_input_layout(i);
  }
}

void NodeActionState::set_broadcast_layout() {
  auto views = std::vector<std::vector<denotation_view_t>>{};
  for (auto layouts : {&input_layouts_, &output_layouts_}) {
    for (auto i = 0u; i < layouts->size(); ++i) {
      auto layout = (*layouts)[i].get();
      for (auto dim_idx = 0u; dim_idx < layout->size(); ++dim_idx) {
        auto j = layout->size() - dim_idx - 1;
        if (dim_idx >= views.size()) {
          views.emplace_back();
        }
        views[dim_idx].emplace_back(denotation_view_t{(*layout)[j]});
      }
    }
  }
  for (auto& view : views) {
    negotiate_layout(view);
  }
}

void NodeActionState::check_is_modified() {
  if (error_counter_ > 0) {
    return;
  }
  modified_ = false;
  for (auto i = 0u; i < inputs_.size(); ++i) {
    auto pdenotation = node_arg_get_denotation(*inputs_[i]);
    if (nullptr == pdenotation)
      continue;
    auto input_layout = input_layouts_[i].get();
    modified_ = modified_ || (*input_layout != *pdenotation);
  }
  for (auto i = 0u; i < outputs_.size(); ++i) {
    auto pdenotation = node_arg_get_denotation(*outputs_[i]);
    if (nullptr == pdenotation)
      continue;
    auto output_layout = output_layouts_[i].get();
    modified_ = modified_ || (*output_layout != *pdenotation);
  }
}

void NodeActionState::update_layout() {
  for (auto i = 0u; i < inputs_.size(); ++i) {
    auto input_layout = input_layouts_[i].get();
    VAIP_ORT_API(node_arg_set_denotation)(*inputs_[i], *input_layout);
  }
  for (auto i = 0u; i < outputs_.size(); ++i) {
    auto output_layout = output_layouts_[i].get();
    VAIP_ORT_API(node_arg_set_denotation)(*outputs_[i], *output_layout);
  }
}

NodeDenotation&
NodeDenotation::set_output_layout(const std::vector<std::string>& layout0) {
  auto layout = layout0;
  return action([layout](NodeActionState& self, Graph& graph,
                         const Node& node) { self.set_output_layout(layout); });
}

NodeDenotation&
NodeDenotation::set_input_layout(size_t input_index,
                                 const std::vector<std::string>& layout0) {
  auto layout = layout0;
  return action([layout, input_index](NodeActionState& self, Graph& graph,
                                      const Node& node) {
    self.set_input_layout(input_index, layout);
  });
}

NodeDenotation& NodeDenotation::copy_input_layout(size_t input_index) {
  return action(
      [input_index](NodeActionState& self, Graph& graph, const Node& node) {
        self.copy_input_layout(input_index);
      });
}

NodeDenotation& NodeDenotation::copy_all_input() {
  return action([](NodeActionState& self, Graph& graph, const Node& node) {
    self.copy_all_input();
  });
}

NodeDenotation& NodeDenotation::set_broadcast_layout() {
  return action([](NodeActionState& self, Graph& graph, const Node& node) {
    self.set_broadcast_layout();
  });
}

bool denotation_node(Graph& graph, const Node& node) {
  static std::unordered_map<std::string, NodeDenotation> actions = {
      // begin
      {"ai.onnx:Conv", NodeDenotation()
                           .set_input_layout(0, {"N", "C", "H", "W"})
                           .set_input_layout(1, {"O", "I", "H", "W"})
                           .set_input_layout(2, {"O"})
                           .set_output_layout({"N", "C", "H", "W"})},
      {"ai.onnx:MaxPool", NodeDenotation()
                              .set_input_layout(0, {"N", "C", "H", "W"})
                              .set_output_layout({"N", "C", "H", "W"})},
      {"ai.onnx:AveragePool", NodeDenotation()
                                  .set_input_layout(0, {"N", "C", "H", "W"})
                                  .set_output_layout({"N", "C", "H", "W"})},
      {"ai.onnx:Add", NodeDenotation().set_broadcast_layout()},
      {"ai.onnx:Mul", NodeDenotation().set_broadcast_layout()},
      {"ai.onnx:Relu", NodeDenotation().copy_input_layout(0)},
      {"ai.onnx:Pad", NodeDenotation().copy_input_layout(0)},
      {"ai.onnx:QuantizeLinear", NodeDenotation().copy_input_layout(0)},
      {"ai.onnx:DequantizeLinear", NodeDenotation().copy_input_layout(0)},
      {"ai.onnx:Concat", NodeDenotation().copy_all_input()},
      {"ai.onnx:Sigmoid", NodeDenotation().copy_input_layout(0)},
      {"ai.onnx:Clip", NodeDenotation().copy_input_layout(0)},
      {"ai.onnx:Reshape",
       NodeDenotation().action(
           [](NodeActionState& self, Graph& graph, const Node& node) {
             MY_LOG(1) << "Reshape layout " << node_as_string(node);
             self.handle_reshape();
           })},
      {"ai.onnx:Transpose",
       NodeDenotation().action(
           [](NodeActionState& self, Graph& graph, const Node& node) {
             self.handle_transpose();
           })},
      {"ai.onnx:Gemm", NodeDenotation().action(
                           [](NodeActionState& self, Graph& graph,
                              const Node& node) { self.handle_gemm(node); })},
      {"ai.onnx:Squeeze",
       NodeDenotation().action(
           [](NodeActionState& self, Graph& graph, const Node& node) {
             self.handle_squeeze(graph, node);
           })},
      {"com.xilinx:fix", NodeDenotation().copy_input_layout(0)},
      // end
  };
  auto& op_type = VAIP_ORT_API(node_op_type)(node);
  auto& op_domain = VAIP_ORT_API(node_op_domain)(node);
  auto domain = op_domain.empty() ? std::string("ai.onnx") : op_domain;
  auto key = domain + ":" + op_type;
  auto it = actions.find(key);
  auto ret = false;
  if (it != actions.end()) {
    ret = it->second.apply(graph, node);
  }
  return ret;
}

bool layout_all_set(const layout_t& layout) {
  return std::all_of(layout.begin(), layout.end(),
                     [](const denotation_t& d) { return !d.empty(); });
}
bool layout_none_set(const layout_t& layout) {
  return std::all_of(layout.begin(), layout.end(),
                     [](const denotation_t& d) { return d.empty(); });
}

DenotationPass::~DenotationPass() {
  // auto map = std::unordered_map<std::string, const NodeArg*>();
  // for (auto node_idx :
  // VAIP_ORT_API(graph_get_node_in_topoligical_order)(*graph_))
  // {
  //   auto node = VAIP_ORT_API(graph_get_node)(graph_, node_idx);
  //   // auto inputs = node_get_input_node_args(*node);
  //   // auto input_layouts = get_node_args_layouts(inputs);
  //   auto& output = VAIP_ORT_API(node_get_output_node_arg)(*node);
  //   auto output_layouts = get_node_args_layouts({&output});
  //   // for (auto i = 0u; i < inputs.size(); ++i) {
  //   //   if (layout_all_set(input_layouts[i])) {
  //   //   } else if (layout_none_set(input_layouts[i])) {
  //   //     map[node_arg_get_name(*inputs[i])] = inputs[i];
  //   //   } else {
  //   //     map[node_arg_get_name(*inputs[i])] = inputs[i];
  //   //   }
  //   // }

  //   if (layout_all_set(output_layouts[0])) {
  //   } else if (layout_none_set(output_layouts[0])) {
  //     // map[node_arg_get_name(output)] = &output;
  //     LOG(WARNING) << "no layout: " << node_as_string(*node);
  //   } else {
  //     LOG(WARNING) << "partial layout: " <<
  //     node_as_string(*node);
  //     // map[node_arg_get_name(output)] = &output;
  //   }
  // }
  // for (auto& m : map) {
  //   LOG(WARNING) << "no layout: " <<
  //   node_arg_as_string(*m.second);
  // }
}
} // namespace vaip_pass_denotation
