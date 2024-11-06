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

#include "node_arg.hpp"
#include "vaip/vaip.hpp"
namespace {
using namespace vaip_core;
/*
  test case PST
*/

static float get_const_data_convert_to_float(IPass& self,
                                             const NodeInput& node_input) {
  float ret = 0.0f;
  if (node_arg_get_element_type(*node_input.node_arg) == 4) {
    auto data = self.get_const_data<uint16_t>(*node_input.node);
    ret = (float)data[0];
  } else if (node_arg_get_element_type(*node_input.node_arg) == 3) {
    auto data = self.get_const_data<int8_t>(*node_input.node);
    ret = (float)data[0];
  } else {
    LOG(FATAL) << "TODO, not supported data type "
               << node_arg_get_element_type(*node_input.node_arg);
  }
  return ret;
}

struct ConvertPad {
  ConvertPad(IPass& self) : self_{self} {}
  std::unique_ptr<Rule> create_rule() {
    auto builder = PatternBuilder();
    std::shared_ptr<Pattern> pat_input_ = builder.wildcard();
    std::shared_ptr<Pattern> pat_pads_ = builder.xir_const_op();
    std::shared_ptr<Pattern> pat_dq_input_ = builder.xir_const_op();
    std::shared_ptr<Pattern> pat_dq_scale_ = builder.xir_const_op();
    std::shared_ptr<Pattern> pat_dq_zero_point_ = builder.xir_const_op();
    std::shared_ptr<Pattern> pat_dq_ =
        builder.node3("com.xilinx:dequantize_linear",
                      {pat_dq_input_, pat_dq_scale_, pat_dq_zero_point_},
                      {false, false, true});
    std::shared_ptr<Pattern> pat_pad_ =
        builder.node2("Pad", {pat_input_, pat_pads_, pat_dq_});

    return Rule::create_rule(
        pat_pad_, [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
          LOG(INFO) << " ====  PAD match";
          auto input = binder[pat_input_->get_id()];
          auto pad_node = binder[pat_pad_->get_id()];
          auto pads_node = binder[pat_pads_->get_id()];
          auto dq_input = binder[pat_dq_input_->get_id()];
          auto dq_scale = binder[pat_dq_scale_->get_id()];
          auto dq_zero_point = binder[pat_dq_zero_point_->get_id()];
          if (node_arg_get_element_type(*dq_input.node_arg) != 4 &&
              node_arg_get_element_type(*dq_input.node_arg) != 3) {
            LOG(WARNING) << "cancel xir conversion , Pad to pad, not supported "
                            "data type "
                         << node_arg_get_element_type(*dq_input.node_arg);
            return false;
          }
          auto mode = node_get_attr_string_with_default(
              *pad_node.node, "mode", std::string("constant"));
          auto xir_mode = std::string("CONSTANT");
          if (mode == "constant") {
            xir_mode = std::string("CONSTANT");
          } else if (mode == "reflect") {
            xir_mode = std::string("REFLECT");
          } else if (mode == "edge") {
            xir_mode = std::string("SYMMETRIC");
          } else {
            LOG(FATAL) << "unknown pad mode: " << mode;
          }
          LOG(INFO) << " ===  mode " << mode;
          gsl::span<const int64_t> pads_data;
          if (pads_node.node != nullptr) {
            pads_data = self_.get_const_data<int64_t>(*pads_node.node);
          } else {
            // since onnx opset version 11 , pads from
            // 'attribute' becomes 'input'
            pads_data = node_get_attr_ints(*pad_node.node, "pads");
          }
          auto size = pads_data.size();
          auto half_size = size / 2;
          auto paddings = std::vector<int64_t>(pads_data.size());
          for (auto i = 0u; i < half_size; ++i) {
            paddings[2 * i] = pads_data[i];
            paddings[2 * i + 1] = pads_data[half_size + i];
          }

          auto dq_input_data = get_const_data_convert_to_float(self_, dq_input);
          auto dq_scale_data = self_.get_const_data<float>(*dq_scale.node);
          auto dq_zero_point_data =
              get_const_data_convert_to_float(self_, dq_zero_point);
          auto constant_value_data = 0.0f;
          constant_value_data =
              (dq_input_data - dq_zero_point_data) * dq_scale_data[0];
          auto constant_values = std::vector<float>(size, constant_value_data);

          NodeBuilder(*graph, self_)
              .set_op_type("pad")
              .set_input_nodes({input.node})
              .add("paddings", paddings)
              .add("mode", xir_mode)
              .add("constant_values", constant_values)
              .set_anchor_point1(*pad_node.node)
              .build();
          return true;
        });
  }
  void process(IPass& self, Graph& graph) { create_rule()->apply(&graph); }

  IPass& self_;
};
} // namespace

DEFINE_VAIP_PASS(ConvertPad, vaip_pass_convert_pad)
