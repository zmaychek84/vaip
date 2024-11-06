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

#include "vaip/vaip.hpp"
#include "vitis/ai/env_config.hpp"
#include <glog/logging.h>
DEF_ENV_PARAM(DEBUG_MERGE_ADD_INTO_CONV_BIAS, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_MERGE_ADD_INTO_CONV_BIAS) >= n)
namespace {
using namespace vaip_core;
struct MergeAddIntoConvBias {
  std::unique_ptr<Rule> create_rule(const char* name) {
    auto builder = PatternBuilder();
    std::shared_ptr<Pattern> pat_conv2d = builder.node2(
        std::string(name), {builder.wildcard(), builder.wildcard()});
    std::shared_ptr<Pattern> pat_right = builder.xir_const_op();
    std::shared_ptr<Pattern> pat_add =
        builder.node2("com.xilinx:add", {pat_conv2d, pat_right});
    return Rule::create_rule(
        pat_add, [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
          auto ni_conv2d = binder[pat_conv2d->get_id()];
          auto ni_add = binder[pat_add->get_id()];
          auto ni_right = binder[pat_right->get_id()];
          auto data = self_.get_const_data<float>(*ni_right.node);
          auto size = data.size();
          auto new_shape = std::vector<int64_t>{(int64_t)size};
          auto bias_data = std::vector<float>(size);
          for (auto i = 0u; i < size; ++i) {
            bias_data[i] = data[i];
          }
          auto& bias =
              NodeBuilder(*graph, self_)
                  .clone_inputs(*ni_right.node)
                  .clone_op_type(*ni_right.node)
                  .clone_data_type(*ni_right.node_arg)
                  .clone_attrs(*ni_right.node)
                  .set_anchor_point3(*ni_right.node_arg, {"reshape"}, new_shape)
                  .build();
          self_.create_const_alias(bias, *ni_right.node);
          self_.copy_fix_info(*ni_right.node, bias);

          NodeBuilder(*graph, self_)
              .clone_node(*ni_conv2d.node)
              .append_input(bias)
              .set_anchor_point1(*ni_add.node)
              .build();
          return true;
        });
  }
  MergeAddIntoConvBias(IPass& self) : self_{self} {}
  void process(IPass& self, Graph& graph) {
    std::unique_ptr<BaseRule> rules[] = {
        create_rule("com.xilinx:conv2d"),                //
        create_rule("com.xilinx:depthwise_conv2d"),      //
        create_rule("com.xilinx:conv2d_nchw"),           //
        create_rule("com.xilinx:depthwise_conv2d_nchw"), //
        create_rule("com.xilinx:conv1d_ncd"),            //
        create_rule("com.xilinx:depthwise_conv1d_ncd"),  //
        create_rule("com.xilinx:depthwise_conv2d_ihwo"), //
    };

    auto chain =
        BaseRule::create_rule_chain(std::vector<std::unique_ptr<BaseRule>>{
            std::make_move_iterator(std::begin(rules)),
            std::make_move_iterator(std::end(rules))});
    chain->apply(&graph);
  }
  IPass& self_;
};
} // namespace
DEFINE_VAIP_PASS(MergeAddIntoConvBias, vaip_pass_merge_add_into_conv_bias)
