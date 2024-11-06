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

#include "vaip/vaip.hpp"
#include "vitis/ai/env_config.hpp"

DEF_ENV_PARAM(DEBUG_CONST_FOLD_BATCHNOMR_TO_SCALE, "0")
#define MY_LOG(n)                                                              \
  LOG_IF(INFO, ENV_PARAM(DEBUG_CONST_FOLD_BATCHNOMR_TO_SCALE) >= n)

/**
 * test case model 5
 *
 * Replace pattern pass
 * X : wildcard()
 * From : batchnorm(input=*, gamma=fix(const()), beta=fix(const()),
 *                  moving_mean=const().where(all_zeros()), moving_var=const())
 *
 *     output = (input - moving_mean) /
 *             sqrt(moving_var + epsilon) * gamma + beta
 *
 * To  : scale(input, weights, bias)
 *
 *  where weights and bias are `fix(const())`
 */
namespace {
using namespace vaip_core;
struct BatchNormToScale {
  BatchNormToScale(IPass& self) : self_{self} {}
  std::unique_ptr<Rule> create_rule() {
    auto builder = PatternBuilder();
    std::shared_ptr<Pattern> input_ = builder.wildcard();
    std::shared_ptr<Pattern> gamma_ =
        builder.node2("com.xilinx:fix", {builder.xir_const_op()});
    std::shared_ptr<Pattern> beta_ =
        builder.node2("com.xilinx:fix", {builder.xir_const_op()});
    std::shared_ptr<Pattern> moving_mean_ = builder.xir_const_op();
    std::shared_ptr<Pattern> moving_var_ = builder.xir_const_op();
    std::shared_ptr<Pattern> pattern_ =
        builder.node2("com.xilinx:batchnorm",
                      {input_, gamma_, beta_, moving_mean_, moving_var_});
    return Rule::create_rule(
        pattern_, [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
          auto input = binder[input_->get_id()];
          auto gamma = binder[gamma_->get_id()];
          auto beta = binder[beta_->get_id()];
          auto moving_mean = binder[moving_mean_->get_id()];
          auto moving_var = binder[moving_var_->get_id()];
          auto pattern = binder[pattern_->get_id()];
          CHECK(input.node != nullptr);
          CHECK(gamma.node != nullptr);
          CHECK(beta.node != nullptr);
          CHECK(moving_mean.node != nullptr);
          CHECK(moving_var.node != nullptr);
          CHECK(pattern.node != nullptr);

          auto moving_var_data = self_.get_const_data<float>(*moving_var.node);
          auto epsilon = node_get_attr_float(*pattern.node, "epsilon");
          if (epsilon != 0.0f) {
            MY_LOG(1)
                << "TODO: epision is not zero, it is not supported yet. cancel "
                   "constant folding";
            return false;
          }
          auto all_ones =
              std::all_of(moving_var_data.begin(), moving_var_data.end(),
                          [](float f) { return f == 1.0f; });
          if (!all_ones) {
            MY_LOG(1) << "TODO: moving_var are not all ones. it is not "
                         "supported yet. cancel "
                         "constant folding";
            return false;
          }

          auto moving_mean_data =
              self_.get_const_data<float>(*moving_mean.node);
          auto all_zeros =
              std::all_of(moving_mean_data.begin(), moving_mean_data.end(),
                          [](float f) { return f == 0.0f; });
          if (!all_zeros) {
            MY_LOG(1) << "moving_mean are not all zeros. cancal const folding";
            return false;
          }
          NodeBuilder(*graph, self_)
              .set_op_type("scale")
              .set_input_node_args(
                  {input.node_arg, gamma.node_arg, beta.node_arg})
              .clone_data_type(*pattern.node)
              .clone_shape(*pattern.node)
              .add("axis", node_get_attr_int(*pattern.node, "axis"))
              .set_anchor_point1(*pattern.node)
              .build();
          return true;
        });
  }
  void process(IPass& self, Graph& graph) { create_rule()->apply(&graph); }

  IPass& self_;
};
} // namespace

DEFINE_VAIP_PASS(BatchNormToScale, vaip_pass_const_fold_batchnorm_to_scale)