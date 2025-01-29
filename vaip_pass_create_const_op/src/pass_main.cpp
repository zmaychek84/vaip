/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

#include "vaip/vaip.hpp"
#include <cstdint>
#include <glog/logging.h>

#include "vitis/ai/env_config.hpp"
#include <vitis/ai/dim_calc.hpp>

DEF_ENV_PARAM(XLNX_ENABLE_DUMP_CONSTANT, "0")

using namespace vaip_core;
namespace vaip_pass_create_const_op {
void create_const_ops(IPass& pass, Graph& graph);
} // namespace vaip_pass_create_const_op
using namespace vaip_pass_create_const_op;
#include "./const_fold_rule.hpp"

#include "ops/_common.hpp"
#include "ops/add.hpp"
#include "ops/dequantize_linear.hpp"
#include "ops/div.hpp"
#include "ops/fix_neuron.hpp"
#include "ops/gather.hpp"
#include "ops/mul.hpp"
#include "ops/quantize_linear.hpp"
#include "ops/reshape.hpp"
#include "ops/shape.hpp"
#include "ops/transpose.hpp"

struct ConstantFoldingPass {
  ConstantFoldingPass(IPass& self) {}
  void preprocess(IPass& self, Graph& graph) {
    vaip_pass_create_const_op::create_const_ops(self, graph);
    std::unique_ptr<BaseRule> rules[] = {
        DequantizeLinear(self),       DequantizeLinear_int32_t(self),
        QuantizeLinear(self),         FixNeuron(self),
        Reshape<int8_t, float>(self), Transpose<int8_t, float>(self),
        Gather<int64_t>(self),        Add<float, int64_t>(self),
        Div<float, int64_t>(self),    Mul<float, int64_t>(self),
    };
    chain_ = BaseRule::create_rule_chain(std::vector<std::unique_ptr<BaseRule>>{
        std::make_move_iterator(std::begin(rules)),
        std::make_move_iterator(std::end(rules))});
    if (ENV_PARAM(XLNX_ENABLE_DUMP_CONSTANT)) {
      self.dump_const_info("const_info_before_const_folding.txt");
    }
  }

  bool process(IPass& self, Graph& graph, const Node& node) {
    return chain_->apply_once(&graph, &node);
  }
  void postprocess(IPass& self, Graph& graph) {
    if (ENV_PARAM(XLNX_ENABLE_DUMP_CONSTANT)) {
      self.dump_fix_info("fix_info.txt");
      self.dump_const_info("const_info_after_const_folding.txt");
      self.dump_const_data("const.bin");
    }
  }
  ~ConstantFoldingPass() {}
  std::unique_ptr<BaseRule> chain_;
};

DEFINE_VAIP_PASS(ConstantFoldingPass, vaip_pass_create_const_op)
