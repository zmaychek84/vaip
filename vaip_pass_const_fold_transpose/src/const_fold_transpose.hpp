/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

/**
 * ConstFoldTranspose pattern pass
 * X : wildcard()
 * From : matmul(input_a=*, input_b=fix(transpose(fix(const()))),
 *               transpose_a=0, transpose_b=0)
 *
 * To  : matmul(input_a=*, input_b=fix(const()),
 *              transpose_a=0, transpose_b=1)
 *
 */
#pragma once
#include "vaip/vaip.hpp"
namespace vaip_pass_const_fold_transpose {
using namespace vaip_core;
class ConstFoldTransposeRule : public Rule {
public:
  ConstFoldTransposeRule();

private:
  virtual const Pattern* pattern() const override;
  virtual bool action(onnxruntime::Graph* graph,
                      binder_t& binder) const override;

private:
  std::shared_ptr<Pattern> input_weight_;     // com.xilinx:const
  std::shared_ptr<Pattern> input_weight_fix_; // com.xilinx:fix
  std::shared_ptr<Pattern> transpose_;        // com.xilinx:transpose
  std::shared_ptr<Pattern> transpose_fix_;    // com.xilinx:fix
  std::shared_ptr<Pattern> input_a_;          // wildcard()
  std::shared_ptr<Pattern> matmul_;           // com.xilinx:matmul
};
} // namespace vaip_pass_const_fold_transpose
