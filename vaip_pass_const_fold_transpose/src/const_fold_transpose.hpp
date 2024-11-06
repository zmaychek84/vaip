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
