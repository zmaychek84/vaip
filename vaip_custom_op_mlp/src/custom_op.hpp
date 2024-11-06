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

/*

**/
#pragma once

#include "vaip/vaip.hpp"
#include "vart/runner_ext.hpp"
#include <algorithm>
#include <future>
#include <mutex>

#include <ryzenai/dynamic_dispatch/ops/elwmul/elwmul.hpp>
#include <ryzenai/dynamic_dispatch/ops/mladfmatmulbias/mladfmatmulbias.hpp>
#include <ryzenai/dynamic_dispatch/ops/silu/silu.hpp>

namespace vaip_mlp_custom_op {
using namespace vaip_core;

static std::shared_ptr<void> ewmul_;
static std::shared_ptr<void> silu_;
static std::shared_ptr<void> gate_proj_;
static std::shared_ptr<void> up_proj_;
static std::shared_ptr<void> down_proj_;

class MyCustomOp : public CustomOpImp {
public:
  MyCustomOp(std::shared_ptr<const PassContext> context,
             const std::shared_ptr<MetaDefProto>& meta_def,
             onnxruntime::Model* model);

  virtual ~MyCustomOp();

private:
  virtual void Compute(const OrtApi* api,
                       OrtKernelContext* context) const override final;

  template <typename dtype>
  std::vector<dtype> loadbin(std::string& fname) const;

private:
  uint16_t* input_data_{nullptr};
  int cnt_;
  bool dry_run_;
  // Kernels
  // std::shared_ptr<void> ewmul_;
  // std::shared_ptr<void> silu_;
  // std::shared_ptr<void> gate_proj_;
  // std::shared_ptr<void> up_proj_;
  // std::shared_ptr<void> down_proj_;
  // Vars
  int64_t gp_k, gp_n, gp_bits;
  int gp_block_size;
  int64_t up_k, up_n, up_bits;
  int up_block_size;
  int64_t dp_k, dp_n, dp_bits;
  int dp_block_size;
  // Initialize the static instance pointer to nullptr
  // ReportInventory* ReportInventory::instance = nullptr;
};

} // namespace vaip_mlp_custom_op
