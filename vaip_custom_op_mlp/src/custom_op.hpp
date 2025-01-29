/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
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
