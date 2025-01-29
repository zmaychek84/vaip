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
// #include <tvm/runtime/module.h>
#include <xrt/xrt_bo.h>

#include <chrono>
// #define PROFILE_MATMULINTEGER
#ifdef PROFILE_MATMULINTEGER
#  define USE_TIMER_MATMULINTEGER(timer) timer
#else
#  define USE_TIMER_MATMULINTEGER(timer)
#endif

namespace vaip_matmul_integer_custom_op {
using namespace vaip_core;
class MyCustomOp : public CustomOpImp {
public:
  MyCustomOp(std::shared_ptr<const PassContext> context,
             const std::shared_ptr<MetaDefProto>& meta_def,
             onnxruntime::Model* model);

  virtual ~MyCustomOp();

private:
  virtual void Compute(const OrtApi* api,
                       OrtKernelContext* context) const override final;
  std::shared_ptr<void> gemm_;
  std::tuple<int, int> wts_shape_;
  std::vector<int32_t> wts_sum_;
  std::string impl_;
  std::string quant_mode_;
};

} // namespace vaip_matmul_integer_custom_op
