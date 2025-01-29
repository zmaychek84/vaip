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
#include <xrt/xrt_bo.h>

#include <chrono>
#ifdef PROFILE_GMATMULINTEGER
#  define USE_TIMER_GMATMULINTEGER(timer) timer
#else
#  define USE_TIMER_GMATMULINTEGER(timer)
#endif

namespace vaip_gmatmul_integer_custom_op {
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
  std::string impl_;
  std::tuple<int, int> wts_shape_;
  std::vector<int32_t> wts_sum_;
  std::vector<int32_t> wts_shape_dim_split_;
  std::string device_;
};

} // namespace vaip_gmatmul_integer_custom_op
