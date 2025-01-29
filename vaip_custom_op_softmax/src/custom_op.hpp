/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */
#pragma once

#include "vaip/vaip.hpp"
#include "vart/runner_ext.hpp"

#include <algorithm>
#include <future>
#include <mutex>
class SoftMax;

// XRT includes
#include "../../xrt_shared_context/xrt_shared_context.hpp"
// FD Post kernel

namespace vaip_softmax_custom_op {
using namespace vaip_core;
class MyCustomOp : public CustomOpImp {
public:
  MyCustomOp(std::shared_ptr<const PassContext> context,
             const std::shared_ptr<MetaDefProto>& meta_def,
             onnxruntime::Model* model);

  virtual ~MyCustomOp();

private:
  std::shared_ptr<vaip::Context> context_;
  std::unique_ptr<SoftMax> kernel_softmax_;
  xrt::bo instr_bo_softmax_;
  std::string kernel_name_softmax_;
  std::vector<int64_t> input_shape_;
  std::vector<int64_t> output_shape_;
  // std::vector<int64_t> target_shape_;
  bool en_transpose_{false};

  virtual void Compute(const OrtApi* api,
                       OrtKernelContext* context) const override final;
};

} // namespace vaip_softmax_custom_op
