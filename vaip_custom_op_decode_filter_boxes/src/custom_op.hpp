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

// #define FDPOST_CPU_KERNEL

// XRT includes
#ifndef FDPOST_CPU_KERNEL
#  include "../../xrt_shared_context/xrt_shared_context.hpp"
// FD Post kernel
#  include "fdpost.hpp"
#endif

namespace vaip_decode_filter_boxes_custom_op {
using namespace vaip_core;
class MyCustomOp : public CustomOpImp {
public:
  MyCustomOp(std::shared_ptr<const PassContext> context,
             const std::shared_ptr<MetaDefProto>& meta_def,
             onnxruntime::Model* model);

  virtual ~MyCustomOp();

private:
  std::vector<std::array<float, 4>> anchors_;
#ifndef FDPOST_CPU_KERNEL
  std::shared_ptr<vaip::Context> context_;
  std::unique_ptr<FDPOST> kernel_;
  xrt::bo instr_bo_;
  std::string kernel_name_;
  const uint32_t box_scale_ = 300;
#endif
  virtual void Compute(const OrtApi* api,
                       OrtKernelContext* context) const override final;
};

} // namespace vaip_decode_filter_boxes_custom_op
