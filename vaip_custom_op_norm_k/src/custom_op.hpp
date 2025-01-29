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

#include <xir/graph/graph.hpp>

namespace Ort {
struct KernelContext;
}

namespace vaip_norm_k_custom_op {
using namespace vaip_core;

class MyCustomOp : public CustomOpImp {

public:
  MyCustomOp(std::shared_ptr<const PassContext> context,
             const std::shared_ptr<MetaDefProto>& meta_def,
             onnxruntime::Model* model);

  virtual ~MyCustomOp();

public:
private:
  virtual void Compute(const OrtApi* api,
                       OrtKernelContext* context) const override final;

private:
  std::vector<uint16_t> pre_computed_output;
};

} // namespace vaip_norm_k_custom_op
