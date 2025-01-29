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
#include <chrono>
#include <future>
#include <mutex>

namespace vaip_qdq_op_custom_op {
using namespace vaip_core;
class MyCustomOp : public CustomOpImp {
public:
  MyCustomOp(std::shared_ptr<const PassContext> context,
             const std::shared_ptr<MetaDefProto>& meta_def,
             onnxruntime::Model* model);

  virtual ~MyCustomOp();

private:
  float scale;
  float zp;
  bool is_qop;
  float maxval;
  float minval;
  bool is_uint16;
  bool is_int8;
  bool is_int16;
  virtual void Compute(const OrtApi* api,
                       OrtKernelContext* context) const override final;
};

} // namespace vaip_qdq_op_custom_op
