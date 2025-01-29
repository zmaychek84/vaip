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

namespace vaip_dqcastgather_custom_op {
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

  int8_t zp;
  float scale;
  int data_shape;
  int indeces_shape;
  int32_t indeces;
  int ifm_dim_0;
  int ifm_dim_1;
  std::vector<float> in_dq;
  std::vector<int8_t> in_data;
};

} // namespace vaip_dqcastgather_custom_op