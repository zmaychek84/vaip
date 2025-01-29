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

namespace vaip_gather_add_custom_op {
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

  int data_shape_;
  int indeces_shape_;
  int ifm_dim_0_;
  int ifm_dim_1_;
  std::vector<uint8_t> in_data_;
  std::vector<float> wts_data_;
  std::vector<int64_t> in_indeces_;
  float act_scale_, wt_scale_;
  int act_zp_, wt_zp_;
};

} // namespace vaip_gather_add_custom_op