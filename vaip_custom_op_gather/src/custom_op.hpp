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

namespace vaip_gather_custom_op {
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

  int data_shape;
  int indeces_shape;
  int64_t indeces;
  int ifm_dim_0;
  int ifm_dim_1;
  std::vector<uint8_t> in_data;
  std::vector<int64_t> in_indeces;
  int is_const; // is indices input to gather constant flag
};

} // namespace vaip_gather_custom_op