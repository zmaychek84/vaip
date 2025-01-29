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

namespace tvm {
namespace runtime {
class Module;
}
} // namespace tvm

using TvmModule = ::tvm::runtime::Module;
using AsrTensorShape = std::vector<int64_t>;
using AsrShapeVec = std::vector<AsrTensorShape>;

namespace vaip_asr_custom_op {
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
  void RunWithoutOutputCopy(Ort::KernelContext& context) const;
  void RunWithOutputCopy(Ort::KernelContext& context) const;
  AsrShapeVec ort_input_shapes_;
  AsrShapeVec ort_output_shapes_;
  std::shared_ptr<void> asr_mod_holder_;
  TvmModule* asr_mod_;
  bool output_zero_copy_ = false;
};

} // namespace vaip_asr_custom_op
