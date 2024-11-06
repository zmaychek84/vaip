/*
 *     The Xilinx Vitis AI Vaip in this distribution are provided under the
 * following free and permissive binary-only license, but are not provided in
 * source code form.  While the following free and permissive license is similar
 * to the BSD open source license, it is NOT the BSD open source license nor
 * other OSI-approved open source license.
 *
 *      Copyright (C) 2022 Xilinx, Inc.
 *      Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights
 * reserved.
 *
 *      Redistribution and use in binary form only, without modification, is
 * permitted provided that the following conditions are met:
 *
 *      1. Redistributions must reproduce the above copyright notice, this list
 * of conditions and the following disclaimer in the documentation and/or other
 * materials provided with the distribution.
 *
 *      2. The name of Xilinx, Inc. may not be used to endorse or promote
 * products redistributed with this software without specific prior written
 * permission.
 *
 *      THIS SOFTWARE IS PROVIDED BY XILINX, INC. "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL XILINX, INC. BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 *      PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE
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
