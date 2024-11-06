/*
 *     The Xilinx Vitis AI Vaip in this distribution are provided under the
 * following free and permissive binary-only license, but are not provided in
 * source code form.  While the following free and permissive license is similar
 * to the BSD open source license, it is NOT the BSD open source license nor
 * other OSI-approved open source license.
 *
 *      Copyright (C) 2022 Xilinx, Inc. All rights reserved.
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

#include "onnxruntime_api.hpp"
#include <glog/logging.h>
//
#include "vaip/vaip.hpp"

#include <functional>
#include <memory>
#include <ostream>
#include <vart/tensor_buffer.hpp>

#include "vart/runner_ext.hpp"
#include <vector>

namespace vaip_dpu_custom_op {

std::shared_ptr<vart::TensorBuffer>
create_onnx_input_tensor_buffer(const std::string& node_arg_name,
                                Ort::KernelContext& context, int idx);

std::shared_ptr<vart::TensorBuffer>
create_onnx_output_tensor_buffer(const std::string& node_arg_name,
                                 Ort::KernelContext& context, int idx,
                                 std::vector<int64_t> onnx_shape);

std::shared_ptr<vart::TensorBuffer> create_xir_tensor_buffer(
    const std::string& tensor_name,
    const std::vector<vart::TensorBuffer*>& tensor_buffers);

void trans_data(std::shared_ptr<vart::TensorBuffer> from,
                std::shared_ptr<vart::TensorBuffer> to,
                const vaip_core::DataOperator& op, int64_t onnx_batch);

// debug helper functions
inline std::ostream& operator<<(std::ostream& s,
                                const std::vector<int64_t>& v) {
  s << "[";
  for (auto c = 0u; c < v.size(); ++c) {
    if (c != 0) {
      s << ",";
    }
    s << v[c];
  }
  s << "] ";
  return s;
}
inline std::ostream& operator<<(std::ostream& s,
                                const std::vector<int32_t>& v) {
  s << "[";
  for (auto c = 0u; c < v.size(); ++c) {
    if (c != 0) {
      s << ",";
    }
    s << v[c];
  }
  s << "] ";
  return s;
}

} // namespace vaip_dpu_custom_op
