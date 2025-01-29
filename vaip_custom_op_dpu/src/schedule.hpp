/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
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
