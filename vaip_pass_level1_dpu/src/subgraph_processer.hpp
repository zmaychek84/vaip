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

#pragma once
// clang-format off
#include "vaip/vaip.hpp"
#include "vaip/dpu_sg_report.pb.h"
#include "vaip/xir_headers.hpp"
#include <unordered_map>
// clang-format on

namespace vaip_level1_dpu {
using namespace vaip_core;

struct ProcessInfo {
  using ap_ptr_v = std::vector<std::unique_ptr<AnchorPoint>>;
  ProcessInfo(bool status) { this->status = status; }
  ProcessInfo(std::vector<std::string> inputs, std::vector<std::string> outputs,
              std::vector<TensorBufferParam> xir_input_tbs,
              ap_ptr_v& xir_input_anchor_points,
              std::vector<TensorBufferParam> xir_output_tbs,
              ap_ptr_v& xir_output_anchor_points)
      : inputs(inputs), outputs(outputs), xir_input_tbs(xir_input_tbs),
        xir_output_tbs(xir_output_tbs) {
    this->xir_input_anchor_points =
        std::make_unique<ap_ptr_v>(std::move(xir_input_anchor_points));
    this->xir_output_anchor_points =
        std::make_unique<ap_ptr_v>(std::move(xir_output_anchor_points));
    this->status = true;
  }
  ProcessInfo() = default;
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
  std::vector<TensorBufferParam> xir_input_tbs;
  std::unique_ptr<ap_ptr_v> xir_input_anchor_points;
  std::vector<TensorBufferParam> xir_output_tbs;
  std::unique_ptr<ap_ptr_v> xir_output_anchor_points;
  bool status;
};
class DPUSubgraphProcessor {
public:
  DPUSubgraphProcessor(onnxruntime::Graph& graph, IPass& self,
                       xir::Graph* xir_xmodel, xir::Graph* compiled_xmodel);
  DPUSubgraphProcessor() = delete;
  ProcessInfo find_xir_anchor_point(const xir::Subgraph* subgraph);
  std::unique_ptr<MetaDefProto> process(const xir::Subgraph* subgraph,
                                        const ProcessInfo& above_context);
  std::unique_ptr<MetaDefProto>
  process_internal(const xir::Subgraph* subgraph,
                   const ProcessInfo& above_context);
  DpuSubgraphEntryProto* get_proto() { return &report_; };

private:
  bool is_shape_compatible(const std::string& tensor_name_on_xir_xmodel,
                           const std::string& tensor_name_on_compiled_xmodel);
  std::vector<std::unique_ptr<AnchorPoint>> create_anchor_points_by_xir_tensors(
      const IPass& pass, const onnxruntime::Graph& graph,
      const std::vector<const xir::Tensor*>& tensors,
      TensorBufferType tenor_type, const xir::Subgraph* subgraph);
  std::unique_ptr<AnchorPoint> create_anchor_point_by_xir_tensor(
      const IPass& pass, const onnxruntime::Graph& onnx_dot_onnx_graph,
      const xir::Tensor* tensor, TensorBufferType tenor_type,
      const xir::Subgraph* subgraph);
  ProcessInfo find_xir_anchor_point2(const xir::Subgraph* subgraph);
  bool
  check_4d_dimention_tensors(const std::vector<const xir::Tensor*>& tensors);
  bool check_4d_dimention(const xir::Subgraph* subgraph);

private:
  onnxruntime::Graph& onnx_graph_;
  IPass& self_;
  xir::Graph* xir_xmodel_;
  xir::Graph* compiled_xmodel_;
  DpuSubgraphEntryProto report_;
  std::stringstream report_comment_;
};
} // namespace vaip_level1_dpu
