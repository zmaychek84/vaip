/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
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
