/*
 *     The Xilinx Vitis AI Vaip in this distribution are provided under the
 * following free and permissive binary-only license, but are not provided in
 * source code form.  While the following free and permissive license is similar
 * to the BSD open source license, it is NOT the BSD open source license nor
 * other OSI-approved open source license.
 *
# Copyright (C) 2022 Xilinx, Inc.
 *      Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc.
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
#include "vaip/vaip.hpp"
#include <cstdint>
#include <glog/logging.h>
#include <string>
#include <unordered_map>
#include <vector>
namespace vaip_dd_merge_qma {
using namespace vaip_core;
struct DdMergeQmhaProcessor {
public:
  DdMergeQmhaProcessor(
      IPass& self, onnxruntime::Graph* graph, binder_t* binder,
      const std::unordered_map<std::string, std::vector<std::string>>&
          binder_params);
  const NodeArg& process(int output_pat_id);
  const NodeArg& process_m7h4xjg(int output_pat_id);

private:
  float node_arg_get_const_data_as_float(const std::string& name, size_t index);
  uint16_t node_arg_get_const_data_as_u16(const std::string& name,
                                          size_t index);
  int64_t get_k_dim(NodeInput ni1, NodeInput ni2);
  const NodeArg& create_node_arg(onnxruntime::Graph& graph,
                                 const std::string& name,
                                 const std::vector<int64_t>& shape,
                                 const std::vector<int64_t>& value);

  NodeInput get_node_input(const std::string& name, size_t index) const;

private:
  IPass& self_;
  onnxruntime::Graph* graph_;
  binder_t* binder_;
  const std::unordered_map<std::string, std::vector<std::string>>&
      binder_params_;
};
} // namespace vaip_dd_merge_qma
