/*
 *     The Xilinx Vitis AI Vaip in this distribution are provided under the
 * following free and permissive binary-only license, but are not provided in
 * source code form.  While the following free and permissive license is similar
 * to the BSD open source license, it is NOT the BSD open source license nor
 * other OSI-approved open source license.
 *
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
#include <glog/logging.h>

#ifdef __GNUC__
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wpedantic"
#  pragma GCC diagnostic ignored "-Wconversion"
#endif

#include "vaip/dd/coeffs.hpp"
#include "vaip/dd/dd_utils.hpp"
#include "vaip/pattern_zoo.hpp"

#ifdef __GNUC__
#  pragma GCC diagnostic pop
#endif

#include "vaip/vaip.hpp"
#include "vitis/ai/env_config.hpp"

DEF_ENV_PARAM(DEBUG_DD_MERGE_QRESIZE, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_DD_MERGE_QRESIZE) >= n)

/**
 * test case: <???>
 *
 *
 * Replace pattern:
 *
 * From: <???>
 * To  : <???>
 */

// add the following line in your vaip_config.json
/*
    { "name": "vaip_pass_dd_merge_qresize",
       "plugin": "vaip-pass_dd_merge_qresize",
       "disabled": false
    }
*/
namespace {
using namespace vaip_core;
struct Dd_merge_qresize {
  Dd_merge_qresize(IPass& self) : self_{self} {}
  std::unique_ptr<Rule> create_rule(IPass* self) {
    auto com_microsoft_QuantizeLinear_0 =
        vaip::pattern_zoo::get_pattern("m_qresize");
    CHECK(com_microsoft_QuantizeLinear_0 != nullptr)
        << "Pattern returned is null";
    return Rule::create_rule(
        com_microsoft_QuantizeLinear_0,
        [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
          auto in_node = binder["input_0"];
          auto resize_node = binder["Resize_0"];
          auto out_node = binder["com_microsoft_QuantizeLinear_0"];

          std::vector<std::string> ns = vaip::dd::get_node_names(graph, binder);
          MY_LOG(1) << "found match at " << ns.front();

          std::vector<uint8_t> coeffs(16, 0);
          auto node_name = node_arg_get_name(*out_node.node_arg);
          std::string coeff_name = std::string(node_name + "_qdq_");
          auto& qdq_arg = vaip::dd::insert_named_tensor_in_graph(
              graph, coeff_name, coeffs, std::vector({(int64_t)coeffs.size()}));

          // hard code for mzdk5, may need to change
          std::vector<std::string> input_types{"uint16", "uint8"};
          std::vector<std::string> output_types{"uint16"};

          NodeBuilder(*graph, *self)
              .set_input_node_args({in_node.node_arg, &qdq_arg})
              .set_op_type("QResize", "com.xilinx")
              .clone_attrs(*resize_node.node)
              .add("nodes", ns)
              .add("in_dtypes", input_types)
              .add("out_dtypes", output_types)
              .set_anchor_point1(*out_node.node)
              .build();
          return true;
        });
  }
  // apply the rule
  void process(IPass& self, Graph& graph) {
    MY_LOG(1) << self_.get_pass_proto().name() << "["
              << self_.get_pass_proto().plugin() << "] start processing graph";
    create_rule(&self)->apply(&graph);
    MY_LOG(1) << self_.get_pass_proto().name() << "["
              << self_.get_pass_proto().plugin() << "] finish processing graph";
  }

  IPass& self_;
};
} // namespace

DEFINE_VAIP_PASS(Dd_merge_qresize, vaip_pass_dd_merge_qresize)
