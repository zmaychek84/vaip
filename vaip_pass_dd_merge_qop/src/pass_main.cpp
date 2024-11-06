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

DEF_ENV_PARAM(DEBUG_DD_MERGE_QOP, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_DD_MERGE_QOP) >= n)

template <typename Type> std::string to_str(const Type& t) {
  std::ostringstream os;
  os << t;
  return os.str();
}
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
    { "name": "vaip_pass_dd_merge_quant",
       "plugin": "vaip-pass_dd_merge_quant",
       "disabled": false
    }
*/
namespace {
using namespace vaip_core;
struct Dd_merge_qop {
  Dd_merge_qop(IPass& self) : self_{self} {}
  std::unique_ptr<Rule> create_rule(IPass* self) {
    auto dq = vaip::pattern_zoo::get_pattern("m_q");
    CHECK(dq != nullptr) << "Pattern returned is null";
    return Rule::create_rule(
        dq, [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
          auto in_node = binder["input"];
          auto in_scale_node = binder["in_s"];
          auto in_zp_node = binder["in_z"];
          auto out_node = binder["dq"];

          auto in_scale =
              node_arg_get_const_data_as_float(*graph, *in_scale_node.node_arg);
          auto zp_dtype = node_arg_get_element_type(*in_zp_node.node_arg);
          float in_zero_point;
          if (zp_dtype == (int)ONNX_NAMESPACE::TensorProto_DataType_INT8) {
            in_zero_point = (float)tensor_proto_as_i8(
                *graph, node_arg_get_const_data_as_tensor(
                            *graph, *in_zp_node.node_arg));
          } else if (zp_dtype ==
                     (int)ONNX_NAMESPACE::TensorProto_DataType_INT16) {
            in_zero_point = (float)tensor_proto_as_i16(
                *graph, node_arg_get_const_data_as_tensor(
                            *graph, *in_zp_node.node_arg));
          } else {
            in_zero_point =
                (float)vaip::dd::get_zp_from_node(*graph, *in_zp_node.node_arg);
          }
          std::vector<float> input_q_params{in_scale, float(in_zero_point)};
          auto node_name = node_arg_get_name(*out_node.node_arg);

          std::vector<std::string> inputs = {
              node_arg_get_name(*in_node.node_arg),
              node_arg_get_name(*in_scale_node.node_arg),
              node_arg_get_name(*in_zp_node.node_arg)};
          std::vector<std::string> outputs = {node_name};
          auto constant_initializers = std::vector<std::string>{};
          auto [meta_def, err] =
              self->try_fuse(*graph, "qop_" + node_name, inputs, outputs,
                             constant_initializers, "QDQ_OP");
          if (meta_def == nullptr) {
            LOG(FATAL) << "Cannot fuse DQ:  " << err.comments;
          }
          auto& generic_param = *meta_def->mutable_generic_param();

          generic_param["in_scale"] = to_str(in_scale);
          generic_param["in_zero_point"] = std::to_string(in_zero_point);
          generic_param["is_qop"] = "1";
          if (zp_dtype == (int)ONNX_NAMESPACE::TensorProto_DataType_UINT8) {
            // std::cout << "Uint8 matched" << std::endl;
            generic_param["zp_dtype"] = "uint8";
          }
          if (zp_dtype == (int)ONNX_NAMESPACE::TensorProto_DataType_UINT16) {
            // std::cout << "Uint16 matched" << std::endl;
            generic_param["zp_dtype"] = "uint16";
          }
          if (zp_dtype == (int)ONNX_NAMESPACE::TensorProto_DataType_INT8) {
            // std::cout << "int8 matched" << std::endl;
            generic_param["zp_dtype"] = "int8";
          }
          if (zp_dtype == (int)ONNX_NAMESPACE::TensorProto_DataType_INT16) {
            // std::cout << "int16 matched" << std::endl;
            generic_param["zp_dtype"] = "int16";
          }

          [[maybe_unused]] auto& fused_node =
              self->fuse(*graph, std::move(*meta_def));

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

DEFINE_VAIP_PASS(Dd_merge_qop, vaip_pass_dd_merge_qop)
