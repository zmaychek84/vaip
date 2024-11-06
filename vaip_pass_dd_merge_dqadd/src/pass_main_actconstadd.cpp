/*
 *     The Xilinx Vitis AI Vaip in this distribution are provided under the
 * following free and permissive binary-only license, but are not provided in
 * source code form.  While the following free and permissive license is similar
 * to the BSD open source license, it is NOT the BSD open source license nor
 * other OSI-approved open source license.
 *
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

#ifdef __GNUC__
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wpedantic"
#  pragma GCC diagnostic ignored "-Wconversion"
#endif

#include "vaip/dd/coeffs.hpp"
#include "vaip/dd/dd_utils.hpp"

#ifdef __GNUC__
#  pragma GCC diagnostic pop
#endif
#include "vaip/pattern_zoo.hpp"
#include "vaip/vaip.hpp"
#include "vitis/ai/env_config.hpp"
#include <glog/logging.h>

DEF_ENV_PARAM(DEBUG_DD_DQADD, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_DD_DQADD) >= n)
using namespace vaip_core;

struct MergeActConstAdd {
  MergeActConstAdd(IPass& self) : self_{self} {}

  static std::vector<std::string> change_inputs(const NodeArg& a,
                                                const NodeArg& b) {
    std::vector<std::string> dtypes;
    // dtypes.emplace_back(vaip::dd::nodearg_dtype_to_string(a));
    // dtypes.emplace_back(vaip::dd::nodearg_dtype_to_string(b));
    // Add conditional code here (Below may only work for mtea0a)
    dtypes.emplace_back("bfloat16");
    dtypes.emplace_back("bfloat16");
    // dtypes.emplace_back("int32");
    return dtypes;
  }

  static std::vector<std::string> change_outputs(const NodeArg& a) {
    std::vector<std::string> dtypes;
    // Add conditional code here (Below may only work for mtea0a)
    dtypes.emplace_back("bfloat16");
    return dtypes;
  }

  std::unique_ptr<Rule> create_rule(IPass* self) {

    auto Add_0 = vaip::pattern_zoo::get_pattern("m_ActConstAdd_0");
    CHECK(Add_0 != nullptr) << "Pattern returned is null";

    return Rule::create_rule(
        Add_0, [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
          auto attr_nodes = vaip::dd::get_node_names(graph, binder);

          auto a_node = binder["input_0"];
          auto as_node = binder["constant_0"];
          auto az_node = binder["constant_1"];
          auto a_sc =
              node_arg_get_const_data_as_float(*graph, *as_node.node_arg);
          auto a_zp = vaip::dd::get_zp_from_node(*graph, *az_node.node_arg);
          auto a_shape = node_arg_get_shape_i64(*a_node.node_arg);

          auto b_node = binder["input_1"];
          auto bs_node = binder["constant_2"];
          auto bz_node = binder["constant_3"];
          auto b_shape = node_arg_get_shape_i64(*b_node.node_arg);
          auto b_sc =
              node_arg_get_const_data_as_float(*graph, *bs_node.node_arg);
          auto b_zp = vaip::dd::get_zp_from_node(*graph, *bz_node.node_arg);

          auto add_node = binder["Add_0"];

          auto add_shape = node_arg_get_shape_i64(*add_node.node_arg);
          auto node_name = node_arg_get_name(*add_node.node_arg);

          std::vector<float> input_q_params = {a_sc, float(a_zp), b_sc,
                                               float(b_zp)};

          auto b_const_in =
              node_arg_get_const_data_as_u8s(*graph, *b_node.node_arg);
          std::vector<uint8_t> b_const_vec(b_const_in.begin(),
                                           b_const_in.end());

          uint8_t bzp = uint8_t(b_zp);
          auto beta =
              vaip::dd::qmatmulcalc::dq_vec_to_bf16(b_const_vec, b_sc, (bzp));
          // auto beta_shape = node_arg_get_shape_i64(*b_node.node_arg);

          auto& input_beta_arg = vaip::dd::insert_named_tensor_in_graph(
              graph, node_name + "_beta_", beta, *b_shape);

          auto add = NodeBuilder(*graph, *self);
          add.set_input_node_args(
              // {a_node.node_arg, b_node.node_arg});//, &qdq_arg});
              {a_node.node_arg, &input_beta_arg});

          add.set_op_type("QActConstAdd", "com.xilinx");
          add.set_anchor_point1(*add_node.node);
          add.add("nodes", attr_nodes);
          add.add("in_dtypes",
                  change_inputs(*a_node.node_arg, *b_node.node_arg));
          add.add("out_dtypes", change_outputs(*add_node.node_arg));
          add.add("input1_shape", *(a_shape.get()));
          add.add("input2_shape", *(b_shape.get()));
          add.add("input_q_params", input_q_params);
          add.add("orig_output_shape", *(add_shape.get()));

          add.build();

          return true;
        });
  }

  void process(IPass& self, Graph& graph) { create_rule(&self)->apply(&graph); }

public:
  IPass& self_;
};

DEFINE_VAIP_PASS(MergeActConstAdd, vaip_pass_dd_merge_actconstadd)
