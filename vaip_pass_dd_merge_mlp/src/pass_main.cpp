/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
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

DEF_ENV_PARAM(DEBUG_DD_MERGE_MLP, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_DD_MERGE_MLP) >= n)

#define MLADF_VERSION "v1"
#define MAX_SEQ_LENGTH 3072

namespace {
using namespace vaip_core;
static bool first_mm = true;

struct Dd_merge_mlp {
  Dd_merge_mlp(IPass& self) : self_{self} {}
  std::unique_ptr<Rule> create_rule(IPass* self) {

    auto builder = PatternBuilder();
    auto p_input = builder.wildcard();

    // Cast primary
    auto cast_gp = builder.node2("Cast", {p_input});
    auto cast_up = builder.node2("Cast", {p_input});

    // Gate projection constants
    auto gate_wts = builder.wildcard();
    auto gate_scl = builder.wildcard();
    // Up projection constants
    auto up_wts = builder.wildcard();
    auto up_scl = builder.wildcard();

    // Gate projection MatMul
    auto gate_proj =
        builder.node3("com.microsoft:MatMulNBits",
                      {cast_gp, gate_wts, gate_scl}, {false, false, false});
    // Up projection MatMul
    auto up_proj =
        builder.node3("com.microsoft:MatMulNBits", {cast_up, up_wts, up_scl},
                      {false, false, false});
    // Cast gp output
    auto cast_gp_out = builder.node2("Cast", {gate_proj});
    // Cast up output
    auto cast_up_out = builder.node2("Cast", {up_proj});

    //// GP down
    auto cast_gp_left = builder.node2("Cast", {cast_gp_out});
    auto cast_gp_right = builder.node2("Cast", {cast_gp_out});

    // Sigmoid
    auto sigmoid = builder.node2("Sigmoid", {cast_gp_right});
    auto sigmoid_bf16 = builder.node2("Cast", {sigmoid});
    auto sigmoid_fp32 = builder.node2("Cast", {sigmoid_bf16});
    // Mul
    auto mul_1 = builder.node2("Mul", {cast_gp_left, sigmoid_fp32});
    auto mul_1_bf16 = builder.node2("Cast", {mul_1});
    auto mul_1_fp32 = builder.node2("Cast", {mul_1_bf16});
    // Up out cast fp32
    auto cast_up_fp32 = builder.node2("Cast", {cast_up_out});

    // Mul
    auto mul_2 = builder.commutable_node("Mul", mul_1_fp32, cast_up_fp32);
    auto mlp_output = builder.node2("Cast", {mul_2});

    CHECK(mlp_output != nullptr) << "Pattern returned is null";
    return Rule::create_rule(
        mlp_output, [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
          auto p_input_node = binder[p_input->get_id()];
          // Gate projection constants
          auto gate_wts_node = binder[gate_wts->get_id()];
          auto gate_scl_node = binder[gate_scl->get_id()];
          // Up projection constants
          auto up_wts_node = binder[up_wts->get_id()];
          auto up_scl_node = binder[up_scl->get_id()];
          // MatMuls
          auto gate_proj_node = binder[gate_proj->get_id()];
          auto up_proj_node = binder[up_proj->get_id()];
          // Output
          auto out_node = binder[mlp_output->get_id()];

          MY_LOG(1) << "Pattern matched mlp at "
                    << node_arg_get_name(*out_node.node_arg) << std::endl;

          // shapes
          auto in_shape = node_arg_get_shape_i64(*p_input_node.node_arg);
          auto out_shape = node_arg_get_shape_i64(*out_node.node_arg);
          // consts weights
          //  auto g_w_shape = node_arg_get_shape_i64(*gate_wts_node.node_arg);
          //  auto g_w_shape_vec = *(g_w_shape.get());
          auto gate_weights =
              node_arg_get_const_data_as_u8s(*graph, *gate_wts_node.node_arg);
          std::vector<uint8_t> gate_wts_vec(gate_weights.begin(),
                                            gate_weights.end());
          std::string gate_wts_name =
              node_arg_get_name(*gate_wts_node.node_arg) + "0";

          // std::cout << "gate wts vector shape = " << gate_wts_vec.size()
          // <<std::endl;
          auto& gate_wts_arg = vaip::dd::insert_named_tensor_in_graph<uint8_t>(
              graph, gate_wts_name, gate_wts_vec,
              std::vector<int64_t>{(int64_t)gate_wts_vec.size()});

          auto up_weights =
              node_arg_get_const_data_as_u8s(*graph, *up_wts_node.node_arg);
          std::vector<uint8_t> up_wts_vec(up_weights.begin(), up_weights.end());
          std::string up_wts_name =
              node_arg_get_name(*up_wts_node.node_arg) + "0";
          auto& up_wts_arg = vaip::dd::insert_named_tensor_in_graph<uint8_t>(
              graph, up_wts_name, up_wts_vec,
              std::vector<int64_t>{(int64_t)up_wts_vec.size()});

          // scales
          auto gate_scales = node_arg_get_const_data_as_floats(
              *graph, *gate_scl_node.node_arg);
          std::vector<float> gate_scl_vec(gate_scales.begin(),
                                          gate_scales.end());
          std::string gate_scl_name =
              node_arg_get_name(*gate_scl_node.node_arg) + "0";
          auto& gate_scl_arg = vaip::dd::insert_named_tensor_in_graph<float>(
              graph, gate_scl_name, gate_scl_vec,
              std::vector<int64_t>{(int64_t)gate_scl_vec.size()});

          auto up_scales =
              node_arg_get_const_data_as_floats(*graph, *up_scl_node.node_arg);
          std::vector<float> up_scl_vec(up_scales.begin(), up_scales.end());
          std::string up_scl_name =
              node_arg_get_name(*up_scl_node.node_arg) + "0";
          auto& up_scl_arg = vaip::dd::insert_named_tensor_in_graph<float>(
              graph, up_scl_name, up_scl_vec,
              std::vector<int64_t>{(int64_t)up_scl_vec.size()});

          // zeros
          std::vector<uint8_t> gate_zeros_vec(gate_scl_vec.size() / 2, 0);
          std::string gate_zeros_name =
              node_arg_get_name(*gate_proj_node.node_arg) + "_zeros";
          auto& gate_zeros_arg =
              vaip::dd::insert_named_tensor_in_graph<uint8_t>(
                  graph, gate_zeros_name, gate_zeros_vec,
                  std::vector<int64_t>{(int64_t)gate_zeros_vec.size()});

          std::vector<uint8_t> up_zeros_vec(up_scl_vec.size() / 2, 0);
          std::string up_zeros_name =
              node_arg_get_name(*up_proj_node.node_arg) + "_zeros";
          auto& up_zeros_arg = vaip::dd::insert_named_tensor_in_graph<uint8_t>(
              graph, up_zeros_name, up_zeros_vec,
              std::vector<int64_t>{(int64_t)up_zeros_vec.size()});

          // bias
          std::vector<float> gate_bias_vec((*(out_shape.get()))[2], 0);
          std::string gate_bias_name =
              node_arg_get_name(*gate_proj_node.node_arg) + "_bias";
          auto& gate_bias_arg = vaip::dd::insert_named_tensor_in_graph<float>(
              graph, gate_bias_name, gate_bias_vec,
              std::vector<int64_t>{(int64_t)gate_bias_vec.size()});

          std::vector<float> up_bias_vec((*(out_shape.get()))[2], 0);
          std::string up_bias_name =
              node_arg_get_name(*up_proj_node.node_arg) + "_bias";
          auto& up_bias_arg = vaip::dd::insert_named_tensor_in_graph<float>(
              graph, up_bias_name, up_bias_vec,
              std::vector<int64_t>{(int64_t)up_bias_vec.size()});

          std::vector<std::string> in_dtypes{"bfloat16", "uint8", "float",
                                             "uint8",    "float", "uint8",
                                             "float",    "uint8", "float"};
          std::vector<std::string> out_dtypes{"bfloat16"};
          //   std::cout <<"out shape = " << (*(out_shape.get()))[0] <<" "<<
          //   (*(out_shape.get()))[1] <<" "<< (*(out_shape.get()))[2]
          //   <<std::endl; int M = (*(out_shape.get()))[1]; int N =
          //   (*(out_shape.get()))[2]; int K = (*(in_shape.get()))[2];
          //   std::vector<uint64_t>MKN_shape{static_cast<uint64_t>(M),
          //         static_cast<uint64_t>(K), static_cast<uint64_t>(N)};
          //   std::vector<int>MKN_shape{M,K,N};
          std::string mladf_version_(MLADF_VERSION);
          int32_t default_shape = 1;
          int32_t max_seq_len = MAX_SEQ_LENGTH;
          std::vector<int64_t>& input_shape = *in_shape;
          //   for(int i=0; i<input_shape.size(); i++)
          //     {
          //         std::cout<<input_shape[i]<<std::endl;
          //         std::cout<<sizeof(input_shape[i])<<std::endl;
          //     }
          input_shape[0] = input_shape[1];
          input_shape[1] = input_shape[2];
          input_shape[2] = (*out_shape)[2];

          NodeBuilder(*graph, *self)
              .set_input_node_args({p_input_node.node_arg, &gate_wts_arg,
                                    &gate_scl_arg, &gate_zeros_arg,
                                    &gate_bias_arg, &up_wts_arg, &up_scl_arg,
                                    &up_zeros_arg, &up_bias_arg})
              .set_op_type("FlatMLP", "com.xilinx")
              // .add("nodes", ns)
              .add("in_dtypes", in_dtypes)
              .add("out_dtypes", out_dtypes)
              .add("orig_output_shape", *out_shape)
              .add("default_shape", static_cast<int64_t>(default_shape))
              .add("op_version", mladf_version_)
              .add("max_m", static_cast<int64_t>(max_seq_len))
              .add("input_shape", *in_shape)
              // .add("group_size", static_cast<int64_t>(block_size))
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

DEFINE_VAIP_PASS(Dd_merge_mlp, vaip_pass_dd_merge_mlp)
