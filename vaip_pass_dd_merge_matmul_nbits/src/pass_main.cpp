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

DEF_ENV_PARAM(DEBUG_DD_MERGE_MATMUL_NBITS, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_DD_MERGE_MATMUL_NBITS) >= n)

#define MLADF_VERSION "v1"
#define MAX_SEQ_LENGTH 3072

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
    { "name": "vaip_pass_dd_merge_matmul_nbits",
       "plugin": "vaip-pass_dd_merge_matmul_nbits",
       "disabled": false
    }
*/
namespace {
using namespace vaip_core;
static bool first_mm = true;

struct Dd_merge_matmul_nbits {
  Dd_merge_matmul_nbits(IPass& self) : self_{self} {}
  std::unique_ptr<Rule> create_rule(IPass* self) {

    auto builder = PatternBuilder();
    auto p_input = builder.wildcard();
    auto input_w = builder.wildcard();
    auto scales = builder.wildcard();
    auto zp = builder.wildcard();
    auto cast_in = builder.node2("Cast", {p_input});

    auto matmul = builder.node3("com.microsoft:MatMulNBits",
                                {cast_in, input_w, scales, zp},
                                {false, false, false, true});

    auto mm_nbits = builder.node2("Cast", {matmul});
    CHECK(mm_nbits != nullptr) << "Pattern returned is null";
    return Rule::create_rule(
        mm_nbits, [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
          auto in_node = binder[p_input->get_id()];
          auto out_node = binder[mm_nbits->get_id()];
          auto wts_node = binder[input_w->get_id()];
          auto wts_scale_node = binder[scales->get_id()];
          auto wts_zp_node = binder[zp->get_id()];
          auto matmul_node = binder[matmul->get_id()];

          bool is_zp_present = false;
          if (wts_zp_node.node_arg != nullptr) {
            is_zp_present = true;
          }

          int64_t k_k = node_get_attr_int(*matmul_node.node, "K");
          int64_t k_n = node_get_attr_int(*matmul_node.node, "N");
          int64_t bits = node_get_attr_int(*matmul_node.node, "bits");
          int64_t block_size =
              node_get_attr_int(*matmul_node.node, "block_size");
          size_t kblks = k_k / block_size;

          std::vector<std::string> ns = vaip::dd::get_node_names(graph, binder);
          auto node_name = node_arg_get_name(*out_node.node_arg);
          MY_LOG(1) << "found match at " << ns.front();
          // Extracting the scales and zero points for inputs and weights

          auto out_shape = node_arg_get_shape_i64(*out_node.node_arg);
          auto o_s = *out_shape;
          auto in0_shape = node_arg_get_shape_i64(*in_node.node_arg);

          auto w_shape = node_arg_get_shape_i64(*wts_node.node_arg);
          auto wts_shape = *w_shape;
          // Weights shape coming as N X K = (9216 X 3027)
          auto w_sc_shape = node_arg_get_shape_i64(*wts_scale_node.node_arg);
          auto sc_shape = *w_sc_shape;

          auto weight_data_type = node_arg_get_element_type(*wts_node.node_arg);

          // if(weight_data_type == 3){
          auto wts_sc_shape = *w_sc_shape;
          // Initializing the weights and scales and zero points to add them as
          // tensors in node builder
          gsl::span<const uint8_t> weights;
          gsl::span<const float> weights_scale;
          float weights_sc;
          gsl::span<const int8_t> weights_zero_point;
          std::vector<float> wts_sc_vec;
          std::vector<float> wts_sc_vec_1;
          std::vector<float> wts_sc_vec_bs_1;
          std::vector<int8_t> wts_zp_vec_1;
          std::vector<int8_t> wts_zp_vec_bs;
          std::vector<float> wts_sc_vec_bs;
          int8_t weights_zp;
          std::vector<int8_t> wts_zp_vec;

          weights = node_arg_get_const_data_as_u8s(*graph, *wts_node.node_arg);
          std::vector<int8_t> wts_vec(weights.begin(), weights.end());

          std::vector<int8_t> const_wts(k_k * k_n, 0);
          for (int64_t i = 0; i < k_k; i += 2) {
            for (int64_t j = 0; j < k_n; j++) {
              auto srcv = wts_vec[j * k_k / 2 + i / 2];
              auto src0 = (srcv & 0xf) - 8;
              auto src1 = ((srcv & 0xf0) >> 4) - 8;
              const_wts[i * k_n + j] = static_cast<int8_t>(src0);
              const_wts[(i + 1) * k_n + j] = static_cast<int8_t>(src1);
            }
          }

          std::string wts_initializer_name =
              node_arg_get_name(*wts_node.node_arg) + "0";
          const std::vector<int64_t> wts_initializer_shape = {(int64_t)k_k,
                                                              (int64_t)k_n};
          NodeArg& wts_arg = vaip::dd::insert_named_tensor_in_graph<int8_t>(
              graph, wts_initializer_name, const_wts, wts_initializer_shape);

          weights_scale = node_arg_get_const_data_as_floats(
              *graph, *wts_scale_node.node_arg);
          std::vector<float> scl_vec(weights_scale.begin(),
                                     weights_scale.end());
          std::vector<float> const_scl(k_k * k_n / block_size);
          for (int i = 0; i < k_n; i++) {
            for (int j = 0; j < kblks; j++) {
              const_scl[j * k_n + i] = scl_vec[i * kblks + j];
            }
          }
          std::string wts_sc_initializer_name =
              node_arg_get_name(*wts_scale_node.node_arg) + "0";
          const std::vector<int64_t> wts_sc_initializer_shape = {
              (int64_t)1, (int64_t)(sc_shape[0])};

          NodeArg& wts_sc_arg = vaip::dd::insert_named_tensor_in_graph<float>(
              graph, wts_sc_initializer_name, const_scl,
              wts_sc_initializer_shape);

          std::string bias_initializer_name =
              node_arg_get_name(*wts_scale_node.node_arg) + "bias";
          const std::vector<int64_t> bias_initializer_shape = {
              (int64_t)1, (int64_t)wts_shape[0]};
          std::vector<float> bias_vec(wts_shape[0], 0);
          NodeArg& bias_arg = vaip::dd::insert_named_tensor_in_graph<float>(
              graph, bias_initializer_name, bias_vec, bias_initializer_shape);

          std::string wts_zp_initializer_name =
              node_arg_get_name(*wts_scale_node.node_arg) + "zp";
          const std::vector<int64_t> wts_zp_initializer_shape = {
              (int64_t)1, (int64_t)(sc_shape[0])};
          std::vector<uint8_t> const_zp(sc_shape[0], 0);
          NodeArg& wts_zp_arg = vaip::dd::insert_named_tensor_in_graph<uint8_t>(
              graph, wts_zp_initializer_name, const_zp,
              wts_zp_initializer_shape);

          // hard code for mzdk5, may need to change
          std::vector<std::string> input_types{"bfloat16", "int8", "float",
                                               "float", "uint8"};
          std::vector<std::string> output_types{"bfloat16"};
          std::string mladf_version_(MLADF_VERSION);
          int32_t default_shape = 1;
          int32_t max_seq_len = MAX_SEQ_LENGTH;
          NodeBuilder(*graph, *self)
              .set_input_node_args({in_node.node_arg, &wts_arg, &bias_arg,
                                    &wts_sc_arg, &wts_zp_arg})
              .set_op_type("MladfMatMul", "com.xilinx")
              .add("nodes", ns)
              .add("in_dtypes", input_types)
              .add("out_dtypes", output_types)
              .add("orig_output_shape", *out_shape)
              //.add("input_q_params", input_q_params)
              //.add("output_q_params", input_q_params)
              .add("default_shape", static_cast<int64_t>(default_shape))
              .add("op_version", mladf_version_)
              .add("max_m", static_cast<int64_t>(max_seq_len))
              .add("group_size", static_cast<int64_t>(block_size))
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

DEFINE_VAIP_PASS(Dd_merge_matmul_nbits, vaip_pass_dd_merge_matmul_nbits)
