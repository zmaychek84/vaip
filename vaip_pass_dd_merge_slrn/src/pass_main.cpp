/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */
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
#include <functional>
#include <glog/logging.h>
#include <numeric>

#include "vaip/tensor_proto.hpp"

DEF_ENV_PARAM(DEBUG_DD_MERGE_SLRN, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_DD_MERGE_SLRN) >= n)

namespace {
using namespace vaip_core;
static uint16_t float_to_bfloat16(float x) {
  uint32_t i;
  uint8_t* src = (uint8_t*)&x;
  uint8_t* tmp = (uint8_t*)&i;
  // copy float to uint32_t
  std::memcpy(tmp, src, sizeof(float));
  // round to nearest even
  uint32_t lsb = (i >> 16) & 0x1;
  uint32_t bias = 0x7fff + lsb;
  i += bias;
  // extract upper half of input
  uint16_t y = uint16_t(i >> 16);
  return y;
}

static void float_to_bfloat16_avx512_unrolled(const float* v, uint16_t* out,
                                              size_t size) {
  constexpr size_t nelems_in_vector = sizeof(__m512) / sizeof(float);
  constexpr size_t unroll_factor = 4;
  constexpr size_t nelems_per_loop = nelems_in_vector * unroll_factor;

  static const __m512i ones = _mm512_set1_epi32(0x1);           // 1
  static const __m512i round_value = _mm512_set1_epi32(0x7fff); // 1

  const uint32_t* v32 = reinterpret_cast<const uint32_t*>(v);
  size_t i = 0;
  for (; i < (size / nelems_per_loop) * nelems_per_loop; i += nelems_per_loop) {
    __m512i a0 = _mm512_loadu_epi32(v32 + i + nelems_in_vector * 0);
    __m512i a1 = _mm512_loadu_epi32(v32 + i + nelems_in_vector * 1);
    __m512i a2 = _mm512_loadu_epi32(v32 + i + nelems_in_vector * 2);
    __m512i a3 = _mm512_loadu_epi32(v32 + i + nelems_in_vector * 3);

    _mm_prefetch((const char*)v32 +
                     (i + nelems_in_vector * 0 + nelems_per_loop) * 4,
                 _MM_HINT_T0);
    _mm_prefetch((const char*)v32 +
                     (i + nelems_in_vector * 1 + nelems_per_loop) * 4,
                 _MM_HINT_T0);
    _mm_prefetch((const char*)v32 +
                     (i + nelems_in_vector * 2 + nelems_per_loop) * 4,
                 _MM_HINT_T0);
    _mm_prefetch((const char*)v32 +
                     (i + nelems_in_vector * 3 + nelems_per_loop) * 4,
                 _MM_HINT_T0);

    __m512i c0 = _mm512_srli_epi32(a0, 16);
    __m512i c1 = _mm512_srli_epi32(a1, 16);
    __m512i c2 = _mm512_srli_epi32(a2, 16);
    __m512i c3 = _mm512_srli_epi32(a3, 16);

    __m512i lsb0 = _mm512_and_epi32(c0, ones);
    __m512i lsb1 = _mm512_and_epi32(c1, ones);
    __m512i lsb2 = _mm512_and_epi32(c2, ones);
    __m512i lsb3 = _mm512_and_epi32(c3, ones);

    __m512i bias0 = _mm512_add_epi32(lsb0, round_value);
    __m512i bias1 = _mm512_add_epi32(lsb1, round_value);
    __m512i bias2 = _mm512_add_epi32(lsb2, round_value);
    __m512i bias3 = _mm512_add_epi32(lsb3, round_value);

    __m512i d0 = _mm512_add_epi32(a0, bias0);
    __m512i d1 = _mm512_add_epi32(a1, bias1);
    __m512i d2 = _mm512_add_epi32(a2, bias2);
    __m512i d3 = _mm512_add_epi32(a3, bias3);

    __m512i e0 = _mm512_srli_epi32(d0, 16);
    __m512i e1 = _mm512_srli_epi32(d1, 16);
    __m512i e2 = _mm512_srli_epi32(d2, 16);
    __m512i e3 = _mm512_srli_epi32(d3, 16);

    __m256i z0 = _mm512_cvtusepi32_epi16(e0);
    __m256i z1 = _mm512_cvtusepi32_epi16(e1);
    __m256i z2 = _mm512_cvtusepi32_epi16(e2);
    __m256i z3 = _mm512_cvtusepi32_epi16(e3);

    _mm256_stream_si256((__m256i*)(out + i + nelems_in_vector * 0), z0);
    _mm256_stream_si256((__m256i*)(out + i + nelems_in_vector * 1), z1);
    _mm256_stream_si256((__m256i*)(out + i + nelems_in_vector * 2), z2);
    _mm256_stream_si256((__m256i*)(out + i + nelems_in_vector * 3), z3);
  }
  for (; i < size; ++i) {
    out[i] = float_to_bfloat16(v[i]);
  }
  _mm_sfence();
}
static NodeArg& insert_named_bfloat16_tensor_in_graph(
    onnxruntime::Graph* graph, std::string tensor_name,
    std::vector<int16_t> data, const std::vector<int64_t>& shape) {

  auto n_tensor = tensor_proto_new_bf16(tensor_name, shape, data);
  VAIP_ORT_API(graph_add_initialized_tensor)(*graph, *n_tensor);
  auto& n_arg =
      VAIP_ORT_API(node_arg_new)(*graph, tensor_name, &shape,
                                 ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16);
  return n_arg;
}
struct slrn {
  slrn(IPass& self) : self_{self} {}

  std::unique_ptr<Rule> create_rule(IPass* self) {
    auto builder = PatternBuilder();

    auto input_0 =
        builder.wildcard(); //  id = 2  node_arg_name =
                            //  /model/embed_tokens/Gather_output_0_to_bf16_
    // builder.bind("/model/embed_tokens/Gather_output_0_to_bf16_",input_0);
    auto Cast_0 = builder.node2(
        "Cast",
        {input_0}); //  id = 3  node_arg_name =
                    //  /model/embed_tokens/Gather_output_0_to_fp32_duplicate_
    // builder.bind("/model/embed_tokens/Gather_output_0_to_fp32_duplicate_",Cast_0);
    auto constant_0 =
        builder.constant(); //  id = 4  node_arg_name =
                            //  /model/layers.0/input_layernorm/L2Norm_scale
    // builder.bind("/model/layers.0/input_layernorm/L2Norm_scale",constant_0);
    auto SimplifiedLayerNormalization_0 = builder.node2(
        "SimplifiedLayerNormalization",
        {Cast_0,
         constant_0}); //  id = 5  node_arg_name =
                       //  /model/layers.0/input_layernorm/Mul_1_output_0
    // builder.bind("/model/layers.0/input_layernorm/Mul_1_output_0",SimplifiedLayerNormalization_0);
    auto Cast_1 = builder.node2("Cast", {SimplifiedLayerNormalization_0});
    return Rule::create_rule(
        Cast_1, [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
          std::vector<std::string> ns = vaip::dd::get_node_names(graph, binder);
          auto input_0_node = binder[input_0->get_id()];
          auto const_0_node = binder[constant_0->get_id()];
          auto out_node = binder[Cast_1->get_id()];

          // todo: change const tensor from float to bfloat
          auto const_node_name = node_arg_get_name(*const_0_node.node_arg);
          std::string wts_name = std::string(const_node_name + "_bf16");

          // auto float_wts_tensor =
          //     node_arg_get_const_data_as_tensor(*graph,
          //     *const_0_node.node_arg);
          auto v = tensor_proto_as_floats(*graph,
                                          node_arg_get_const_data_as_tensor(
                                              *graph, *const_0_node.node_arg));
          auto w_shape_vec =
              tensor_proto_get_shape(node_arg_get_const_data_as_tensor(
                  *graph, *const_0_node.node_arg));

          std::vector<int16_t> bf16_wts(w_shape_vec[0], 0);
          float_to_bfloat16_avx512_unrolled(v.data(),
                                            (uint16_t*)(bf16_wts.data()),
                                            w_shape_vec[0]); // K

          auto& wts_arg = insert_named_bfloat16_tensor_in_graph(
              graph, wts_name, bf16_wts, w_shape_vec);

          std::vector<std::string> input_types{"bfloat16", "bfloat16"};
          std::vector<std::string> output_types{"bfloat16"};
          NodeBuilder(*graph, *self)
              .set_input_node_args({input_0_node.node_arg, &wts_arg})
              .set_op_type("MLADFRMSNORM", "com.xilinx")
              .clone_attrs(*out_node.node)
              .add("nodes", ns)
              .set_anchor_point1(*out_node.node)
              .add("in_dtypes", input_types)
              .add("out_dtypes", output_types)
              .build();
          return true; // return true if graph is modified.
        });
  }
  // apply the rule
  void process(IPass& self, Graph& graph) {
    // MY_LOG(1) << self_.get_pass_proto().name() << "[" <<
    // self_.get_pass_proto().plugin() << "] start processing graph";
    create_rule(&self)->apply(&graph);
    // MY_LOG(1) << self_.get_pass_proto().name() << "[" <<
    // self_.get_pass_proto().plugin() << "] finish processing graph";
  }

  IPass& self_;
};
} // namespace

DEFINE_VAIP_PASS(slrn, vaip_pass_dd_merge_slrn)
