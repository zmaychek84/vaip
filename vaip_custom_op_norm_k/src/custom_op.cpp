/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

#include "onnxruntime_api.hpp"

#include "custom_op.hpp"
#include <glog/logging.h>
#include <sstream>

#include "helper.hpp"
#include "vitis/ai/env_config.hpp"
#include <cmath>
#include <thread>
#include <unordered_map>
#include <vitis/ai/weak.hpp>

#ifdef _WIN32
#  pragma warning(push)
#  pragma warning(disable : 4005)
#  pragma warning(pop)
#else
#endif

#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_NORM_K) >= n)
DEF_ENV_PARAM(DEBUG_NORM_K, "0");

namespace vaip_norm_k_custom_op {

MyCustomOp::MyCustomOp(std::shared_ptr<const PassContext> context,
                       const std::shared_ptr<MetaDefProto>& meta_def,
                       onnxruntime::Model* model)
    : CustomOpImp(context, meta_def, model) {
  MY_LOG(1) << " Vitis AI  NORNK custom op running " << meta_def->nodes_size()
            << " Nodes";
  auto input_size = meta_def->asr_param().input_shapes().size();
  std::string input_0_file = meta_def->generic_param().at("input_0_file");
  std::vector<uint16_t> input_0_vec;
  norm_k::read_bin_file(input_0_file, input_0_vec);
  std::vector<float> input_0_float;
  uint16_t const_0 = stoi(meta_def->generic_param().at("const_0"));
  uint16_t const_2 = stoi(meta_def->generic_param().at("const_2"));
  uint16_t const_3 = stoi(meta_def->generic_param().at("const_3"));
  uint16_t const_5 = stoi(meta_def->generic_param().at("const_5"));
  uint16_t const_8 = stoi(meta_def->generic_param().at("const_8"));
  uint16_t const_11 = stoi(meta_def->generic_param().at("const_11"));
  uint16_t const_13 = stoi(meta_def->generic_param().at("const_13"));
  uint16_t const_15 = stoi(meta_def->generic_param().at("const_15"));
  uint16_t const_17 = stoi(meta_def->generic_param().at("const_17"));
  uint16_t const_19 = stoi(meta_def->generic_param().at("const_19"));
  uint16_t const_21 = stoi(meta_def->generic_param().at("const_21"));
  uint16_t const_23 = stoi(meta_def->generic_param().at("const_23"));

  std::string scale_file = meta_def->generic_param().at("scale_file");
  std::vector<float> scale_vec;
  norm_k::read_bin_file(scale_file, scale_vec);
  float const_1 = scale_vec[0];
  float const_4 = scale_vec[1];
  float const_7 = scale_vec[2];
  float const_10 = scale_vec[3];
  float const_12 = scale_vec[4];
  float const_14 = scale_vec[5];
  float const_16 = scale_vec[6];
  float const_18 = scale_vec[7];
  float const_20 = scale_vec[8];
  float const_22 = scale_vec[9];

  norm_k::dequantlinear_op(input_0_vec, input_0_float, const_7, const_8);
  //   norm_k::save_vec_span_2_bin(input_0_float, "input_0_float.bin");

  std::string gather_indices_file =
      meta_def->generic_param().at("gather_indices_file");
  std::vector<int64_t> gather_indices_vec;
  norm_k::read_bin_file(gather_indices_file, gather_indices_vec);
  int64_t gather_axis = stoi(meta_def->generic_param().at("gather_axis"));

  // shape for gather input tensor
  std::vector<int> data_shape = {2000, 64};
  norm_k::Tensor<float> data_tensor(input_0_float, data_shape);
  // shapre for gather indices hardcoded for now
  std::vector<int> indices_shape = {25, 475};
  norm_k::Tensor<int64_t> indices_tensor(gather_indices_vec, indices_shape);
  norm_k::Tensor<float> result =
      norm_k::Gather(data_tensor, indices_tensor, gather_axis);
  auto res_shape = result.Shape();

  //   norm_k::save_vec_span_2_bin(result.Data(), "Gather_output_co.bin");
  std::vector<uint16_t> gather_q_output;
  norm_k::quantizeLinear_op(result.Data(), gather_q_output, const_7, const_8);
  std::vector<float> gather_q_dq_output;
  norm_k::dequantlinear_op(gather_q_output, gather_q_dq_output, const_7,
                           const_8);
  //   norm_k::save_vec_span_2_bin(gather_q_dq_output,
  //   "Gather_q_dq_output_co.bin");

  int64_t reducemean_axis = stoi(meta_def->generic_param().at("const_9"));
  std::vector<int> gather_q_dq_output_shape = {25, 475, 64};
  norm_k::Tensor<float> gather_tensor(gather_q_dq_output,
                                      gather_q_dq_output_shape);
  std::vector<float> res_reduce_mean =
      norm_k::ReduceMean(gather_tensor, reducemean_axis);
  //   norm_k::save_vec_span_2_bin(res_reduce_mean, "ReduceMean_output_co.bin");
  // std::vector<float> res_reduce_mean ;
  // norm_k::read_bin_file("ReduceMean_output_co.bin", res_reduce_mean);
  std::vector<uint16_t> rm0_q_output;
  norm_k::quantizeLinear_op(res_reduce_mean, rm0_q_output, const_10, const_11);
  std::vector<float> rm0_q_dq_output;
  norm_k::dequantlinear_op(rm0_q_output, rm0_q_dq_output, const_10, const_11);
  std::vector<int> reduce_shape = {25, 475, 1};
  norm_k::Tensor<float> sub_sec_tensor(rm0_q_dq_output, reduce_shape);
  std::vector<float> res_sub = norm_k::Sub(gather_tensor, sub_sec_tensor);
  std::vector<uint16_t> sub_q_output;
  norm_k::quantizeLinear_op(res_sub, sub_q_output, const_12, const_13);
  std::vector<float> sub_q_dq_output;
  norm_k::dequantlinear_op(sub_q_output, sub_q_dq_output, const_12, const_13);
  //   norm_k::save_vec_span_2_bin(sub_q_dq_output, "sub_dq_co.bin");

  std::vector<uint16_t> pow_sec_input_fix{const_3};
  std::vector<float> pow_sec_input_float;
  norm_k::dequantlinear_op(pow_sec_input_fix, pow_sec_input_float, const_4,
                           const_5);

  std::vector<float> pow_res =
      norm_k::Pow(sub_q_dq_output, pow_sec_input_float.at(0));
  std::vector<uint16_t> pow_q_output;
  norm_k::quantizeLinear_op(pow_res, pow_q_output, const_14, const_15);
  std::vector<float> pow_q_dq_output;
  norm_k::dequantlinear_op(pow_q_output, pow_q_dq_output, const_14, const_15);
  //   norm_k::save_vec_span_2_bin(pow_q_dq_output, "pow_dq_co.bin");

  norm_k::Tensor<float> reduce_mean_1_input(pow_q_dq_output,
                                            gather_q_dq_output_shape);
  std::vector<float> res_reduce_mean_1 =
      norm_k::ReduceMean(reduce_mean_1_input, reducemean_axis);
  std::vector<uint16_t> rm1_q_output;
  norm_k::quantizeLinear_op(res_reduce_mean_1, rm1_q_output, const_16,
                            const_17);
  std::vector<float> rm1_q_dq_output;
  // norm_k::read_bin_file("rm1_dq_ort.bin", rm1_q_dq_output);
  norm_k::dequantlinear_op(rm1_q_output, rm1_q_dq_output, const_16, const_17);
  //   norm_k::save_vec_span_2_bin(rm1_q_dq_output, "rm1_dq_co.bin");

  std::vector<uint16_t> add_sec_input_fix{const_0};
  std::vector<float> add_sec_input_float;
  norm_k::dequantlinear_op(add_sec_input_fix, add_sec_input_float, const_1,
                           const_2);
  std::vector<float> add_res =
      norm_k::Add(rm1_q_dq_output, add_sec_input_float);
  std::vector<uint16_t> add_q_output;
  norm_k::quantizeLinear_op(add_res, add_q_output, const_18, const_19);
  std::vector<float> add_q_dq_output;
  norm_k::dequantlinear_op(add_q_output, add_q_dq_output, const_18, const_19);
  //   norm_k::save_vec_span_2_bin(rm1_q_dq_output, "add_dq_co.bin");

  std::vector<float> sqrt_res = norm_k::Sqrt(rm1_q_dq_output);
  std::vector<uint16_t> sqrt_q_output;
  norm_k::quantizeLinear_op(sqrt_res, sqrt_q_output, const_20, const_21);
  std::vector<float> sqrt_q_dq_output;
  norm_k::dequantlinear_op(sqrt_q_output, sqrt_q_dq_output, const_20, const_21);
  //   norm_k::save_vec_span_2_bin(sqrt_q_dq_output, "sqrt_dq_co.bin");

  norm_k::Tensor<float> div_first_tensor(sub_q_dq_output,
                                         gather_q_dq_output_shape);
  norm_k::Tensor<float> div_sec_tensor(sqrt_q_dq_output, reduce_shape);
  std::vector<float> div_res = norm_k::Div(div_first_tensor, div_sec_tensor);
  std::vector<uint16_t> div_q_output;
  norm_k::quantizeLinear_op(div_res, div_q_output, const_22, const_23);
  std::vector<float> div_q_dq_output;
  norm_k::dequantlinear_op(div_q_output, div_q_dq_output, const_22, const_23);
  //   norm_k::save_vec_span_2_bin(div_q_dq_output, "div_dq_co.bin");

  norm_k::Tensor<float> transpose_input(div_q_dq_output,
                                        gather_q_dq_output_shape);
  norm_k::Tensor<float> trans_res = norm_k::Transpose(transpose_input);

  norm_k::quantizeLinear_op(trans_res.Data(), pre_computed_output, const_22,
                            const_23);
  //   norm_k::save_vec_span_2_bin(pre_computed_output, "trans_q_co.bin");

  // auto bin_file_path = R"(C:\chuanlia\const_fold\output.bin)";
  // norm_k::read_bin_file(bin_file_path, pre_computed_output);
}

MyCustomOp::~MyCustomOp() {}

void MyCustomOp::Compute(const OrtApi* api, OrtKernelContext* context) const {
  Ort::KernelContext ctx(context);
  std::vector<int64_t> out_shape{25, 64, 475};
  auto output_tensor = ctx.GetOutput(0, {out_shape.begin(), out_shape.end()});
  auto out = output_tensor.GetTensorMutableData<uint16_t>();
  memcpy((void*)out, (void*)pre_computed_output.data(),
         pre_computed_output.size() * sizeof(uint16_t));
}

} // namespace vaip_norm_k_custom_op
