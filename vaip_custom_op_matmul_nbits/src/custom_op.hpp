/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

/*

**/
#pragma once

#include "vaip/vaip.hpp"
#include <algorithm>
#include <future>
#include <mutex>
#include <xrt/xrt_bo.h>

#include <chrono>
// #define PROFILE_MATMULNBITS
#ifdef PROFILE_MATMULNBITS
#  define USE_TIMER_MATMULNBITS(timer) timer
#else
#  define USE_TIMER_MATMULNBITS(timer)
#endif

uint16_t float_to_bfloat16(float value);

void float_to_bfloat16_avx512_unrolled(const float* v, uint16_t* out,
                                       size_t size);
float bfloat16_to_float(uint16_t x);
void bfloat16_to_float_avx512_unrolled(uint16_t* s, float* d, int n);
void bfloat16_to_float_full(uint16_t* s, float* d, int n);

namespace vaip_matmul_nbits_custom_op {

using namespace vaip_core;

class MyCustomOp : public CustomOpImp {
public:
  MyCustomOp(std::shared_ptr<const PassContext> context,
             const std::shared_ptr<MetaDefProto>& meta_def,
             onnxruntime::Model* model);

  virtual ~MyCustomOp();

private:
  virtual void Compute(const OrtApi* api,
                       OrtKernelContext* context) const override final;
  void init_op_mladf_dd(std::vector<int8_t> b, std::vector<int8_t> zeros,
                        std::vector<float> scales, std::vector<float> bias);
  void execute_mladf_dd(const uint16_t* input_data, uint16_t* out,
                        std::vector<int64_t> input_shape,
                        std::vector<int> wts_shape, int grp_size,
                        int run_cnt) const;
  static std::shared_ptr<void> gemm__;
  int k_k, k_n, k_bits, k_block_size;
  int k_asymmetric = 0;
  // std::tuple<int, int> wts_shape_;
  mutable uint16_t* input_data_ = nullptr;
  int cnt;
  bool dry_run_;
};

} // namespace vaip_matmul_nbits_custom_op
