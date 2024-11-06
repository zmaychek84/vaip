/*
 *     The Xilinx Vitis AI Vaip in this distribution are provided under the
 * following free and permissive binary-only license, but are not provided in
 * source code form.  While the following free and permissive license is similar
 * to the BSD open source license, it is NOT the BSD open source license nor
 * other OSI-approved open source license.
 *
 *      Copyright (C) 2022 Xilinx, Inc. All rights reserved.
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
#define ORT_API_MANUAL_INIT
#include <onnxruntime_cxx_api.h>
#undef ORT_API_MANUAL_INIT

#include <algorithm>

// #include <ryzenai/dynamic_dispatch/ops/bmm/bmm.hpp>
#include <ryzenai/dynamic_dispatch/ops/elwmul/elwmul.hpp>
#include <ryzenai/dynamic_dispatch/ops/mladfadd/mladfadd.hpp>
#include <ryzenai/dynamic_dispatch/ops/mladfmatmulbias/mladfmatmulbias.hpp>
#include <ryzenai/dynamic_dispatch/ops/mladfrmsnorm/mladfrmsnorm.hpp>
#include <ryzenai/dynamic_dispatch/ops/silu/silu.hpp>
#include <xrt/xrt_bo.h>

namespace ort_ssmlp_custom_op {

struct OrtTensor {
  std::vector<int64_t> shape;
  size_t size;
  void* data;
};

class MyCustomOpKernel {
public:
  MyCustomOpKernel(const OrtKernelInfo* info);
  ~MyCustomOpKernel();
  void LazyInit();
  void Compute(OrtKernelContext* context);

private:
  bool sslrn_cpu_out = false;
  static int instances__;
  int cnt_;
  size_t num_el, num_el2;

  Ort::Logger m_logger{nullptr};
  static std::once_flag initFlag;
  float epsilon_;
  Ort::Op op_k{nullptr};
  Ort::Op op_k2{nullptr};

  // sslrn cpu io
  static float* input_a;
  static float* input_b;
  static float* output_1;
  static float* output_2;

  // aie kernels from DD
  ryzenai::rms_norm<uint16_t, uint16_t, uint16_t>* rms_norm_{nullptr};
  ryzenai::mladf_add<uint16_t, uint16_t, uint16_t>* add_{nullptr};
  ryzenai::rms_norm<uint16_t, uint16_t, uint16_t>* rms_norm2_{nullptr};
  ryzenai::mladf_add<uint16_t, uint16_t, uint16_t>* add2_{nullptr};

  static std::shared_ptr<void> ewmul_;
  static std::shared_ptr<void> silu_;
  static std::shared_ptr<void> gate_proj_;
  static std::shared_ptr<void> up_proj_;
  static std::shared_ptr<void> down_proj_;

  std::vector<size_t> supported_lengths{3072, 2048, 1920, 1792, 1664, 1536,
                                        1408, 1280, 1152, 1024, 800,  768,
                                        640,  512,  384,  256,  128};
  uint16_t* wts_ = nullptr;
  uint16_t* wts2_ = nullptr;
  Ort::ConstValue m_weights;
  Ort::ConstValue m2_weights;

  int64_t gp_k, gp_n, gp_bits;
  int gp_block_size;
  int64_t up_k, up_n, up_bits;
  int up_block_size;
  int64_t dp_k, dp_n, dp_bits;
  int dp_block_size;
  bool dry_run_;
};

struct MyCustomOp : Ort::CustomOpBase<MyCustomOp, MyCustomOpKernel> {
  explicit MyCustomOp() {}

  OrtCustomOpInputOutputCharacteristic
  GetInputCharacteristic(size_t) const noexcept {
    // zero-points input is optional
    return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_OPTIONAL;
  }

  OrtCustomOpInputOutputCharacteristic
  GetOutputCharacteristic(size_t) const noexcept {
    // zero-points input is optional
    return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_OPTIONAL;
  }

  void* CreateKernel(const OrtApi&, const OrtKernelInfo* info) const {
    return new MyCustomOpKernel(info);
  };

  const char* GetName() const { return "SSMLP"; };

  size_t GetInputTypeCount() const { return 13; };
  size_t GetOutputTypeCount() const { return 2; };

  ONNXTensorElementDataType GetInputType(size_t index = 0) const {

    if (index == 2 || index == 4 || index == 7 || index == 10 || index == 12)
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    else if (index == 0 || index == 1)
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
    else if (index == 3 || index == 5 || index == 6 || index == 8 ||
             index == 9 || index == 11)
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
    else
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  };

  ONNXTensorElementDataType GetOutputType(size_t index = 0) const {
    if (index == 0)
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
    else if (index == 1)
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
    else
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  };
};

} // namespace ort_ssmlp_custom_op