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

#include "schedule.hpp"
#include "ort_tensor_buffer.hpp"

#include "vaip/vaip.hpp"
#include "vitis/ai/env_config.hpp"
#include <cmath>
#include <thread>
#include <xir/util/tool_function.hpp>
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_DPU_CUSTOM_OP) >= n)
DEF_ENV_PARAM(DEBUG_DPU_CUSTOM_OP, "0");
DEF_ENV_PARAM(USE_CPU_RUNNER, "0");
DEF_ENV_PARAM(NUM_OF_PAD_THREADS, "1");
DEF_ENV_PARAM(XLNX_ENABLE_DUMP, "0");
DEF_ENV_PARAM(XLNX_ENABLE_BATCH, "0")
namespace vaip_dpu_custom_op {

static std::vector<int32_t> vec_int64_to_int32(std::vector<int64_t> shape) {
  auto ret = std::vector<int32_t>();
  ret.reserve(shape.size());
  for (auto dim : shape) {
    ret.push_back((int32_t)dim);
  }
  return ret;
}
static std::vector<int64_t> vec_int32_to_int64(std::vector<int32_t> shape) {
  auto ret = std::vector<int64_t>();
  ret.reserve(shape.size());
  for (auto dim : shape) {
    ret.push_back((int64_t)dim);
  }
  return ret;
}
// batch can be specify by cliet code

static void fix2float(void* dst, int8_t* data, int size, float scale) {
  float* ret = (float*)dst;
  for (int i = 0; i < size; i++) {
    ret[i] = (float)data[i] * scale;
  }
}
static void float2fix(void* dst, const float* data, int size, float scale) {
  int8_t* ret = (int8_t*)dst;
  for (int i = 0; i < size; i++) {
    ret[i] = (int8_t)(data[i] * scale);
  }
}
static void float2bfloat16(void* dst, const float* data, int size) {
  xir::bfloat16_t* ret = (xir::bfloat16_t*)dst;
  for (int i = 0; i < size; i++) {
    ret[i] = data[i];
  }
}
static void bfloat16_2_float(void* dst, const xir::bfloat16_t* data, int size) {
  float* ret = (float*)dst;
  for (int i = 0; i < size; i++) {
    ret[i] = data[i];
  }
}
static void uint82int8(int8_t* dst, const uint8_t* data, int size) {

  for (int i = 0; i < size; i++) {
    dst[i] = (int8_t)((int32_t)data[i] - 128);
    // LOG(INFO) << "== i" << i << " " << (int)data[i] << " " << (int)dst[i];
  }
}
static void int82uint8(uint8_t* dst, const int8_t* data, int size) {
  for (int i = 0; i < size; i++) {
    dst[i] = (uint8_t)((int32_t)data[i] + 128);
  }
}

std::shared_ptr<vart::TensorBuffer>
create_onnx_input_tensor_buffer(const std::string& node_arg_name,
                                Ort::KernelContext& context, int idx) {
  MY_LOG(1) << "hello create onnx input tensor buffer ";
  MY_LOG(1) << " create_onnx_input_tensor_buffer : node_arg_name "
            << node_arg_name << ", idx " << idx;
  auto input_tensor = context.GetInput(idx);
  auto tensor_info = input_tensor.GetTensorTypeAndShapeInfo();
  auto tensor_type = tensor_info.GetElementType();
  auto shape = tensor_info.GetShape();
  void* data;
  auto data_type = xir::DataType{xir::DataType::FLOAT, 32};
  if (tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    data_type = xir::DataType{xir::DataType::FLOAT, 32};
    data = (void*)input_tensor.GetTensorData<float>();
  } else if (tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8) {
    data_type = xir::DataType{xir::DataType::XINT, 8};
    data = (void*)input_tensor.GetTensorData<int8_t>();
  } else if (tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) {
    data_type = xir::DataType{xir::DataType::UINT, 8};
    data = (void*)input_tensor.GetTensorData<uint8_t>();
  } else if (tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16) {
    data_type = xir::DataType{xir::DataType::INT, 16};
    data = (void*)input_tensor.GetTensorData<int16_t>();
  } else if (tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16) {
    data_type = xir::DataType{xir::DataType::UINT, 16};
    data = (void*)input_tensor.GetTensorData<uint16_t>();
  } else if (tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16) {
    data_type = xir::DataType{xir::DataType::BFLOAT, 16};
    data = (void*)input_tensor.GetTensorData<uint16_t>();
  } else {
    LOG(FATAL) << "not support, onnx tensor_type " << tensor_type;
  }
  MY_LOG(1) << " create_onnx_input_tensor_buffer : node_arg_name "
            << node_arg_name << ", idx " << idx << ", data " << std::hex
            << data;

  auto tensor =
      xir::Tensor::create(node_arg_name, vec_int64_to_int32(shape), data_type);
  return OrtTensorBuffer::create(std::move(tensor), data);
}

std::shared_ptr<vart::TensorBuffer>
create_onnx_output_tensor_buffer(const std::string& node_arg_name,
                                 Ort::KernelContext& context, int idx,
                                 std::vector<int64_t> onnx_shape) {

  auto output_tensor = context.GetOutput(idx,            //
                                         &onnx_shape[0], //
                                         onnx_shape.size());
  auto tensor_info = output_tensor.GetTensorTypeAndShapeInfo();
  auto tensor_type = tensor_info.GetElementType();

  void* data;
  auto data_type = xir::DataType{xir::DataType::FLOAT, 32};
  if (tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    data_type = xir::DataType{xir::DataType::FLOAT, 32};
    data = (void*)output_tensor.GetTensorData<float>();
  } else if (tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8) {
    data_type = xir::DataType{xir::DataType::XINT, 8};
    data = (void*)output_tensor.GetTensorData<int8_t>();
  } else if (tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) {
    data_type = xir::DataType{xir::DataType::UINT, 8};
    data = (void*)output_tensor.GetTensorData<uint8_t>();
  } else if (tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16) {
    data_type = xir::DataType{xir::DataType::INT, 16};
    data = (void*)output_tensor.GetTensorData<int16_t>();
  } else if (tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16) {
    data_type = xir::DataType{xir::DataType::UINT, 16};
    data = (void*)output_tensor.GetTensorData<uint16_t>();
  } else if (tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16) {
    data_type = xir::DataType{xir::DataType::BFLOAT, 16};
    data = (void*)output_tensor.GetTensorData<uint16_t>();
  } else {
    LOG(FATAL) << "not support, onnx tensor_type " << tensor_type;
  }
  MY_LOG(1) << " create_onnx_output_tensor_buffer : node_arg_name"
            << node_arg_name << ", idx " << idx << ", data " << std::hex
            << data;
  auto tensor = xir::Tensor::create(node_arg_name,
                                    vec_int64_to_int32(onnx_shape), data_type);
  return OrtTensorBuffer::create(std::move(tensor), data);
}
vart::TensorBuffer*
find_tensor_buffer(const std::vector<vart::TensorBuffer*>& tensor_buffers,
                   const std::string& tensor_name) {
  for (auto tb : tensor_buffers) {
    if (tb->get_tensor()->get_name() == tensor_name) {
      return tb;
    }
  }
  return nullptr;
}

std::shared_ptr<vart::TensorBuffer> create_xir_tensor_buffer(
    const std::string& tensor_name,
    const std::vector<vart::TensorBuffer*>& tensor_buffers) {
  return std::shared_ptr<vart::TensorBuffer>(
      find_tensor_buffer(tensor_buffers, tensor_name), [](auto*) {});
}

static bool is_float_32_data(const xir::Tensor& tensor) {
  return tensor.get_data_type().type == xir::DataType::FLOAT &&
         tensor.get_data_type().bit_width == 32;
}

static bool is_uint_8_data(const xir::Tensor& tensor) {
  return tensor.get_data_type().type == xir::DataType::UINT &&
         tensor.get_data_type().bit_width == 8;
}
static bool is_int_8_data(const xir::Tensor& tensor) {
  return (tensor.get_data_type().type == xir::DataType::XINT ||
          tensor.get_data_type().type == xir::DataType::INT) &&
         tensor.get_data_type().bit_width == 8;
}
static bool is_int_16_data(const xir::Tensor& tensor) {
  return (tensor.get_data_type().type == xir::DataType::XINT ||
          tensor.get_data_type().type == xir::DataType::INT) &&
         tensor.get_data_type().bit_width == 16;
}
static bool is_uint_16_data(const xir::Tensor& tensor) {
  return (tensor.get_data_type().type == xir::DataType::XUINT ||
          tensor.get_data_type().type == xir::DataType::UINT) &&
         tensor.get_data_type().bit_width == 16;
}

static bool is_bfloat_16_data(const xir::Tensor& tensor) {
  return tensor.get_data_type().type == xir::DataType::BFLOAT &&
         tensor.get_data_type().bit_width == 16;
}
static void pad_c_hwc(const int8_t* src, int8_t* pad_data, int h, int w, int c,
                      const std::vector<int32_t>& paddings) {
  auto c1 = paddings[6];
  auto c2 = paddings[7];
  // copy source data
  for (int i = 0; i < h * w; ++i) {
    std::copy_n(src + i * (c - c1 - c2), (c - c1 - c2), pad_data + i * c + c1);
  }
}

static void pad_c_hwc_u8(const uint8_t* src, uint8_t* pad_data, int h, int w,
                         int c, const std::vector<int32_t>& paddings) {
  auto c1 = paddings[6];
  auto c2 = paddings[7];
  // copy source data
  for (int i = 0; i < h * w; ++i) {
    std::copy_n(src + i * (c - c1 - c2), (c - c1 - c2), pad_data + i * c + c1);
  }
}

static void pad_c(int8_t* pad_data, const int8_t* src, int h, int w, int c,
                  const std::vector<int32_t>& paddings) {
  int8_t pad_value = 0;
  auto c1 = paddings[6];
  auto c2 = paddings[7];
  // pad c1
  if (c1 > 0) {
    std::fill_n(pad_data, c1 * h * w, pad_value);
  }
  // pad c2
  if (c2 > 0) {
    std::fill_n(pad_data + (c - c2) * h * w, c2 * h * w, pad_value);
  }
  // copy source data
  std::copy_n(src, (c - c1 - c2) * h * w, pad_data + c1 * h * w);
}

static void pad_c(int16_t* pad_data, const int16_t* src, int h, int w, int c,
                  const std::vector<int32_t>& paddings) {
  int8_t pad_value = 0;
  auto c1 = paddings[6];
  auto c2 = paddings[7];
  // pad c1
  if (c1 > 0) {
    std::fill_n(pad_data, c1 * h * w, pad_value);
  }
  // pad c2
  if (c2 > 0) {
    std::fill_n(pad_data + (c - c2) * h * w, c2 * h * w, pad_value);
  }
  // copy source data
  std::copy_n(src, (c - c1 - c2) * h * w, pad_data + c1 * h * w);
}
static std::vector<std::int32_t> get_index_zeros(const xir::Tensor* tensor) {
  auto ret = tensor->get_shape();
  std::fill(ret.begin(), ret.end(), 0);
  return ret;
}

static std::string get_tensor_name(const xir::Tensor* tensor) {
  auto tensor_name = tensor->get_name();
  std::replace(tensor_name.begin(), tensor_name.end(), '/', '_');
  std::replace(tensor_name.begin(), tensor_name.end(), ':', '_');
  std::replace(tensor_name.begin(), tensor_name.end(), '(', '_');
  std::replace(tensor_name.begin(), tensor_name.end(), ')', '_');
  if (tensor_name.length() > 60u) {
    tensor_name = tensor_name.substr(0u, 60u);
  }
  return tensor_name;
}

void trans_data(std::shared_ptr<vart::TensorBuffer> from_tensor_buffer,
                std::shared_ptr<vart::TensorBuffer> to_tensor_buffer,
                const vaip_core::DataOperator& op, int64_t onnx_batch) {

  MY_LOG(1) << " trans_data:\n "                                        //
            << "\t  from_tensor_buffer "                                //
            << from_tensor_buffer->to_string() << "\n "                 //
            << " \t to_tensor_buffer " << to_tensor_buffer->to_string() //
            << "\n  \t DataOperator " << op.DebugString();
  auto from_tensor = from_tensor_buffer->get_tensor();
  auto to_tensor = to_tensor_buffer->get_tensor();
  CHECK_LE(onnx_batch, from_tensor->get_shape()[0])
      << "onnx model batch must be less than or equal to HW batch  ";
  CHECK_LE(onnx_batch, to_tensor->get_shape()[0])
      << "onnx model batch must be less than or equal to HW batch  ";

  auto from_batch_size =
      from_tensor->get_element_num() / from_tensor->get_shape()[0];
  auto to_batch_size = to_tensor->get_element_num() / to_tensor->get_shape()[0];
  CHECK_EQ(from_batch_size, to_batch_size);
  auto batch_size = from_batch_size;

  if (!ENV_PARAM(XLNX_ENABLE_BATCH)) {
    // when ignore HW batch
    batch_size = from_tensor->get_element_num();
    CHECK_EQ(from_tensor->get_element_num(), to_tensor->get_element_num())
        << " not support batch onnx model";
  }

  auto padding = std::vector<int32_t>();
  padding.reserve(op.padding_size());
  for (auto i = 0; i < op.padding_size(); ++i) {
    padding.push_back((int32_t)op.padding(i));
  }

  auto order = std::vector<int64_t>();
  for (auto i = 0; i < op.order_size(); ++i) {
    order.push_back(op.order(i));
  }

  auto op_is_pad = op.is_pad();
  if (ENV_PARAM(USE_CPU_RUNNER)) {
    op_is_pad = false;
  }

  for (auto index = 0; index < onnx_batch; ++index) {
    uint64_t data_from = 0u;
    size_t size_from = 0u;
    auto idx_from = get_index_zeros(from_tensor);
    idx_from[0] = (int)index;
    std::tie(data_from, size_from) = from_tensor_buffer->data(idx_from);

    uint64_t data_to = 0u;
    size_t size_to = 0u;
    auto idx_to = get_index_zeros(to_tensor);
    idx_to[0] = (int)index;
    std::tie(data_to, size_to) = to_tensor_buffer->data(idx_to);

    // float -> int8
    if (is_float_32_data(*from_tensor) && is_int_8_data(*to_tensor)) {
      CHECK(op.float2fix());
      CHECK(!op.fix2float())
          << "MetaSchedule not support, fix2float and "
             "float2fix is mutually exclusive,   MetaSchedule : "
          << op.DebugString();
      float input_fixed_scale = std::exp2f(1.0f * (float)op.fix_point());
      if (!op_is_pad && op.is_layout_transform()) { // float2fix + transpose
        MY_LOG(1) << "float->int8 , float2fix + transpose";

        auto transpose_src_shape = vec_int32_to_int64(from_tensor->get_shape());
        transpose_src_shape[0] = 1;
        auto ret_data = std::vector<int8_t>(batch_size, 0);
        float2fix(/*dst*/ &ret_data[0], reinterpret_cast<float*>(data_from),
                  (int)batch_size, input_fixed_scale);
        vaip_core::transpose_i8(/*src */ &ret_data[0],
                                /*dst*/ reinterpret_cast<int8_t*>(data_to),
                                transpose_src_shape, order);

      } else if (op_is_pad && !op.is_layout_transform()) {
        // testcase
        // /group/modelzoo/vai_q_onnx/P1_U8S8_quantized_models_36e81b6/res2net101_26w_4s/res2net101_26w_4s.onnx
        MY_LOG(1) << "float->int8 , float2fix + pad";
        auto fix_data = std::vector<int8_t>(batch_size, 0);
        float2fix(/*dst*/ &fix_data[0],
                  /*src */ reinterpret_cast<float*>(data_from), (int)batch_size,
                  input_fixed_scale);

        auto vart_tensor_shape = to_tensor->get_shape();
        // xir input layout is NHWC
        CHECK_EQ(vart_tensor_shape.size(), 4u) << "pad only support 4-dims";
        CHECK_EQ(padding.size(), 8)
            << "shape size is 4, paddings size must be 8." << op.DebugString();
        auto height = vart_tensor_shape.at(1);
        auto width = vart_tensor_shape.at(2);
        auto channel = vart_tensor_shape.at(3);
        channel += padding[6] + padding[7];
        pad_c_hwc(/*src*/ &fix_data[0],
                  /*dst*/ reinterpret_cast<int8_t*>(data_to), height, width,
                  channel, padding);
      } else if (!op_is_pad && !op.is_layout_transform()) { // only float2fix
        MY_LOG(1) << "float->int8 , only float2fix";
        float2fix(/*dst*/ reinterpret_cast<int8_t*>(data_to),
                  /*src */ reinterpret_cast<float*>(data_from), (int)batch_size,
                  input_fixed_scale);

      } else {
        LOG(FATAL) << "not support DataOperator (float -> int8) :  "
                   << op.DebugString();
      }

    }
    // int8->int8
    else if (is_int_8_data(*from_tensor) && is_int_8_data(*to_tensor)) {
      // Qlinear's model fix2float is true float2fix is false, or vice versa
      // CHECK_EQ(op.fix2float(), op.float2fix()) << op.DebugString();

      if (op_is_pad && op.is_layout_transform()) {
        MY_LOG(1) << "int8->int8 : pad && transpose ";
        // only IPU input, maybe pad + layout_transform for PT model, from
        // onnx   -> xir
        auto vart_tensor_shape = to_tensor->get_shape();
        // xir input layout is NHWC
        CHECK_EQ(vart_tensor_shape.size(), 4u)
            << "pad only support 4-dims (N3HW->N4HW)";
        CHECK_EQ(padding.size(), 8)
            << "shape size is 4, paddings size must be 8." << op.DebugString();

        auto height = vart_tensor_shape.at(1);
        auto width = vart_tensor_shape.at(2);
        auto channel_before_pad = vart_tensor_shape.at(3);
        auto channel = channel_before_pad + padding[6] + padding[7];

        CHECK_GT(channel_before_pad, 0) << "channel should be > 0";
        batch_size = batch_size * channel / channel_before_pad;

        auto ret_data = std::vector<int8_t>(batch_size, 0);
        pad_c(&ret_data[0], reinterpret_cast<int8_t*>(data_from), height, width,
              channel, padding);
        auto transpose_src_shape = vec_int32_to_int64(from_tensor->get_shape());
        // new shape is padding after shape
        transpose_src_shape[0] = 1;
        transpose_src_shape[1] = channel;
        vaip_core::transpose_i8(/*src */ &ret_data[0],
                                /*dst */
                                reinterpret_cast<int8_t*>(data_to),
                                transpose_src_shape, order);

      } else if (!op_is_pad && op.is_layout_transform()) {
        MY_LOG(1) << "int8->int8 , only transpose";
        auto transpose_src_shape = vec_int32_to_int64(from_tensor->get_shape());
        transpose_src_shape[0] = 1;
        vaip_core::transpose_i8(/*src */
                                reinterpret_cast<int8_t*>(data_from),
                                /*dst */
                                reinterpret_cast<int8_t*>(data_to),
                                transpose_src_shape, order);

      } else if (op_is_pad && !op.is_layout_transform()) {
        MY_LOG(1) << "int8->int8 , only pad";
        // only IPU input, maybe pad input for TF model,  from onnx -> xir
        auto vart_tensor_shape = to_tensor->get_shape();
        // xir input layout is NHWC
        CHECK_EQ(vart_tensor_shape.size(), 4u)
            << "pad only support 4-dims (N3HW->N4HW)";
        CHECK_EQ(padding.size(), 8)
            << "shape size is 4, paddings size must be 8." << op.DebugString();
        auto height = vart_tensor_shape.at(1);
        auto width = vart_tensor_shape.at(2);
        auto channel = vart_tensor_shape.at(3);
        channel += padding[6] + padding[7];

        if (channel == 4 && padding[6] == 0 && padding[7] == 1) {
          LOG(FATAL) << "TODO : int8->int8 , only pad, pad mode: pad_c_hwc_mt";
        } else {
          pad_c_hwc(/*src*/ reinterpret_cast<int8_t*>(data_from),
                    /*dst*/ reinterpret_cast<int8_t*>(data_to), height, width,
                    channel, padding);
        }

      } else {
        MY_LOG(1) << "int8->int8 , memcpy size " << batch_size;
        std::memcpy(/*dst*/ reinterpret_cast<int8_t*>(data_to),
                    /*src */ reinterpret_cast<int8_t*>(data_from),
                    batch_size * sizeof(int8_t));
      }

    }
    // int8->float
    else if (is_int_8_data(*from_tensor) && is_float_32_data(*to_tensor)) {

      CHECK(op.fix2float());
      CHECK(!op.float2fix());
      CHECK(!op_is_pad);
      float output_fixed_scale = std::exp2f(-1.0f * (float)op.fix_point());
      if (op.is_layout_transform()) {
        MY_LOG(1) << "int8->float, fix2float && transpose ";
        auto transpose_src_shape = vec_int32_to_int64(from_tensor->get_shape());
        transpose_src_shape[0] = 1;
        auto ret_data = std::vector<int8_t>(batch_size, 0);
        vaip_core::transpose_i8(/*src
                                 */
                                reinterpret_cast<int8_t*>(data_from),
                                /*dst */ &ret_data[0], transpose_src_shape,
                                order);
        fix2float(/*dst*/ reinterpret_cast<float*>(data_to), &ret_data[0],
                  (int)batch_size, output_fixed_scale);

      } else {
        MY_LOG(1) << "int8->float, only fix2float";
        fix2float(/*dst*/ reinterpret_cast<float*>(data_to),
                  /*src */ reinterpret_cast<int8_t*>(data_from),
                  (int)batch_size, output_fixed_scale);
      }
    }
    // uint8 -> int8
    else if (is_uint_8_data(*from_tensor) && is_int_8_data(*to_tensor)) {
      if (!op_is_pad && !op.is_layout_transform()) {
        MY_LOG(1) << "uint8->int8 ";
        uint82int8(/*dst*/ reinterpret_cast<int8_t*>(data_to),
                   /*src */ reinterpret_cast<uint8_t*>(data_from),
                   (int)batch_size);
      } else if (op_is_pad && !op.is_layout_transform()) {
        MY_LOG(1) << "uint8 -> int8 , only pad";
        // onnx -> xir
        auto vart_tensor_shape = to_tensor->get_shape();
        CHECK_EQ(vart_tensor_shape.size(), 4u) << "pad only support 4-dims";
        CHECK_EQ(padding.size(), 8)
            << "shape size is 4, paddings size must be 8." << op.DebugString();
        auto height = vart_tensor_shape.at(1);
        auto width = vart_tensor_shape.at(2);
        // only support padding in latest dim.
        auto channel_before_pad = vart_tensor_shape.at(3);
        auto channel = channel_before_pad + padding[6] + padding[7];
        batch_size = batch_size * channel / channel_before_pad;

        auto pad_data = std::vector<uint8_t>(batch_size, 0);
        pad_c_hwc_u8(/*src*/ reinterpret_cast<uint8_t*>(data_from),
                     /*dst*/ &pad_data[0], height, width, channel, padding);
        uint82int8(/*dst*/ reinterpret_cast<int8_t*>(data_to),
                   /*src */ &pad_data[0], (int)batch_size);

      } else if (!op_is_pad && op.is_layout_transform()) {
        MY_LOG(1) << "uint8->int8 , transpose";
        auto transpose_src_shape = vec_int32_to_int64(from_tensor->get_shape());
        transpose_src_shape[0] = 1;
        auto ret_data = std::vector<int8_t>(batch_size, 0);
        uint82int8(/*dst*/ &ret_data[0],
                   /*src */ reinterpret_cast<uint8_t*>(data_from),
                   (int)batch_size);
        vaip_core::transpose_i8(/*src */ &ret_data[0],
                                /*dst */ reinterpret_cast<int8_t*>(data_to),
                                transpose_src_shape, order);
      } else {
        MY_LOG(1) << "uint8->int8 , pad && transpose";
        // test case: issue #1163
        // /group/modelzoo/vai_q_onnx/P1_U8S8_quantized_models_bb8d36d/ecaresnet50d_pruned/ecaresnet50d_pruned.onnx
        // onnx -> xir
        auto vart_tensor_shape = to_tensor->get_shape();
        // xir input layout is NHWC
        CHECK_EQ(vart_tensor_shape.size(), 4u)
            << "pad only support 4-dims (N3HW->N4HW)";
        CHECK_EQ(padding.size(), 8)
            << "shape size is 4, paddings size must be 8." << op.DebugString();

        auto height = vart_tensor_shape.at(1);
        auto width = vart_tensor_shape.at(2);
        auto channel_before_pad = vart_tensor_shape.at(3);
        auto channel = channel_before_pad + padding[6] + padding[7];

        CHECK_GT(channel_before_pad, 0) << "channel should be > 0";
        auto batch_size_after = batch_size * channel / channel_before_pad;

        auto ret_data_i8 = std::vector<int8_t>(batch_size, 0);
        auto ret_data = std::vector<int8_t>(batch_size_after, 0);
        uint82int8(/*dst*/ &ret_data_i8[0],
                   /*src */ reinterpret_cast<uint8_t*>(data_from),
                   (int)batch_size);
        pad_c(&ret_data[0], &ret_data_i8[0], height, width, channel, padding);
        auto transpose_src_shape = vec_int32_to_int64(from_tensor->get_shape());
        // new shape is padding after shape
        transpose_src_shape[0] = 1;
        transpose_src_shape[1] = channel;
        vaip_core::transpose_i8(/*src */ &ret_data[0],
                                /*dst */
                                reinterpret_cast<int8_t*>(data_to),
                                transpose_src_shape, order);
      }
    }
    // int8 -> uint8
    else if (is_int_8_data(*from_tensor) && is_uint_8_data(*to_tensor)) {
      if (!op_is_pad && !op.is_layout_transform()) {
        MY_LOG(1) << "int8->uint8 ";
        int82uint8(/*dst*/ reinterpret_cast<uint8_t*>(data_to),
                   /*src */ reinterpret_cast<int8_t*>(data_from),
                   (int)batch_size);
      } else if (!op_is_pad && op.is_layout_transform()) {
        MY_LOG(1) << "int8->uint8 + transpose";
        auto transpose_src_shape = vec_int32_to_int64(from_tensor->get_shape());
        transpose_src_shape[0] = 1;
        auto ret_data = std::vector<int8_t>(batch_size, 0);
        vaip_core::transpose_i8(/*src */
                                reinterpret_cast<int8_t*>(data_from),
                                /*dst */
                                &ret_data[0], transpose_src_shape, order);
        int82uint8(/*dst*/ reinterpret_cast<uint8_t*>(data_to),
                   /*src */ &ret_data[0], (int)batch_size);
      } else {
        LOG(FATAL) << "TODO : int8 -> uint8 pad " << op_is_pad
                   << " layout transfrom " << op.is_layout_transform();
      }
    }
    // uint8 -> uint8
    else if (is_uint_8_data(*from_tensor) && is_uint_8_data(*to_tensor)) {
      // test case: edgenext_small_rw 9df0a329,see vaip#1231
      if (!op_is_pad && !op.is_layout_transform()) {
        MY_LOG(1) << "uint8->uint8 , memcpy";
        std::memcpy(/*dst*/ reinterpret_cast<uint8_t*>(data_to),
                    /*src */ reinterpret_cast<uint8_t*>(data_from),
                    batch_size * sizeof(uint8_t));
      } else if (!op_is_pad && op.is_layout_transform()) {
        MY_LOG(1) << "uint8->uint8 + transpose";
        auto transpose_src_shape = vec_int32_to_int64(from_tensor->get_shape());
        transpose_src_shape[0] = 1;
        vaip_core::transpose_ui8(/*src */
                                 reinterpret_cast<uint8_t*>(data_from),
                                 /*dst */
                                 reinterpret_cast<uint8_t*>(data_to),
                                 transpose_src_shape, order);
      } else {
        LOG(FATAL) << "TODO : uint8 -> uint8 pad " << op_is_pad
                   << " layout transfrom " << op.is_layout_transform();
      }
    }
    // int16 -> int16
    else if (is_int_16_data(*from_tensor) && is_int_16_data(*to_tensor)) {
      // int16->int16
      if (op_is_pad && op.is_layout_transform()) {
        MY_LOG(1) << "int16->int16 : pad && transpose ";
        // only IPU input, maybe pad + layout_transform for PT model, from
        // onnx   -> xir
        auto vart_tensor_shape = to_tensor->get_shape();
        // xir input layout is NHWC
        CHECK_EQ(vart_tensor_shape.size(), 4u)
            << "pad only support 4-dims (N3HW->N4HW)";
        CHECK_EQ(padding.size(), 8)
            << "shape size is 4, paddings size must be 8." << op.DebugString();

        auto height = vart_tensor_shape.at(1);
        auto width = vart_tensor_shape.at(2);
        auto channel_before_pad = vart_tensor_shape.at(3);
        auto channel = channel_before_pad + padding[6] + padding[7];

        CHECK_GT(channel_before_pad, 0) << "channel should be > 0";
        batch_size = batch_size * channel / channel_before_pad;

        auto ret_data = std::vector<int16_t>(batch_size, 0);
        pad_c(&ret_data[0], reinterpret_cast<int16_t*>(data_from), height,
              width, channel, padding);
        auto transpose_src_shape = vec_int32_to_int64(from_tensor->get_shape());
        // new shape is padding after shape
        transpose_src_shape[0] = 1;
        transpose_src_shape[1] = channel;
        vaip_core::transpose_i16(&ret_data[0],
                                 reinterpret_cast<int16_t*>(data_to),
                                 transpose_src_shape, order);

      } else if (!op_is_pad && op.is_layout_transform()) {
        MY_LOG(1) << "int16->int16 , only transpose";
        auto transpose_src_shape = vec_int32_to_int64(from_tensor->get_shape());
        transpose_src_shape[0] = 1;
        vaip_core::transpose_i16(reinterpret_cast<int16_t*>(data_from),
                                 reinterpret_cast<int16_t*>(data_to),
                                 transpose_src_shape, order);

      } else if (op_is_pad && !op.is_layout_transform()) {
        LOG(FATAL) << "TODO: int16->int16 , only pad";
      } else {
        MY_LOG(1) << "int16_t->int16_t , memcpy";
        std::memcpy(reinterpret_cast<int16_t*>(data_to),
                    reinterpret_cast<int16_t*>(data_from),
                    batch_size * sizeof(int16_t));
      }
    }
    // uint16 -> uint16
    else if (is_uint_16_data(*from_tensor) && is_uint_16_data(*to_tensor)) {
      if (!op_is_pad && op.is_layout_transform()) {
        MY_LOG(1) << "uint16->uint16 , only transpose";
        auto transpose_src_shape = vec_int32_to_int64(from_tensor->get_shape());
        transpose_src_shape[0] = 1;
        vaip_core::transpose_u16(reinterpret_cast<uint16_t*>(data_from),
                                 reinterpret_cast<uint16_t*>(data_to),
                                 transpose_src_shape, order);
      } else if (!op_is_pad && !op.is_layout_transform()) {
        MY_LOG(1) << "uint16->uint16 , memcpy";
        std::memcpy(reinterpret_cast<uint16_t*>(data_to),
                    reinterpret_cast<uint16_t*>(data_from),
                    batch_size * sizeof(uint16_t));
      } else {
        LOG(FATAL) << "TODO : uint16 -> uint16 pad " << op_is_pad
                   << " layout transfrom " << op.is_layout_transform();
      }
    }
    // float32 -> bfloat16
    else if (is_float_32_data(*from_tensor) && is_bfloat_16_data(*to_tensor)) {
      CHECK(!op.fix2float());
      CHECK(op.float2fix());
      if (op.quantize_linear() || op.dequantize_linear()) {
        LOG(FATAL)
            << "Check compiled.xmodel boundary tensor name, maybe it's wrong, "
               "if really need do quantize_linear or dequantize_linear,  "
            << "TODO : float32 -> bfloat16 pad " << op_is_pad
            << " layout transfrom " << op.is_layout_transform()
            << " quantize_linear " << op.quantize_linear()
            << " dequantize_linear " << op.dequantize_linear();
      }

      if (!op_is_pad && op.is_layout_transform()) {
        MY_LOG(1) << "float32->bfloat16 , only transpose";
        auto transpose_src_shape = vec_int32_to_int64(from_tensor->get_shape());
        transpose_src_shape[0] = 1;
        auto ret_data = std::vector<float>(batch_size, 0);
        vaip_core::transpose_f(reinterpret_cast<float*>(data_from),
                               &ret_data[0], transpose_src_shape, order);
        float2bfloat16(/*dst*/ reinterpret_cast<xir::bfloat16_t*>(data_to),
                       &ret_data[0], (int)batch_size);
      } else if (!op_is_pad && !op.is_layout_transform()) {
        MY_LOG(1) << "only float32->bfloat16";
        float2bfloat16(/*dst*/ reinterpret_cast<xir::bfloat16_t*>(data_to),
                       /*src */ reinterpret_cast<float*>(data_from),
                       (int)batch_size);
      } else {
        LOG(FATAL) << "TODO : float32 -> bfloat16 pad " << op_is_pad
                   << " layout transfrom " << op.is_layout_transform();
      }
    }
    // bfloat16 -> float32
    else if (is_bfloat_16_data(*from_tensor) && is_float_32_data(*to_tensor)) {
      CHECK(op.fix2float());
      CHECK(!op.float2fix());
      if (op.quantize_linear() || op.dequantize_linear()) {
        LOG(FATAL)
            << "Check compiled.xmodel boundary tensor name, maybe it's wrong, "
               "if really need do quantize_linear or dequantize_linear,  "
            << "TODO : bfloat16 -> float32 pad " << op_is_pad
            << " layout transfrom " << op.is_layout_transform()
            << " quantize_linear " << op.quantize_linear()
            << " dequantize_linear " << op.dequantize_linear();
      }

      if (!op_is_pad && op.is_layout_transform()) {
        MY_LOG(1) << "bfloat16->float32 , only transpose";
        auto transpose_src_shape = vec_int32_to_int64(from_tensor->get_shape());
        transpose_src_shape[0] = 1;
        auto ret_data = std::vector<float>(batch_size, 0);
        bfloat16_2_float(/*dst*/ &ret_data[0],
                         reinterpret_cast<xir::bfloat16_t*>(data_from),
                         (int)batch_size);
        vaip_core::transpose_f(&ret_data[0], reinterpret_cast<float*>(data_to),
                               transpose_src_shape, order);
      } else if (!op_is_pad && !op.is_layout_transform()) {
        MY_LOG(1) << "only bfloat16->float32";
        bfloat16_2_float(/*dst*/ reinterpret_cast<float*>(data_to),
                         /*src */ reinterpret_cast<xir::bfloat16_t*>(data_from),
                         (int)batch_size);
      } else {
        LOG(FATAL) << "TODO : bfloat16 -> float32 pad " << op_is_pad
                   << " layout transfrom " << op.is_layout_transform();
      }
    }
    // bfloat16 -> bfloat16
    else if (is_bfloat_16_data(*from_tensor) && is_bfloat_16_data(*to_tensor)) {
      if (op.quantize_linear() || op.dequantize_linear()) {
        LOG(FATAL)
            << "Check compiled.xmodel boundary tensor name, maybe it's wrong, "
               "if really need do quantize_linear or dequantize_linear,  "
            << "TODO : bfloat16 -> bfloat16 pad " << op_is_pad
            << " layout transfrom " << op.is_layout_transform()
            << " quantize_linear " << op.quantize_linear()
            << " dequantize_linear " << op.dequantize_linear();
      }

      if (!op_is_pad && op.is_layout_transform()) {
        MY_LOG(1) << "bfloat16->bfloat16 , only transpose";
        auto transpose_src_shape = vec_int32_to_int64(from_tensor->get_shape());
        transpose_src_shape[0] = 1;
        vaip_core::transpose_bf16(reinterpret_cast<xir::bfloat16_t*>(data_from),
                                  reinterpret_cast<xir::bfloat16_t*>(data_to),
                                  transpose_src_shape, order);
      } else if (!op_is_pad && !op.is_layout_transform()) {
        MY_LOG(1) << "only bfloat16->bfloat16";
        std::memcpy(/*dst*/ reinterpret_cast<xir::bfloat16_t*>(data_to),
                    /*src */ reinterpret_cast<xir::bfloat16_t*>(data_from),
                    batch_size * sizeof(xir::bfloat16_t));
      } else {
        LOG(FATAL) << "TODO : bfloat16 -> bfloat16 pad " << op_is_pad
                   << " layout transfrom " << op.is_layout_transform();
      }
    } else {
      LOG(FATAL) << "TODO: Not support data type. from: ("
                 << from_tensor->get_data_type().type << ","
                 << from_tensor->get_data_type().bit_width << ") to ("
                 << to_tensor->get_data_type().type << ","
                 << to_tensor->get_data_type().bit_width << ")";
    }
    if (ENV_PARAM(XLNX_ENABLE_DUMP)) {
      auto from_dump_batch_size = batch_size;
      if (is_uint_8_data(*from_tensor) || is_int_8_data(*from_tensor)) {
        from_dump_batch_size = batch_size;
      } else if (is_int_16_data(*from_tensor) ||
                 is_uint_16_data(*from_tensor) ||
                 is_bfloat_16_data(*from_tensor)) {
        from_dump_batch_size = batch_size * 2;
      } else if (is_float_32_data(*from_tensor)) {
        from_dump_batch_size = batch_size * 4;
      } else {
        MY_LOG(1) << "please check the from_tensor elem_type, default single "
                     "elem size is 8bit.";
      }
      auto from_filename = "onnx-dpu_interface_from_" +
                           get_tensor_name(from_tensor) + "_batch_" +
                           std::to_string(index) + ".bin";
      auto mode =
          std::ios_base::out | std::ios_base::binary | std::ios_base::trunc;
      CHECK(std::ofstream(from_filename, mode)
                .write((char*)data_from, from_dump_batch_size)
                .good())
          << " faild to write to " << from_filename;
      MY_LOG(1) << "from_filename: " << from_filename;

      auto to_dump_batch_size = batch_size;
      if (is_uint_8_data(*to_tensor) || is_int_8_data(*to_tensor)) {
        to_dump_batch_size = batch_size;
      } else if (is_int_16_data(*to_tensor) || is_uint_16_data(*to_tensor) ||
                 is_bfloat_16_data(*to_tensor)) {
        to_dump_batch_size = batch_size * 2;
      } else if (is_float_32_data(*to_tensor)) {
        to_dump_batch_size = batch_size * 4;
      } else {
        MY_LOG(1) << "please check the to_tensor elem_type, default single "
                     "elem size is 8bit.";
      }
      auto to_filename = "onnx-dpu_interface_to_" + get_tensor_name(to_tensor) +
                         "_batch_" + std::to_string(index) + ".bin";
      CHECK(std::ofstream(to_filename, mode)
                .write((char*)data_to, to_dump_batch_size)
                .good())
          << " faild to write to " << to_filename;
      MY_LOG(1) << "to_filename: " << to_filename;
    }
  }
}

} // namespace vaip_dpu_custom_op
