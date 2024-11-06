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

#include <cstring>
#include <iostream>
#include <vector>

namespace vaip_dod_custom_op {

static float bfloat_to_float(uint16_t x) {
  float i = 0;
  uint8_t* src = (uint8_t*)&x;
  uint8_t* tmp = (uint8_t*)&i;
  // copy uint16_t to float (msb)
  std::memcpy(tmp + 2, src, sizeof(uint16_t));
  return i;
}

static uint16_t float_to_bfloat16_1(float x) {
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

static std::vector<std::string> split_string(const std::string& msg,
                                             const std::string& delim = ",") {
  auto start = 0;
  auto end = std::string::npos;
  std::vector<std::string> tokens;
  while (true) {
    end = msg.find(delim, start);
    if (end == std::string::npos) {
      auto sub = msg.substr(start, end - start);
      tokens.push_back(sub);
      break;
    }
    auto sub = msg.substr(start, end - start);
    tokens.push_back(sub);
    start = end + delim.size();
  }
  return tokens;
}

template <typename DType>
static std::vector<DType> string_to_values(std::string str_values) {
  // convert to float
  std::stringstream ss(str_values);
  std::vector<DType> values;
  for (DType dim; ss >> dim;) {
    values.push_back(dim);
    if (ss.peek() == ' ')
      ss.ignore();
  }
  return values;
}

template <typename T>
static std::vector<std::vector<T>>
cs_string_to_nested_list(const std::string& msg) {
  std::vector<std::vector<T>> res;
  std::vector<std::string> tokens = split_string(msg);
  for (const auto& token : tokens) {
    auto values = string_to_values<T>(token);
    res.push_back(std::move(values));
  }
  return res;
}

// Function to convert NCHW to NHWC
template <typename T>
static void convert_NCHW_to_NHWC(const T* input_nchw, T* output_nhwc, int N,
                                 int C, int H, int W, DDTimer& ddtimer) {
  WRAP(auto t1 = GET_TIMESTAMP();)
  for (int n = 0; n < N; ++n) {
    for (int c = 0; c < C; ++c) {
      for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
          // The index in the NCHW format
          int nchw_index = n * C * H * W + c * H * W + h * W + w;
          // The index in the NHWC format
          int nhwc_index = n * H * W * C + h * W * C + w * C + c;
          output_nhwc[nhwc_index] = input_nchw[nchw_index];
        }
      }
    }
  }
  WRAP(auto t2 = GET_TIMESTAMP();)
  WRAP(ddtimer.nchw2nhwc_time.push_back(GET_INTERVAL(t1, t2));)
}

// Function to convert NHWC to NCHW
template <typename T>
static void convert_NHWC_to_NCHW(const T* input_nhwc, T* output_nchwc, int N,
                                 int H, int W, int C, DDTimer& ddtimer) {
  WRAP(auto t1 = GET_TIMESTAMP();)
  for (int n = 0; n < N; ++n) {
    for (int c = 0; c < C; ++c) {
      for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
          // The index in the NCHW format
          int nchw_index = n * C * H * W + c * H * W + h * W + w;
          // The index in the NHWC format
          int nhwc_index = n * H * W * C + h * W * C + w * C + c;
          output_nchwc[nchw_index] = input_nhwc[nhwc_index];
        }
      }
    }
  }
  WRAP(auto t2 = GET_TIMESTAMP();)
  WRAP(ddtimer.nhwc2nchw_time.push_back(GET_INTERVAL(t1, t2));)
}

// Function to convert NHWC to NCHW
template <typename T>
static void convert_HNWC_to_NCHW(const T* input_nhwc, T* output_nchwc, int N,
                                 int H, int W, int C) {
  for (int n = 0; n < N; ++n) {
    for (int c = 0; c < C; ++c) {
      for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
          // The index in the NCHW format
          int nchw_index = n * C * H * W + c * H * W + h * W + w;
          // The index in the NHWC format
          int nhwc_index = n * H * W * C + h * W * C + w * C + c;
          output_nchwc[nchw_index] = input_nhwc[nhwc_index];
        }
      }
    }
  }
}

template <typename T>
static void C4HWtoHWC4(const T* input, T* output, int H, int W, T pad_value) {
  constexpr int C = 4;
  const T* c0_ptr = input + 0 * H * W;
  const T* c1_ptr = input + 1 * H * W;
  const T* c2_ptr = input + 2 * H * W;
  const T* c3_ptr = input + 3 * H * W;
  for (int i = 0; i < H * W; ++i) {
    T c0 = *c0_ptr++;
    T c1 = *c1_ptr++;
    T c2 = *c2_ptr++;
    T c3 = *c3_ptr++;
    output[i * C + 0] = c0;
    output[i * C + 1] = c1;
    output[i * C + 2] = c2;
    output[i * C + 3] = pad_value;
  }
}

template <typename T>
static void C3HWtoHWC8(const T* input, T* output, int H, int W, T pad_value) {
  constexpr int C = 8;
  const T* c0_ptr = input + 0 * H * W;
  const T* c1_ptr = input + 1 * H * W;
  const T* c2_ptr = input + 2 * H * W;
  const T* c3_ptr = input + 3 * H * W;
  for (int i = 0; i < H * W; ++i) {
    T c0 = *c0_ptr++;
    T c1 = *c1_ptr++;
    T c2 = *c2_ptr++;
    T c3 = *c3_ptr++;
    output[i * C + 0] = c0;
    output[i * C + 1] = c1;
    output[i * C + 2] = c2;
    output[i * C + 3] = pad_value;
    output[i * C + 4] = pad_value;
    output[i * C + 5] = pad_value;
    output[i * C + 6] = pad_value;
    output[i * C + 7] = pad_value;
  }
}

template <typename OrtType, typename DoDType>
static bool
C3HW_to_HWC4_conversion_required(const std::vector<OrtType>& ort_shape,
                                 const std::vector<DoDType> dod_shape) {
  // ort_shape : NCHW
  // dod_shape : NHWC
  if ((ort_shape.size() != 4) || (dod_shape.size() != 4) || // 4D tensor
      (ort_shape.at(0) != 1) || (dod_shape.at(0) != 1) ||   // Batch size
      (ort_shape.at(1) != 3) || (dod_shape.at(3) != 4) ||   // Channel size
      (ort_shape.at(2) != dod_shape.at(1)) ||
      (ort_shape.at(3) != dod_shape.at(2))) {
    return false;
  }

  return true;
}

template <typename OrtType, typename DoDType>
static bool
NC3HW_to_HNWC4_conversion_required(const std::vector<OrtType>& ort_shape,
                                   const std::vector<DoDType> dod_shape) {
  // ort_shape : NCHW
  // dod_shape : NHWC
  if ((ort_shape.size() != 4) || (dod_shape.size() != 4) || // 4D tensor
      (ort_shape.at(0) != 1) || (dod_shape.at(1) != 1) ||   // Batch size
      (ort_shape.at(1) != 3) || (dod_shape.at(3) != 4) ||   // Channel size
      (ort_shape.at(2) != dod_shape.at(0)) ||
      (ort_shape.at(3) != dod_shape.at(2))) {
    return false;
  }

  return true;
}

template <typename OrtType, typename DoDType>
static bool
NC3HW_to_HNWC8_conversion_required(const std::vector<OrtType>& ort_shape,
                                   const std::vector<DoDType> dod_shape) {
  // ort_shape : NCHW (c == 3)
  // dod_shape : HNWC (c == 8)
  if ((ort_shape.size() != 4) || (dod_shape.size() != 4) || // 4D tensor
      (ort_shape.at(0) != 1) || (dod_shape.at(1) != 1) ||   // Batch size
      (ort_shape.at(1) != 3) || (dod_shape.at(3) != 8) ||   // Channel size
      (ort_shape.at(2) != dod_shape.at(0)) ||               // Height
      (ort_shape.at(3) != dod_shape.at(2))) {               // Width
    return false;
  }

  return true;
}

template <typename OrtType, typename DoDType>
static bool
C3HW_to_FOLD_ICONV_IFM_required(const std::vector<OrtType>& ort_shape,
                                const std::vector<DoDType> dod_shape) {

  if ((ort_shape.size() != 4) || (dod_shape.size() != 4)) // 4D tensor
    return false;

  if ((ort_shape.at(0) != 1) || (dod_shape.at(0) != 1) ||     // Batch size
      (ort_shape.at(1) != 3) || (dod_shape.at(1) != 3) ||     // Channel size
      (ort_shape.at(2) != 224) || (dod_shape.at(2) != 224) || // H
      (ort_shape.at(3) != 224) || (dod_shape.at(3) != 224)) {
    return false;
  }

  return true;
}

template <typename OrtType, typename DoDType>
static bool
C4HW_to_FORMAT_ICONV_IFM_required(const std::vector<OrtType>& ort_shape,
                                  const std::vector<DoDType> dod_shape) {

  if ((ort_shape.size() != 4) || (dod_shape.size() != 4)) // 4D tensor
    return false;

  if ((ort_shape.at(0) != 1) || (dod_shape.at(0) != 1) ||
      ((ort_shape.at(1) != 4 && ort_shape.at(1) != 64)) ||
      (dod_shape.at(1) != 64) ||
      ((ort_shape.at(2) != 64) && (ort_shape.at(2) != 64)) ||
      (dod_shape.at(2) != 64) ||
      ((ort_shape.at(3) != 64) && (ort_shape.at(3) != 4)) ||
      (dod_shape.at(3) != 4)) {
    return false;
  }

  return true;
}

template <typename OrtType, typename DoDType>
static bool
NCHW_to_NHWC_conversion_required(const std::vector<OrtType>& ort_shape,
                                 const std::vector<DoDType> dod_shape) {
  // ort_shape : NCHW
  // dod_shape : NHWC
  if ((ort_shape.size() != 4) || (dod_shape.size() != 4))
    return false;
  if ((ort_shape.size() != 4) || (dod_shape.size() != 4) || // 4D tensor
      (ort_shape.at(0) != 1) || (dod_shape.at(0) != 1) ||   // Batch size
      (ort_shape.at(1) != dod_shape.at(3)) ||               // Channel size
      (ort_shape.at(2) != dod_shape.at(1)) ||               // Height
      (ort_shape.at(3) != dod_shape.at(2))) {
    return false;
  }
  return true;
}

template <typename OrtType, typename DoDType>
static bool
NCHW_to_HNWC_conversion_required(const std::vector<OrtType>& ort_shape,
                                 const std::vector<DoDType> dod_shape) {
  //  ort_shape : NCHW
  //  dod_shape : NHWC
  if ((ort_shape.size() != 4) || (dod_shape.size() != 4))
    return false;
  if ((ort_shape.at(0) != 1) || (dod_shape.at(1) != 1) || // Batch size
      (ort_shape.at(1) != dod_shape.at(3)) ||             // Channel size
      (ort_shape.at(2) != dod_shape.at(0)) ||             // Height
      (ort_shape.at(3) != dod_shape.at(2))) {
    return false;
  }
  return true;
}

template <typename T>
static void convert_C4HW_to_HWC4(const std::vector<T>& src, std::vector<T>& dst,
                                 int H, int W, T pad_value, DDTimer& ddtimer) {
  WRAP(auto t1 = GET_TIMESTAMP();)
  auto tmp_src(src);
  C4HWtoHWC4(tmp_src.data(), dst.data(), H, W, pad_value);
  WRAP(auto t2 = GET_TIMESTAMP();)
  WRAP(ddtimer.c4hw2hwc4_time.push_back(GET_INTERVAL(t1, t2));)
}

template <typename T>
static void convert_NC3HW_to_HNWC8(const std::vector<T>& src,
                                   std::vector<T>& dst, int H, int W,
                                   T pad_value) {
  auto tmp_src(src);
  C3HWtoHWC8(tmp_src.data(), dst.data(), H, W, pad_value);
}
#if 0
template <typename SrcDType, typename DstDType>
void pad(const SrcDType* src, std::vector<int64_t>& src_shape,
                     DstDType* dst, std::vector<size_t>& dst_shape, int dim,
                     DTypeConvert flag, float scale, float zp) const {
  int elems = 1;
  for (int i = dim + 1; i < src_shape.size(); ++i)
    elems *= src_shape[i];

  int iters = 1;
  for (int i = 0; i < dim; ++i)
    iters *= src_shape[i];

  int dimx = src_shape[dim];
  int dimy = dst_shape[dim];

  int s_off = dimx * elems;
  int d_off = dimy * elems;

  if (flag == DTypeConvert::AS_IS) {
    for (auto f = 0; f < iters; ++f) {
      std::memcpy(dst + (f * d_off), src + (f * s_off),
                  s_off * sizeof(DstDType));
    }
  } else if (flag == DTypeConvert::TO_BF16) {
    for (auto f = 0; f < iters; ++f) {
      for (auto idx = 0; idx < s_off; ++idx) {
        float value = (static_cast<float>(src[f * s_off + idx]) - zp) * scale;
        dst[f * d_off + idx] = float_to_bfloat16_1(value);
      }
    }
    // Assuming attention mask 4D only 1x1x1x77  would be padded to 1x1x1x128
    // with bfloat16(-zp*scale).
    if (dst_shape.size() == 4) {
      for (int i = 77; i < 128; i++) {
        float value = (float)((-zp) * (scale));
        dst[i] = float_to_bfloat16_1(value);
      }
    }
  } else {
    LOG(FATAL) << "- Incorrect flag for Padding input, only to_bf16 conversion "
                  "is possible.";
  }
}

static enum DTypeConvert { TO_BF16 = 1, FROM_BF16 = 2, AS_IS = 3 };

template <typename SrcDType, typename DstDType>
void depad(SrcDType* src, std::vector<size_t>& src_shape,
                       DstDType* dst, std::vector<int64_t>& dst_shape, int dim,
                       DTypeConvert flag, float scale, float zp) const {
  WRAP(auto t1 = GET_TIMESTAMP();)

  int elems = 1;
  for (int i = dim + 1; i < src_shape.size(); ++i)
    elems *= src_shape[i];

  int iters = 1;
  for (int i = 0; i < dim; ++i)
    iters *= src_shape[i];

  int dimx = src_shape[dim];
  int dimy = dst_shape[dim];

  int s_off = dimx * elems;
  int d_off = dimy * elems;

  if (flag == DTypeConvert::AS_IS) {
    for (auto f = 0; f < iters; ++f) {
      std::memcpy(dst + (f * d_off), src + (f * s_off),
                  d_off * sizeof(DstDType));
    }
  } else if (flag == DTypeConvert::FROM_BF16) {
    for (auto f = 0; f < iters; ++f) {
      for (auto idx = 0; idx < d_off; ++idx) {
        float value = bfloat_to_float(src[f * s_off + idx]);
        dst[f * d_off + idx] = static_cast<DstDType>(
            std::clamp(std::roundf(value / scale) + zp, 0.0f, 65535.0f));
      }
    }
  } else {
    LOG(FATAL) << "- Incorrect flag for De-padding output, only from_bf16 "
                  "conversion is possible.";
  }
  WRAP(auto t2 = GET_TIMESTAMP();)
  WRAP(ddtimer_.depad_time.push_back(GET_INTERVAL(t1, t2));)
}
#endif

static dd_timer_t sum_of(const std::vector<dd_timer_t>& vec) {
  return std::accumulate(vec.begin(), vec.end(), dd_timer_t{0});
}

struct DDFileLogger {
private:
  std::ofstream ofs;

public:
  DDFileLogger(const std::string& filename) : ofs(filename) {
    ofs << "compute_time,pre_dod_time,dod_time,post_dod_time,"
        << "pad_time,depad_time,nchw2nhwc_time,nhwc2nchw_time,"
        << "c4hw2hwc4_time,data_conv_time,iconv_prep_time,subgraph"
        << std::endl;
  }

  template <typename T> DDFileLogger& operator<<(const T& item) {
    ofs << item;
    return *this;
  }

  DDFileLogger& operator<<(std::ostream& (*func)(std::ostream&)) {
    ofs << func;
    return *this;
  }
};

} // namespace vaip_dod_custom_op