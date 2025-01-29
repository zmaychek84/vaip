/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */
#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <string>

#ifdef __GNUC__
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wunused-function"
#endif

/* PP HW Kernel to PS Kernel Map
 *
 * PDI0 - PP_KERNEL_0
 * PDI1 - PP_KERNEL_1
 * PDI2 - PP_KERNEL_2
 */
static const std::map<int, std::string> PPKernel2PDIMap = {
    {PP_FD_PRE, "PP_KERNEL_0"},
    {PP_SSIM_PRE, "PP_KERNEL_0"},
    {PP_EGC_PRE, "PP_KERNEL_0"},
    {PP_EGC_POST, "PP_KERNEL_0"},
    {PP_RESIZE_DOWN, "PP_KERNEL_0"},
    {PP_NORM, "PP_KERNEL_0"},
    {PP_CLAMP, "PP_KERNEL_0"},
    {PP_PIXELWISE_SELECT, "PP_KERNEL_1"},
    {PP_BLENDING, "PP_KERNEL_1"},
    {PP_ROW_FILTER, "PP_KERNEL_1"},
    {PP_TOPK, "PP_KERNEL_1"},
    {PP_SOFTMAX, "PP_KERNEL_1"},
    {PP_ROW_FILTER_1CH, "PP_KERNEL_1"},
    {PP_RESIZE_UP, "PP_KERNEL_1"},
    {PP_MASK_GEN, "PP_KERNEL_1"},
    {PP_MASK_GEN_TRACK_PARAM_COMP, "PP_KERNEL_1"},
    {PP_PWS_RD, "PP_KERNEL_1"},
    {PP_MIN_MAX, "PP_KERNEL_1"},
    {PP_RESIZEUP_MINMAX_FUSION, "PP_KERNEL_1"},
    {PP_FD_POST, "PP_KERNEL_1"},
};

static std::string PPGetPSKernelName(int opcode) {
  auto it = PPKernel2PDIMap.find(opcode);
  if (it != PPKernel2PDIMap.end())
    return it->second;
  std::cout << "PP Kernel Opcode not found" << std::endl;
  return "PP_KERNEL_0";
}

template <int FBITS_ALPHA = 0, int FBITS_BETA = 4>
static void get_alphabeta(float mean[4], float stddev[4],
                          unsigned char alpha[4], char beta[4]) {
  for (int i = 0; i < 4; i++) {
    if (i < 3) {
      float a_v = mean[i] * (1 << FBITS_ALPHA);
      float b_v = (stddev[i]) * (1 << FBITS_BETA);

      alpha[i] = (unsigned char)a_v;
      beta[i] = (char)b_v;

      assert((a_v < (1 << 8)) && "alpha values exceeds 8 bit precison");
      assert((b_v < (1 << 8)) && "beta values exceeds 8 bit precison");
    } else {
      alpha[i] = 0;
      beta[i] = 0;
    }
  }
}

template <typename T = uint8_t>
static int check_result(T* out_buf, size_t bytesize, T* refdata,
                        std::string& test_case_name) {
  int acceptableError = 5, errCount = 0, maxError = 0;
  for (int i = 0; i < bytesize; i++) {
    int error = abs(refdata[i] - out_buf[i]);
    maxError = (error > maxError) ? error : maxError;
    if (error > acceptableError) {
      errCount++;
    }
  }

  if (errCount > 0) {
    std::cout << test_case_name
              << " :  Test failed!, with errCount = " << errCount
              << ", maxError = " << maxError << std::endl;
  } else {
    std::cout << test_case_name << " : Test passed!" << std::endl;
  }
  return 0;
}

static std::string shape_to_string(const std::vector<int64_t>& shape) {
  std::ostringstream str;
  str << "[";
  int c = 0;
  for (auto s : shape) {
    if (c != 0) {
      str << ",";
    }
    str << s;
    c = c + 1;
  }
  str << "]";
  return str.str();
}

static std::vector<int64_t> string_to_shape(std::string shape) {
  // remove brackets
  shape.erase(0, 1);
  shape.erase(shape.end() - 1);
  std::replace(shape.begin(), shape.end(), ',', ' ');
  // convert to int
  std::stringstream ss(shape);
  std::vector<int64_t> shape_vector;
  for (int64_t dim; ss >> dim;) {
    shape_vector.push_back(dim);
    if (ss.peek() == ' ')
      ss.ignore();
  }
  return shape_vector;
}

static std::vector<float> string_to_float_shape(std::string input,
                                                char delimiter = ',') {
  // remove brackets
  input.erase(0, 1);
  input.erase(input.end() - 1);

  std::vector<float> result;
  std::istringstream iss(input);
  std::string token;
  // convert to float
  while (std::getline(iss, token, delimiter)) {
    try {
      float value = std::stof(token);
      result.push_back(value);
    } catch (const std::invalid_argument& e) {
      LOG(FATAL) << "-- Invalid float value: " << e.what();
    }
  }
  return result;
}

typedef union value_convert {
  std::uint32_t u;
  float f;
} value_convert_t;

static std::uint32_t f_to_u(float data) {
  value_convert_t vc{};
  vc.f = data;
  return vc.u;
}

static float u_to_f(std::uint32_t data) {
  value_convert_t vc{};
  vc.u = data;
  return vc.f;
}

static float f_to_bf(float data) {
  std::uint32_t u = f_to_u(data);
  u = (u + 0x7fff) & 0xFFFF0000;
  return u_to_f(u);
}

#ifdef __GNUC__
#  pragma GCC diagnostic pop
#endif