#pragma once

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#endif

#include "onnxruntime_api.hpp"
#include "vaiml_client.h"
#include "vaip/vaip.hpp"
#include "vitis/ai/env_config.hpp"
#include <cassert>
#include <codecvt>
#include <deque>
#include <filesystem>
#include <fstream>
#include <glog/logging.h>
#include <istream>
#include <memory>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>
#include <string>

#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_VAIML_PARTITION) >= n)
DEF_ENV_PARAM(DEBUG_VAIML_PARTITION, "1");

#include <iostream>
template <typename... Args> inline void vaiml_print_func(Args&&... args) {
  std::cout << std::fixed << std::setprecision(6);
  (std::cout << ... << args);
}

#define VAIML_DEBUG_PRINT(...)                                                 \
  if (1 && (ENV_PARAM(DEBUG_VAIML_PARTITION) >= 1)) {                          \
    vaiml_print_func(__VA_ARGS__);                                             \
    std::cout << std::endl;                                                    \
  }

#define VAIML_DEBUG_PRINT2(...)                                                \
  if (1 && (ENV_PARAM(DEBUG_VAIML_PARTITION) >= 2)) {                          \
    vaiml_print_func(__VA_ARGS__);                                             \
    std::cout << std::endl;                                                    \
  }

#define VAIP_CUSTOM_OP_VAIML_PROFILING

namespace vaip_vaiml_custom_op {

// only Backward Compatibility (mepTable and targetproto)
inline bool is_absolute_path(const std::string& path) {
#ifdef _WIN32
  if (path.size() > 1 && path[1] == ':') {
    return true;
  }
#else
  if (path.size() > 0 && path[0] == '/') {
    return true;
  }
#endif
  return false;
}

// inline std::string
// get_xclbin_fullpath(std::shared_ptr<const PassContext> context,
//                     const std::string& xclbin) {
//   VAIML_DEBUG_PRINT("    xclbin: ", xclbin);
//   // Check if xclbin is cached
//   std::string xclbin_config_path =
//       context->xclbin_path_to_cache_files(std::filesystem::path(xclbin))
//           .string();
//   VAIML_DEBUG_PRINT("    xclbin_config_path: ", xclbin_config_path);
//   if (!xclbin_config_path.empty()) {
//     if (fs::exists(xclbin_config_path) &&
//         (xclbin_config_path.find(".xclbin") != std::string::npos)) {
//       return (xclbin_config_path);
//     }
//   }
//
//   return xclbin;
// }

inline void read_file_c8(int8_t* buf, std::string file_name, uint32_t size) {
  //    std::ifstream ifs(file_name, std::ifstream::binary);
  //    ifs.read((char*)buf, size);
  //    ifs.close();
  uint32_t* buf_u32 = (uint32_t*)buf;
  std::ifstream ifs;
  ifs.open(file_name);
  if (!ifs)
    std::cout << "Failed to open file" << file_name << " for reading!!"
              << std::endl;
  else
    std::cout << "Opened file " << file_name << " for reading!!" << std::endl;
  for (int i = 0; i < size / sizeof(uint32_t); i++)
    ifs >> std::hex >> buf_u32[i];
  ifs.close();
}
inline void q_int8(const float* in, int8_t* out, float scale, int8_t zp,
                   size_t size) {
  float inv_scale = 1.0f / scale;
  for (int i = 0; i < size; i++) {
    float temp = in[i] * inv_scale + zp;
    temp = temp > 255 ? 255 : temp < -128 ? -128 : temp;
    out[i] = static_cast<int8_t>(temp);
  }
}
inline void dq_int8(const int8_t* in, float* out, float scale, int8_t zp,
                    size_t size) {
  for (int i = 0; i < size; i++) {
    float temp;
    temp = static_cast<float>(in[i]);
    temp = (temp - zp) * scale;
    out[i] = temp;
  }
}
inline void write_file(uint32_t* ddr, uint32_t ofm_size,
                       std::string file_name) {
  // printf("ofm_size = %d\n", ofm_size);
  std::ofstream ofm_ofs;
  ofm_ofs.open(file_name, std::ofstream::binary | std::ofstream::out |
                              std::ofstream::trunc);
  if (!ofm_ofs)
    std::cout << "Failed to open file " << file_name << " for writing "
              << ofm_size << " bytes!!" << std::endl;
  else
    std::cout << "Opened file " << file_name << " for writing " << ofm_size
              << " bytes!!" << std::endl;
  for (int i = 0; i < ofm_size / 4; i++)
    ofm_ofs << std::setw(8) << std::hex << std::setfill('0') << (uint32_t)ddr[i]
            << std::endl;
  ofm_ofs.close();
}
enum SUBGRAPH_ID {
  UNKNOWN = -1,
  HT_LN_SG_LSTM = 0,
  // leave space for HT subgraph
  GT_QKV = 10,
  GT_SM_LINEAR_OUT_FEED_FORWARD = 11,
  GT_MATMUL_REDUCE = 13,
  GT_TRANSFORMER_BLOCK = 14, // 3 in 1 subgraph
  GT_LN_MATMUL_ADD_LN = 15,
  GT_FRONT = 16,
  GT_FRONT_MM = 17,
  // GT constant/CPU subgraph
  GT_CPU_OR_CONSTANT = 100,
  GT_NORM_K = 101,
  GT_CACHE_FRAMES_SLICE = 102,
  // HT constant/CPU subgraph
  HT_SLICE = 200,
  HT_CONCAT = 201,
};

template <typename... Args>
std::string str_fmt(const std::string& format, Args... args) {
  int size_s = std::snprintf(nullptr, 0, format.c_str(), args...) +
               1; // Extra space for '\0'
  if (size_s <= 0) {
    throw std::runtime_error("Error during str formatting.");
  }
  auto size = static_cast<size_t>(size_s);
  std::unique_ptr<char[]> buf(new char[size]);
  std::snprintf(buf.get(), size, format.c_str(), args...);
  return std::string(buf.get(),
                     buf.get() + size - 1); // We don't want the '\0' inside
}

} // namespace vaip_vaiml_custom_op
