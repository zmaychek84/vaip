/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */
#include "file_lock.hpp"
#include <exception>
#include <filesystem>
#include <fstream>
#include <glog/logging.h>
#include <iostream>
#include <vitis/ai/env_config.hpp>
DEF_ENV_PARAM(DEBUG_FILE_LOCK, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_FILE_LOCK) >= n)

namespace vaip_core {
#ifdef ENABLE_BOOST
WithFileLock::WithFileLock(const char* filename) {
  try {
    MY_LOG(1) << "get boost file lock, filename : " << filename;
    if (!std::filesystem::exists(filename)) {
      MY_LOG(1) << "=== create lock file : " << filename;
      std::ofstream ofs(filename);
      ofs.close();
    }

    lock_ = boost::interprocess::file_lock(filename);
    lock_.lock();
    MY_LOG(1) << "get boost file lock success, filename : " << filename
              << " lock : " << &lock_;
  } catch (const std::exception& e) {
    std::cerr << "exception occurs : " << e.what() << "\n";
  }
}
WithFileLock::~WithFileLock() {
  try {
    MY_LOG(1) << "unlock boost file lock ... " << &lock_;
    lock_.unlock();
  } catch (const std::exception& e) {
    std::cerr << "exception occurs : " << e.what() << "\n";
  }
}
#else
static std::mutex& get_mutex_lock() {
  MY_LOG(1) << "get std::mutex lock ... ";
  static std::mutex mutex;
  MY_LOG(1) << "get std::mutex lock success ... ";
  return mutex;
}
WithFileLock::WithFileLock(const char* filename) : lock_(get_mutex_lock()) {}
WithFileLock::~WithFileLock() {
  MY_LOG(1) << "unlock std::mutex lock ... " << std::endl;
}
#endif
} // namespace vaip_core
