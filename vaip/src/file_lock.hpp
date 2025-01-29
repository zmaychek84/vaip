/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */
#pragma once
#ifdef ENABLE_BOOST
#  include <boost/interprocess/sync/file_lock.hpp>
#  include <boost/interprocess/sync/scoped_lock.hpp>
namespace vaip_core {
class WithFileLock {
public:
  WithFileLock(const char* filename);
  ~WithFileLock();

private:
  boost::interprocess::file_lock lock_;
};
#else
#  include <mutex>
namespace vaip_core {
class WithFileLock {
public:
  WithFileLock(const char* filename);

private:
  std::lock_guard<std::mutex> lock_;
};
#endif
} // namespace vaip_core
