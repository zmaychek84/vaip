/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */
#include "tar_ball.hpp"
#include <cstdio>
#include <fstream>
#include <glog/logging.h>
#include <iostream>
// clang-format off
#define DIR void
#ifdef _WIN32
#else
struct stat {};
#endif
#include <tar.h>
// clang-format on
#include <chrono>
#include <cstdint>
#include <string.h>
#include <string>
#include <type_traits>
#ifdef __GNUC__
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wformat-truncation"
#endif

#define BLOCKSIZE 512
union block {
  char buffer[BLOCKSIZE];
  HD_USTAR header;
};
namespace vaip_core {
#define EIGHT_SPACE "        "
#define safe_sprintf(a, fmt, c) snprintf(a, sizeof(a), fmt, c)

static void my_strncpy(char* dst, const char* src, size_t len) {
  // always get a strange error L"Buffer is too small"
  // CHECK(strncpy_s(dst, len, src, len+1) == 0);
  for (auto i = 0u; i < len; ++i) {
    dst[i] = src[i];
  }
}
int tar_checksum(block* header) {
  int unsigned_sum = 0; /* the POSIX one :-) */
  int signed_sum = 0;   /* the Sun one :-( */
  int recorded_sum;
  char* p = header->buffer;

  for (auto i = sizeof *header; i-- != 0;) {
    unsigned_sum += (unsigned char)*p;
    signed_sum += (signed char)(*p++);
  }

  if (unsigned_sum == 0)
    return 0;

  /* Adjust checksum to count the "chksum" field as blanks.  */

  for (auto i = sizeof header->header.chksum; i-- != 0;) {
    unsigned_sum -= (unsigned char)header->header.chksum[i];
    signed_sum -= (signed char)(header->header.chksum[i]);
  }
  unsigned_sum += (int)(' ' * sizeof header->header.chksum);
  signed_sum += (int)(' ' * sizeof header->header.chksum);

  auto parsed_sum = std::stoll(header->header.chksum, nullptr, 8);
  if (parsed_sum < 0)
    return -1;

  recorded_sum = (int)parsed_sum;

  if (unsigned_sum != recorded_sum && signed_sum != recorded_sum)
    return -1;

  return 1;
}

int TarWriter::write(IStreamReader* src, size_t size, const std::string& name) {
  const size_t BUFFER_SIZE = 512u;
  char buffer[512] = {0};
  auto now = std::chrono::system_clock::now();
  std::time_t now_sec = std::chrono::system_clock::to_time_t(now);
  uint64_t mtime = static_cast<uint64_t>(now_sec);
  block block;
  memset(&block.buffer[0], 0, sizeof(block.buffer));
  static_assert(sizeof(block) == 512);
  static_assert(sizeof(EIGHT_SPACE) == 9);
  auto& header = block.header;
  char typeflag = '0';
  bool is_long_name = name.size() >= sizeof(header.name);
  if (!is_long_name) {
    typeflag = '0';
    my_strncpy(header.name, name.c_str(), name.size());
  } else {
    typeflag = 'L';
    my_strncpy(header.name, "././@LongLink", 13);
  }
  my_strncpy(header.mode, "0000644", 7);
  my_strncpy(header.uid, "0006717", 7);
  my_strncpy(header.gid, "0000112", 7);
  if (!is_long_name) {
    safe_sprintf(header.size, "%011lo", (unsigned long)size);
  } else {
    safe_sprintf(header.size, "%011lo", (unsigned long)name.size());
  }
  safe_sprintf(header.mtime, "%011llo", (long long unsigned int)mtime);
  my_strncpy(header.chksum, EIGHT_SPACE, 8);
  header.typeflag = typeflag;
  my_strncpy(header.magic, TMAGIC, strlen(TMAGIC));
  my_strncpy(header.uname, "abcdefg", 7);
  // my_strncpy(header.gname, "hijklmn", 7);
  unsigned int checksum_value = 0;
  for (unsigned int i = 0; i != sizeof(block.buffer); ++i) {
    checksum_value += (uint8_t)block.buffer[i];
  }
  safe_sprintf(header.chksum, "%06o", checksum_value);
  header.chksum[7] = ' ';
  auto t = tarball_->write(&block.buffer[0], sizeof(block));
  CHECK(sizeof(block) == t)
      << "failed to write header. name = " << name << " size = " << size;
  if (name.size() >= sizeof(header.name)) {
    auto size = name.size();
    CHECK(size == tarball_->write(name.data(), size))
        << "failed to write data. name = " << name << " size = " << size;
    auto const padding_size{512u - static_cast<unsigned int>(size % 512)};
    const char padding_data[512] = {0};
    if (padding_size != 512) {
      CHECK(padding_size == tarball_->write(&padding_data[0], padding_size))
          << "failed to write padding. name = " << name
          << " size = " << padding_size;
    }
    // write header again
    my_strncpy(header.name, name.c_str(), sizeof(header.name));
    my_strncpy(header.chksum, EIGHT_SPACE, 8);
    header.typeflag = '0';
    safe_sprintf(header.size, "%011lo", (unsigned long)size);
    unsigned int checksum_value = 0;
    for (unsigned int i = 0; i != sizeof(block.buffer); ++i) {
      checksum_value += (uint8_t)block.buffer[i];
    }
    safe_sprintf(header.chksum, "%06o", checksum_value);
    header.chksum[7] = ' ';
    CHECK(sizeof(block) == tarball_->write(&block.buffer[0], sizeof(block)))
        << "failed to write header. name = " << name << " size = " << size;
  }
  if (size == 0) {
    return 0;
  }
  for (auto i = 0; i < size; i += BUFFER_SIZE) {
    auto read_size = src->read(buffer, BUFFER_SIZE);
    CHECK(read_size > 0) << "failed to read file";
    if (read_size < 512) {
      auto padding_size =
          BUFFER_SIZE - static_cast<unsigned int>(read_size % BUFFER_SIZE);
      memset(buffer + read_size, 0, padding_size);
    }
    CHECK(tarball_->write(buffer, BUFFER_SIZE))
        << "failed to write data. name = " << name << " size = " << i;
  }
  return 0;
}
TarWriter::~TarWriter() {
  // tar end
  const char padding_data[512] = {0};
  CHECK(tarball_->write(padding_data, 512) == 512) << "dtor tarWriter failed.";
  CHECK(tarball_->write(padding_data, 512) == 512) << "dtor tarWriter failed.";
}
int TarReader::read(IStreamWriterBuilder* dst_builder) {
  const size_t BUFFER_SIZE = 512u;
  std::vector<char> buffer(BUFFER_SIZE, 0);
  std::vector<char> block_buffer(sizeof(block), 0);

  auto ret = tarball_->read(block_buffer.data(), sizeof(block));
  if (ret != sizeof(block)) {
    return 0;
  }
  block* block = (union block*)(block_buffer.data());
  auto check_ok = tar_checksum(block);
  if (check_ok == 0) {
    return 0;
  }
  CHECK_EQ(check_ok, 1) << "tallball not valid: checksum failed.";
  auto* header = &block->header;
  std::string filename(header->name);
  unsigned long size_ = 0u;
  if (header->typeflag == 'L') {
    size_ = std::stoul(header->size, nullptr, 8);
    filename.resize(size_);
    ret = tarball_->read(filename.data(), size_);
    CHECK(ret == size_) << "buffer overflow. size_=" << size_;

    auto const padding_size{512u - static_cast<unsigned int>(size_ % 512)};
    if (padding_size != 512) {
      auto tmp_buffer = std::vector<char>(padding_size);
      ret = tarball_->read(tmp_buffer.data(), padding_size);
      CHECK(ret == padding_size) << "buffer overflow. size_=" << size_;
    }

    ret = tarball_->read(block_buffer.data(), sizeof(block));
    if (ret != sizeof(block)) {
      return 0;
    }
    block = (union block*)(block_buffer.data());
    auto check_ok = tar_checksum(block);
    CHECK(check_ok) << "tallball not valid: checksum failed.";
    header = &block->header;
  }
  size_ = std::stoul(header->size, nullptr, 8);
  auto dst = dst_builder->build(filename);
  if (size_ == 0) {
    return 1;
  }
  for (int i = 0; i < size_; i += BUFFER_SIZE) {
    ret = tarball_->read(buffer.data(), BUFFER_SIZE);
    CHECK(ret == BUFFER_SIZE)
        << "buffer overflow. name = " << filename << " size_ =" << size_
        << "bytes, read  " << i << "bytes";
    if (i + BUFFER_SIZE <= size_) {
      dst->write(buffer.data(), BUFFER_SIZE);
    } else {
      dst->write(buffer.data(), size_ % BUFFER_SIZE);
    }
  }
  return 1;
}
} // namespace vaip_core
