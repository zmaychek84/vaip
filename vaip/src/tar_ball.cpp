#include <cstdio>
#include <fstream>
#include <glog/logging.h>
// clang-format off
#define DIR void
#ifdef _WIN32
#else
struct stat {};
#endif
#include "../../3rd-party/tar/src/tar.h"
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

namespace vaip_core {
#define EIGHT_SPACE "        "
static void my_strncpy(char* dst, const char* src, size_t len) {
  // always get a strange error L"Buffer is too small"
  // CHECK(strncpy_s(dst, len, src, len+1) == 0);
  for (auto i = 0u; i < len; ++i) {
    dst[i] = src[i];
  }
}

#define safe_sprintf(a, fmt, c) snprintf(a, sizeof(a), fmt, c)
void tarball_write_file(std::ostream& stream, const std::string& filename,
                        const std::vector<char>& data) {
  auto now = std::chrono::system_clock::now();
  std::time_t now_sec = std::chrono::system_clock::to_time_t(now);
  uint64_t mtime = static_cast<uint64_t>(now_sec);
  block block;
  memset(&block.buffer[0], 0, sizeof(block.buffer));
  static_assert(sizeof(block) == 512);
  static_assert(sizeof(EIGHT_SPACE) == 9);
  auto& header = block.header;
  char typeflag = '0';
  bool is_long_name = filename.size() >= sizeof(header.name);
  if (!is_long_name) {
    typeflag = '0';
    my_strncpy(header.name, filename.c_str(), filename.size());
  } else {
    typeflag = 'L';
    my_strncpy(header.name, "././@LongLink", 13);
  }
  my_strncpy(header.mode, "0000644", 7);
  my_strncpy(header.uid, "0006717", 7);
  my_strncpy(header.gid, "0000112", 7);
  if (!is_long_name) {
    safe_sprintf(header.size, "%011lo", (unsigned long)data.size());
  } else {
    safe_sprintf(header.size, "%011lo", (unsigned long)filename.size());
  }
  safe_sprintf(header.mtime, "%011llo", (long long unsigned int)mtime);
  my_strncpy(header.chksum, EIGHT_SPACE, 8);
  header.typeflag = typeflag;
  my_strncpy(header.magic, OLDGNU_MAGIC, 7);
  my_strncpy(header.uname, "abcdefg", 7);
  // my_strncpy(header.gname, "hijklmn", 7);
  unsigned int checksum_value = 0;
  for (unsigned int i = 0; i != sizeof(block.buffer); ++i) {
    checksum_value += (uint8_t)block.buffer[i];
  }
  safe_sprintf(header.chksum, "%06o", checksum_value);
  header.chksum[7] = ' ';
  auto size = data.size();
  CHECK(stream.write(&block.buffer[0], sizeof(block)).good())
      << "failed to write header. filename=" << filename << " size=" << size;
  if (filename.size() >= sizeof(header.name)) {
    auto size = filename.size();
    CHECK(stream.write(filename.data(), size).good())
        << "failed to write data. filename=" << filename << " size=" << size;
    auto const padding_size{512u - static_cast<unsigned int>(size % 512)};
    const char padding_data[512] = {0};
    if (padding_size != 512) {
      CHECK(stream.write(&padding_data[0], padding_size).good())
          << "failed to write padding. filename=" << filename
          << " size=" << padding_size;
    }
    // write header again
    my_strncpy(header.name, filename.c_str(), sizeof(header.name));
    my_strncpy(header.chksum, EIGHT_SPACE, 8);
    header.typeflag = '0';
    safe_sprintf(header.size, "%011lo", (unsigned long)data.size());
    unsigned int checksum_value = 0;
    for (unsigned int i = 0; i != sizeof(block.buffer); ++i) {
      checksum_value += (uint8_t)block.buffer[i];
    }
    safe_sprintf(header.chksum, "%06o", checksum_value);
    header.chksum[7] = ' ';
    CHECK(stream.write(&block.buffer[0], sizeof(block)).good())
        << "failed to write header. filename=" << filename << " size=" << size;
  }
  if (size == 0) {
    return;
  }
  CHECK(stream.write(data.data(), size).good())
      << "failed to write data. filename=" << filename << " size=" << size;
  auto const padding_size{512u - static_cast<unsigned int>(size % 512)};
  const char padding_data[512] = {0};
  if (padding_size != 512) {
    CHECK(stream.write(&padding_data[0], padding_size).good())
        << "failed to write padding. filename=" << filename
        << " size=" << padding_size;
  }
}
void tarball_end(std::ostream& stream) {
  const char padding_data[512] = {0};
  for (auto i = 0; i < 2; ++i) {
    CHECK(stream.write(&padding_data[0], 512).good())
        << "failed to write padding. i=" << i;
  }
}
/*
std::pair<std::string, std::vector<char>>
tarball_read_file(std::ifstream& stream) {
  block block;
  CHECK(stream.read(&block.buffer[0], sizeof(block)).good())
      << "failed to read header";
  auto& header = block.header;
  std::string filename(header.name);
  if (filename.size() >= 100u) {
    filename.resize(100u);
  }
  if (filename.empty()) {
    return {filename, {}};
  }
  auto size = std::stoul(header.size, nullptr, 8);
  if (size == 0) {
    return {filename, {}};
  }
  std::vector<char> data(size);
  CHECK(stream.read(data.data(), size).good())
      << "failed to read data. filename=" << filename << " size=" << size;
  auto const padding_size{512u - static_cast<unsigned int>(size % 512)};
  char padding_data[512] = {0};
  if (padding_size != 512) {
    CHECK(stream.read(&padding_data[0], padding_size).good())
        << "failed to read padding. filename=" << filename
        << " size=" << padding_size;
  }
  return {filename, data};
}*/
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
std::pair<std::string, std::vector<char>>
tarball_read_file_from_memory(const char*& p, size_t& size) {
  block* block = (union block*)(p);
  auto check_ok = tar_checksum(block);
  if (check_ok == 0) {
    return {"", {}};
  }
  CHECK_EQ(check_ok, 1) << " not valid tar mem: " << size << " bytes";
  p += sizeof(*block);
  size -= sizeof(*block);
  auto* header = &block->header;
  std::string filename(header->name);
  unsigned long size_ = 0u;
  if (header->typeflag == 'L') {
    size_ = std::stoul(header->size, nullptr, 8);
    CHECK_LE(size_, size) << "buffer overflow. size_=" << size_
                          << " size=" << size;
    filename.resize(size_);
    memcpy(filename.data(), p, size_);
    p += size_;
    size -= size_;
    auto const padding_size{512u - static_cast<unsigned int>(size_ % 512)};
    if (padding_size != 512) {
      p += padding_size;
      size -= padding_size;
    }
    block = (union block*)(p);
    auto check_ok = tar_checksum(block);
    CHECK_EQ(check_ok, 1) << " not valid tar mem: " << size << " bytes";
    header = &block->header;
    p += sizeof(*block);
    size -= sizeof(*block);
  }
  size_ = std::stoul(header->size, nullptr, 8);
  CHECK_LE(size_, size) << "buffer overflow. size_=" << size_
                        << " size=" << size;
  if (size_ == 0) {
    return {filename, {}};
  }
  std::vector<char> data(size_);
  memcpy(data.data(), p, size_);
  p += size_;
  size -= size_;
  auto const padding_size{512u - static_cast<unsigned int>(size_ % 512)};
  if (padding_size != 512) {
    p += padding_size;
    size -= padding_size;
  }
  return {filename, data};
}
} // namespace vaip_core
