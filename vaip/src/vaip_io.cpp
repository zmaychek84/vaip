/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */
#include "vaip/vaip_io.hpp"
#include <cstring>
#include <vector>

namespace vaip_core {

// imp
class FileStreamReader : public IStreamReader {
public:
  FileStreamReader(FILE* file) : file_(file) {}

private:
  VAIP_DLL_SPEC size_t read(char* data, size_t size) override final;

private:
  FILE* file_;
};

class FileStreamWriter : public IStreamWriter {
public:
  FileStreamWriter(FILE* file) : file_(file) {}

private:
  VAIP_DLL_SPEC size_t write(const char* data, size_t size) override final;

private:
  FILE* file_;
};

class ByteStreamWriter : public IStreamWriter {
public:
  ByteStreamWriter(std::vector<char>& bytes) : bytes_(bytes) {}

private:
  VAIP_DLL_SPEC size_t write(const char* data, size_t size) override final;

private:
  std::vector<char>& bytes_;
};

class VecStreamWriter : public IStreamWriter {
public:
  VecStreamWriter(std::vector<std::unique_ptr<IStreamWriter>> writers)
      : writers_(std::move(writers)){};

private:
  VAIP_DLL_SPEC size_t write(const char* data, size_t size) override final;

private:
  std::vector<std::unique_ptr<IStreamWriter>> writers_;
};

class ByteStreamReader : public IStreamReader {
public:
  ByteStreamReader(std::vector<char>& bytes) : bytes_(bytes) {}

private:
  VAIP_DLL_SPEC size_t read(char* data, size_t size) override final;

private:
  std::vector<char>& bytes_;
  int pos = 0;
};

//
std::unique_ptr<IStreamReader>
IStreamReader::from_bytes(std::vector<char>& bytes) {
  return std::make_unique<ByteStreamReader>(bytes);
}

std::unique_ptr<IStreamReader> IStreamReader::from_FILE(FILE* file) {
  return std::make_unique<FileStreamReader>(file);
}

std::unique_ptr<IStreamWriter>
IStreamWriter::from_bytes(std::vector<char>& bytes) {
  return std::make_unique<ByteStreamWriter>(bytes);
}
std::unique_ptr<IStreamWriter> IStreamWriter::from_FILE(FILE* file) {
  return std::make_unique<FileStreamWriter>(file);
}

std::unique_ptr<IStreamWriter> IStreamWriter::from_stream_writers(
    std::vector<std::unique_ptr<IStreamWriter>>&& writers) {
  return std::make_unique<VecStreamWriter>(std::move(writers));
}
size_t FileStreamReader::read(char* data, size_t size) {
  return std::fread(data, sizeof(char), size, this->file_);
}

size_t FileStreamWriter::write(const char* data, size_t size) {
  return std::fwrite(data, sizeof(char), size, this->file_);
}

size_t ByteStreamWriter::write(const char* data, size_t size) {
  if (!data || size == 0) {
    return 0;
  }
  bytes_.insert(bytes_.end(), data, data + size);
  return size;
}

size_t VecStreamWriter::write(const char* data, size_t size) {
  for (auto& writer : writers_) {
    writer->write(data, size);
  }
  return size;
}
size_t ByteStreamReader::read(char* data, size_t size) {
  if (pos + size < bytes_.size()) {
    std::memcpy(data, bytes_.data() + pos, size);
    pos = pos + size;
    return size;
  } else {
    int read_size = bytes_.size() - pos;
    std::memcpy(data, bytes_.data() + pos, read_size);
    pos = pos + read_size;
    return read_size;
  }
}

} // namespace vaip_core
