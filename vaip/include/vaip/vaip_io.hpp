/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */
#pragma once
#include <memory>
#include <string>
#include <vector>
#ifndef VAIP_DLL_SPEC
#  if defined(_WIN32) || defined(_WIN64)
#    define VAIP_DLL_SPEC __declspec(dllexport)
#  else
#    define VAIP_DLL_SPEC __attribute__((visibility("default")))
#  endif
#endif
namespace vaip_core {

class IStreamReader {
public:
  virtual size_t read(char* data, size_t size) = 0;

public:
  static std::unique_ptr<IStreamReader> from_bytes(std::vector<char>&);
  static std::unique_ptr<IStreamReader> from_FILE(FILE*);
};
class IStreamWriter {
public:
  virtual size_t write(const char* data, size_t size) = 0;

public:
  static std::unique_ptr<IStreamWriter> from_bytes(std::vector<char>&);
  static std::unique_ptr<IStreamWriter> from_FILE(FILE*);
  static std::unique_ptr<IStreamWriter>
  from_stream_writers(std::vector<std::unique_ptr<IStreamWriter>>&&);
};
class IStreamWriterBuilder {
public:
  virtual std::unique_ptr<IStreamWriter> build(const std::string&) = 0;
};

} // namespace vaip_core
