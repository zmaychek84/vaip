/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */
#pragma once
#include "vaip_io.hpp"
#include <string>

#ifndef VAIP_DLL_SPEC
#  if defined(_WIN32) || defined(_WIN64)
#    define VAIP_DLL_SPEC __declspec(dllexport)
#  else
#    define VAIP_DLL_SPEC __attribute__((visibility("default")))
#  endif
#endif
namespace vaip_core {
class TarWriter {
public:
  TarWriter(IStreamWriter* tall_ball_writer) : tarball_(tall_ball_writer) {}
  VAIP_DLL_SPEC int write(IStreamReader* src, size_t size,
                          const std::string& name);
  VAIP_DLL_SPEC ~TarWriter();

private:
  IStreamWriter* tarball_;
};
class TarReader {
public:
  TarReader(IStreamReader* tall_ball_reader) : tarball_(tall_ball_reader) {}
  VAIP_DLL_SPEC int read(IStreamWriterBuilder* dst_builder);

private:
  IStreamReader* tarball_;
};
} // namespace vaip_core
