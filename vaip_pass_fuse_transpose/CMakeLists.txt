#
#  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
#  Licensed under the MIT License.
#


cmake_minimum_required(VERSION 3.17)
project(
  vaip
  VERSION 1.0.0
  LANGUAGES C CXX)

vai_add_library(
  NAME
  pass_fuse_transpose
  INCLUDE_DIR
  include
  SRCS
  src/pass_main.cpp
  src/transpose_f.cpp)
if(MSVC)
  set_source_files_properties(src/transpose_f.cpp PROPERTIES COMPILE_FLAGS
                                                             "/wd4251")
else(MSVC)
  set_source_files_properties(src/transpose_f.cpp PROPERTIES COMPILE_FLAGS -O3)
endif(MSVC)

target_link_libraries(pass_fuse_transpose PRIVATE vaip::core vart::util
                                                  glog::glog)
