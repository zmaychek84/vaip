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
  pass_const_fold_transpose
  INCLUDE_DIR
  include
  SRCS
  src/pass_main.cpp
  src/const_fold_transpose.cpp
  src/const_fold_transpose.hpp)

target_link_libraries(pass_const_fold_transpose PRIVATE vaip::core vart::util
                                                        glog::glog)
target_compile_definitions(pass_const_fold_transpose
                           PRIVATE "-DVAIP_USE_DEPRECATED_API=1")
