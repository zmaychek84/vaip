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
  pass_fuse_xmodel
  INCLUDE_DIR
  include
  SRCS
  src/pass_main.cpp
  src/fuse_xmodel.cpp
  src/fuse_xmodel.hpp)

target_link_libraries(pass_fuse_xmodel PRIVATE vaip::core glog::glog vart::util)
