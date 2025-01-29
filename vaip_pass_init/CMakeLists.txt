#
#  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
#  Licensed under the MIT License.
#


cmake_minimum_required(VERSION 3.17)
project(
  vaip
  VERSION 1.0.0
  LANGUAGES C CXX)

find_package(unilog REQUIRED)
find_package(xir REQUIRED)
find_package(vart REQUIRED util)
vai_add_library(NAME pass_init INCLUDE_DIR include SRCS src/vaip_pass_init.cpp)

target_link_libraries(pass_init PRIVATE vaip::core vart::util glog::glog vaip::vaip_pattern_zoo)
