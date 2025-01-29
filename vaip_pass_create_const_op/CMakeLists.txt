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
vai_add_library(
  NAME
  pass_create_const_op
  INCLUDE_DIR
  include
  SRCS
  src/const_fold_rule.hpp
  src/const_fold_rule.cpp
  src/create_const_op.cpp
  src/pass_main.cpp)

target_link_libraries(pass_create_const_op PRIVATE vaip::core vart::util
                                                   glog::glog)
if(MSVC)
  set_source_files_properties(src/create_const_op.cpp PROPERTIES COMPILE_FLAGS
                                                                 "/wd4251")
else(MSVC)
  set_source_files_properties(src/create_const_op.cpp PROPERTIES COMPILE_FLAGS
                                                                 -O3)
endif(MSVC)
target_compile_definitions(pass_create_const_op
                           PRIVATE "-DVAIP_USE_DEPRECATED_API=1")
