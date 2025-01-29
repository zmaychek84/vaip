#
#  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
#  Licensed under the MIT License.
#


cmake_minimum_required(VERSION 3.17)
project(
  vaip
  VERSION 1.0.0
  LANGUAGES C CXX)

option(FIND_FINGERPRINT "AUTO FIND FINGERPRINT BY VART dpu-controller" OFF)

vai_add_library(
  NAME
  pass_level1_dpu
  INCLUDE_DIR
  include
  SRCS
  src/pass_main.cpp
  src/subgraph_processer.hpp
  src/subgraph_processer.cpp
  src/compile_model.hpp
  src/compile_model.cpp
  src/export_to_xir.hpp
  src/export_to_xir.cpp)

if(FIND_FINGERPRINT)
  find_package(vart REQUIRED COMPONENTS dpu-controller)
  target_link_libraries(pass_level1_dpu PRIVATE vart::dpu-controller)
  target_compile_definitions(pass_level1_dpu PRIVATE "-DFIND_FINGERPRINT=1")
else(FIND_FINGERPRINT)
  target_compile_definitions(pass_level1_dpu PRIVATE "-DFIND_FINGERPRINT=0")
endif(FIND_FINGERPRINT)

target_link_libraries(pass_level1_dpu PRIVATE vaip::core vart::util glog::glog vaip::encryption)


vai_add_test(generator_all_xir_op_names REQUIRE xir::xir)
