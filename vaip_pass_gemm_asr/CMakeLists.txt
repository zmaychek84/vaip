#
#  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
#  Licensed under the MIT License.
#


cmake_minimum_required(VERSION 3.17)
project(
  vaip
  VERSION 1.0.0
  LANGUAGES C CXX)

vai_add_library(NAME pass_gemm_asr INCLUDE_DIR include SRCS
                src/pass_main.cpp)

target_link_libraries(pass_gemm_asr PRIVATE vaip::core vart::util
                                                 glog::glog)

# tvm
if(MSVC)
  set_source_files_properties(src/pass_main.cpp PROPERTIES COMPILE_FLAGS "/wd4005 /wd4251")
endif(MSVC)

target_link_libraries(pass_gemm_asr PUBLIC tvm_ipu)
add_definitions(-D_CRT_SECURE_NO_WARNINGS)
