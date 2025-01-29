#
#  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
#  Licensed under the MIT License.
#


cmake_minimum_required(VERSION 3.17)
project(
  vaip
  VERSION 1.0.0
  LANGUAGES C CXX)

find_package(vart REQUIRED util runner)

vai_add_library(
  NAME
  vaip_custom_op_norm_k
  SRCS
  src/main.cpp
  src/custom_op.hpp
  src/custom_op.cpp
)

if(BUILD_SHARED_LIBS)
  target_compile_definitions(vaip_custom_op_norm_k
                             PUBLIC -DVAIP_CUSTOM_OP_NORM_K_USE_DLL=1)
else(BUILD_SHARED_LIBS)
  target_compile_definitions(vaip_custom_op_norm_k
                             PUBLIC -DVAIP_CUSTOM_OP_NORM_K_USE_DLL=0)
endif(BUILD_SHARED_LIBS)

target_compile_definitions(vaip_custom_op_norm_k PUBLIC "-DVAIP_CUSTOM_OP=1")
target_link_libraries(
  vaip_custom_op_norm_k PRIVATE glog::glog vaip::core vart::runner
                             vart::util)

set_target_properties(vaip_custom_op_norm_k PROPERTIES OUTPUT_NAME
                                                    "vaip_custom_op_norm_k")

# tvm
# if(MSVC)
#   set_source_files_properties(src/custom_op.cpp PROPERTIES COMPILE_FLAGS "/wd4005 /wd4251")
# endif(MSVC)
# target_link_libraries(vaip_custom_op_norm_k PUBLIC tvm_ipu)
add_definitions(-D_CRT_SECURE_NO_WARNINGS)

add_definitions(-w)
## TODO BEGIN fix warnings
target_compile_definitions(vaip_custom_op_norm_k PRIVATE "-D_CRT_SECURE_NO_WARNINGS=1")
## TODO END
