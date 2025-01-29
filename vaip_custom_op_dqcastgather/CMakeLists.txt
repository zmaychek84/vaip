#
#  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
#  Licensed under the MIT License.
#


cmake_minimum_required(VERSION 3.17)
project(
  vaip
  VERSION 1.0.0
  LANGUAGES C CXX)

include_directories(${ONNXRUNTIME_SRC_DIR}/include/onnxruntime)
find_package(xir)
find_package(vart REQUIRED util runner)

vai_add_library(NAME vaip_custom_op_dqcastgather SRCS src/main.cpp src/custom_op.hpp
                src/custom_op.cpp)

if(BUILD_SHARED_LIBS)
  target_compile_definitions(vaip_custom_op_dqcastgather
                             PUBLIC -DVAIP_CUSTOM_OP_DQCASTGATHER_USE_DLL=1)
else(BUILD_SHARED_LIBS)
  target_compile_definitions(vaip_custom_op_dqcastgather
                             PUBLIC -DVAIP_CUSTOM_OP_DQCASTGATHER_USE_DLL=0)
endif(BUILD_SHARED_LIBS)
target_compile_definitions(vaip_custom_op_dqcastgather PUBLIC "-DVAIP_CUSTOM_OP=1")
target_link_libraries(vaip_custom_op_dqcastgather PRIVATE glog::glog vaip::core xir::xir
                                                 vart::runner vart::util)
set_target_properties(vaip_custom_op_dqcastgather PROPERTIES OUTPUT_NAME
                                                    "vaip_custom_op_DQCASTGATHER")