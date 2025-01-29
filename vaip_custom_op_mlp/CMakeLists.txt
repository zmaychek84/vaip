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
find_package(XRT REQUIRED)

vai_add_library(NAME vaip_custom_op_mlp SRCS src/main.cpp src/custom_op.hpp
                src/custom_op.cpp)

include_directories(${ONNXRUNTIME_SRC_DIR}/include/onnxruntime ${XRT_INCLUDE_DIRS})
include_directories(${CMAKE_INSTALL_PREFIX}/include/)
include_directories(${CMAKE_CURRENT_LIST_DIR}/../vaip_summary_report)
if(NOT WIN32)
target_compile_options(vaip_custom_op_mlp PRIVATE -mavx512bf16 -mavx512f)
endif()
if(BUILD_SHARED_LIBS)
  target_compile_definitions(vaip_custom_op_mlp
                             PUBLIC -DVAIP_CUSTOM_OP_MLP_USE_DLL=1)
else(BUILD_SHARED_LIBS)
  target_compile_definitions(vaip_custom_op_mlp
                             PUBLIC -DVAIP_CUSTOM_OP_MLP_USE_DLL=0)
endif(BUILD_SHARED_LIBS)
target_compile_definitions(vaip_custom_op_mlp PUBLIC "-DVAIP_CUSTOM_OP=1")
if(WIN32)
  target_link_libraries(vaip_custom_op_mlp PRIVATE glog::glog vaip::core xir::xir XRT::xrt_coreutil
                                                 vart::runner vart::util dyn_dispatch_core)
else()
  target_link_libraries(vaip_custom_op_mlp PRIVATE glog::glog vaip::core xir::xir XRT::xrt_coreutil
  vart::runner vart::util ${CMAKE_INSTALL_PREFIX}/lib/libdyn_dispatch_core.so)
endif(WIN32)
set_target_properties(vaip_custom_op_mlp PROPERTIES OUTPUT_NAME
                                                    "vaip_custom_op_MLP")
## TODO BEGIN fix warnings
target_compile_definitions(vaip_custom_op_mlp PRIVATE "-D_CRT_SECURE_NO_WARNINGS=1")
## TODO END
