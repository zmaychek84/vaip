#
#  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
#  Licensed under the MIT License.
#
cmake_minimum_required(VERSION 3.17)
project(
  vaip
  VERSION 1.0.0
  LANGUAGES C CXX)
set (XRT_DIR $ENV{XRT_PATH})

find_package(XRT REQUIRED PATHS ${XRT_DIR})

include_directories("${CMAKE_INSTALL_PREFIX}/include/transformers/include")
include_directories("${CMAKE_INSTALL_PREFIX}/include/transformers/include/utils")
include_directories("${CMAKE_INSTALL_PREFIX}/include/aie_rt/xaiengine")
include_directories("${CMAKE_INSTALL_PREFIX}/include/aie_rt")


include_directories(${ONNXRUNTIME_SRC_DIR}/include/onnxruntime
                    ${XRT_INCLUDE_DIRS})
vai_add_library(NAME vaip_custom_op_gmatmul_integer SRCS src/main.cpp src/custom_op.hpp
                src/custom_op.cpp)

if(BUILD_SHARED_LIBS)
  target_compile_definitions(vaip_custom_op_gmatmul_integer
                             PUBLIC -DVAIP_CUSTOM_OP_GMATMULINTEGER_USE_DLL=1)
else(BUILD_SHARED_LIBS)
  target_compile_definitions(vaip_custom_op_gmatmul_integer
                             PUBLIC -DVAIP_CUSTOM_OP_GMATMULINTEGER_USE_DLL=0)
endif(BUILD_SHARED_LIBS)
target_compile_definitions(vaip_custom_op_gmatmul_integer PUBLIC "-DVAIP_CUSTOM_OP=1")

find_package(Eigen3 REQUIRED)
find_package(spdlog REQUIRED)

target_link_libraries(vaip_custom_op_gmatmul_integer PRIVATE spdlog::spdlog glog::glog vaip::core XRT::xrt_coreutil Eigen3::Eigen)
set_target_properties(vaip_custom_op_gmatmul_integer PROPERTIES OUTPUT_NAME
                                                    "vaip_custom_op_GMATMULINTEGER")
add_definitions(-D_SILENCE_CXX17_RESULT_OF_DEPRECATION_WARNING)
add_definitions(-DXAIE_FEATURE_MSVC)

add_definitions(-w)
