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
include_directories("${CMAKE_INSTALL_PREFIX}/include")
include_directories("${CMAKE_INSTALL_PREFIX}/include/transformers/include/utils")
include_directories("${CMAKE_INSTALL_PREFIX}/include/aie_rt/xaiengine")
include_directories("${CMAKE_INSTALL_PREFIX}/include/aie_rt")
# include_directories("${CMAKE_INSTALL_PREFIX}/include/transformers/include")
# include_directories("${CMAKE_INSTALL_PREFIX}/include/transformers/include/utils")
include_directories(${CMAKE_CURRENT_LIST_DIR}/../vaip_summary_report)
if(NOT WIN32)
set(LINK_DIRS
    ${CMAKE_INSTALL_PREFIX}/lib/
)
endif()
include_directories(${ONNXRUNTIME_SRC_DIR}/include/onnxruntime
                    ${XRT_INCLUDE_DIRS})
vai_add_library(NAME vaip_custom_op_matmul_nbits SRCS src/main.cpp src/custom_op.hpp
                src/custom_op.cpp src/custom_op_mladf.cpp src/util.cpp)

if(BUILD_SHARED_LIBS)
  target_compile_definitions(vaip_custom_op_matmul_nbits
                             PUBLIC -DVAIP_CUSTOM_OP_MATMULNBITS_USE_DLL=1)
else(BUILD_SHARED_LIBS)
  target_compile_definitions(vaip_custom_op_matmul_nbits
                             PUBLIC -DVAIP_CUSTOM_OP_MATMULNBITS_USE_DLL=0)
endif(BUILD_SHARED_LIBS)
target_compile_definitions(vaip_custom_op_matmul_nbits PUBLIC "-DVAIP_CUSTOM_OP=1")
if(NOT WIN32)
# Add the -mavx512fp16 flag to the specific target
target_compile_options(vaip_custom_op_matmul_nbits PRIVATE -mavx512bf16 -mavx512f)
endif()
find_package(Eigen3 REQUIRED)
find_package(spdlog REQUIRED)
if(WIN32)
  target_link_libraries(vaip_custom_op_matmul_nbits PRIVATE  dyn_dispatch_core spdlog::spdlog glog::glog vaip::core XRT::xrt_coreutil Eigen3::Eigen)
else()
target_link_libraries(vaip_custom_op_matmul_nbits PRIVATE  ${CMAKE_INSTALL_PREFIX}/lib/libdyn_dispatch_core.so spdlog::spdlog glog::glog vaip::core XRT::xrt_coreutil Eigen3::Eigen)
endif(WIN32)
set_target_properties(vaip_custom_op_matmul_nbits PROPERTIES OUTPUT_NAME
                                                    "vaip_custom_op_MATMULNBITS")
add_definitions(-D_SILENCE_CXX17_RESULT_OF_DEPRECATION_WARNING)
add_definitions(-DXAIE_FEATURE_MSVC)


add_definitions(-w)
