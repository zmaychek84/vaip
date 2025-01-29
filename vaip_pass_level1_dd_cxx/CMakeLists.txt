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

set (XRT_DIR $ENV{XRT_PATH})
find_package(XRT REQUIRED)
find_package(spdlog REQUIRED)
find_package(dod REQUIRED)
set(INCL_DIRS
    ${CMAKE_INSTALL_PREFIX}/include/aie-rt/include
    ${CMAKE_INSTALL_PREFIX}/include
    ${CMAKE_INSTALL_PREFIX}/src
    ${XRT_INCLUDE_DIRS}
)

set(LINK_DIRS
    ${CMAKE_INSTALL_PREFIX}/lib/
)

set(LINK_LIBS
    XRT::xrt_coreutil
    spdlog::spdlog
    dyn_dispatch_core
    transaction
    xclbin
)
vai_add_library(
  NAME
  pass_level1_dd_cxx
  INCLUDE_DIR
  include
  SRCS
  src/pass_main.cpp
  src/fuse.cpp
  src/fuse.hpp
  src/partitioner.cpp
  src/onnx_graph_partitioner.cpp
  src/prepare_metadata.cpp
  # src/test_partitioner.cpp
)
target_include_directories(pass_level1_dd_cxx PRIVATE ${dod_SOURCE_DIR}/src)
if(MSVC)
  target_compile_options(pass_level1_dd_cxx PRIVATE "/wd4996")
endif(MSVC)
if(FIND_FINGERPRINT)
  find_package(vart REQUIRED COMPONENTS dpu-controller)
  target_link_libraries(pass_level1_dd_cxx PRIVATE vart::dpu-controller)
  target_compile_definitions(pass_level1_dd_cxx PRIVATE "-DFIND_FINGERPRINT=1")
else(FIND_FINGERPRINT)
  target_compile_definitions(pass_level1_dd_cxx PRIVATE "-DFIND_FINGERPRINT=0")
endif(FIND_FINGERPRINT)

target_include_directories(pass_level1_dd_cxx PRIVATE ${INCL_DIRS})
target_link_libraries(pass_level1_dd_cxx PRIVATE vaip::core vart::util glog::glog vaip::encryption ${LINK_LIBS})

# vai_add_test(generator_all_xir_op_names REQUIRE xir::xir)
