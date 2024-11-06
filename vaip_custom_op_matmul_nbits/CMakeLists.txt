#
#   The Xilinx Vitis AI Vaip in this distribution are provided under the following free 
#   and permissive binary-only license, but are not provided in source code form.  While the following free 
#   and permissive license is similar to the BSD open source license, it is NOT the BSD open source license 
#   nor other OSI-approved open source license.
#
# Copyright (C) 2022 Xilinx, Inc.
# Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc.
#
#    Redistribution and use in binary form only, without modification, is permitted provided that the following conditions are met:
#
#    1. Redistributions must reproduce the above copyright notice, this list of conditions and the following disclaimer in 
#    the documentation and/or other materials provided with the distribution.
#
#    2. The name of Xilinx, Inc. may not be used to endorse or promote products redistributed with this software without specific 
#    prior written permission.
#
#    THIS SOFTWARE IS PROVIDED BY XILINX, INC. "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, 
#    THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL XILINX, INC. 
#    BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
#    PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY 
#    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT 
#    OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE
#
cmake_minimum_required(VERSION 3.17)
project(
  vaip
  VERSION 1.0.0
  LANGUAGES C CXX)
set (XRT_DIR $ENV{XRT_PATH})

find_package(XRT REQUIRED PATHS ${XRT_DIR})
include_directories("${CMAKE_INSTALL_PREFIX}/include/transformers/include")
include_directories("${CMAKE_INSTALL_PREFIX}/include")
include_directories("${CMAKE_INSTALL_PREFIX}/include/transformers/include/utils")
include_directories("${CMAKE_INSTALL_PREFIX}/include/aie_rt/xaiengine")
include_directories("${CMAKE_INSTALL_PREFIX}/include/aie_rt")
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


