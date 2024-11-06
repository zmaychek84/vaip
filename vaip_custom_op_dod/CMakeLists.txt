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
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_CXX_EXTENSION OFF)

if (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Zc:__cplusplus")
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
endif()

find_package(XRT REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(spdlog REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(vart REQUIRED runner)
find_package(dod REQUIRED)

# add include paths
set(INCL_DIRS
    ${CMAKE_INSTALL_PREFIX}/include/aie-rt/include
    ${CMAKE_INSTALL_PREFIX}/include
    ${CMAKE_INSTALL_PREFIX}/src
    ${XRT_INCLUDE_DIRS}
    ${ONNXRUNTIME_SRC_DIR}/include/onnxruntime
    ${ONNXRUNTIME_INCLUDE_DIR}
)

set(LINK_DIRS
    ${CMAKE_INSTALL_PREFIX}/lib/
)

set(LINK_LIBS
    XRT::xrt_coreutil
    nlohmann_json::nlohmann_json
    spdlog::spdlog
    xaiengine
    dyn_dispatch_core
    vaip::core
    Eigen3::Eigen
    transaction
    vart::runner
)

vai_add_library(NAME vaip_custom_op_dod SRCS src/main.cpp src/custom_op.hpp
                src/custom_op.cpp)

target_include_directories(vaip_custom_op_dod PRIVATE ${dod_SOURCE_DIR}/src)
if(MSVC)
  target_compile_options(vaip_custom_op_dod PRIVATE "/wd4996")
  target_compile_options(vaip_custom_op_dod PRIVATE "/wd2593")
endif(MSVC)
if(BUILD_SHARED_LIBS)
  target_compile_definitions(vaip_custom_op_dod
                             PUBLIC -DVAIP_CUSTOM_OP_DOD_USE_DLL=1)
else(BUILD_SHARED_LIBS)
  target_compile_definitions(vaip_custom_op_dod
                             PUBLIC -DVAIP_CUSTOM_OP_DOD_USE_DLL=0)
endif(BUILD_SHARED_LIBS)
target_compile_definitions(vaip_custom_op_dod PUBLIC "-DVAIP_CUSTOM_OP=1")

set_target_properties(vaip_custom_op_dod PROPERTIES OUTPUT_NAME
                                                    "vaip_custom_op_DOD")


target_include_directories(vaip_custom_op_dod PRIVATE ${INCL_DIRS})
target_link_directories(vaip_custom_op_dod PUBLIC ${LINK_DIRS})
target_link_libraries(vaip_custom_op_dod PUBLIC ${LINK_LIBS} )
target_link_libraries(vaip_custom_op_dod PRIVATE ZLIB::ZLIB)

add_definitions(-D_SILENCE_CXX17_RESULT_OF_DEPRECATION_WARNING)
add_definitions(-DXAIE_FEATURE_MSVC)
add_definitions(-w)

