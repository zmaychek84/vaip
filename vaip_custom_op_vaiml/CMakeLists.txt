#
#   The Xilinx Vitis AI Vaip in this distribution are provided under the following free
#   and permissive binary-only license, but are not provided in source code form.  While the following free
#   and permissive license is similar to the BSD open source license, it is NOT the BSD open source license
#   nor other OSI-approved open source license.
#
#    Copyright (C) 2022 Xilinx, Inc. All rights reserved.
#    Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
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

find_package(nlohmann_json REQUIRED)
find_package(XRT REQUIRED PATHS ${XRT_DIR})

include_directories("${XRT_DIR}/../../../include")
include_directories("${xaiengine_BINARY_DIR}/include/xaiengine")
include_directories("${xaiengine_BINARY_DIR}/include")

set(LINK_LIBS
    XRT::xrt_coreutil
    nlohmann_json::nlohmann_json
)


find_package(nlohmann_json REQUIRED)

include_directories("${XRT_DIR}/../../../include")

vai_add_library(
  NAME
  vaip_custom_op_vaiml
  SRCS
  $(XRT_DIR)/../../../
  src/main.cpp
  src/HT_1_2/custom_op_ht_1_2.hpp
  src/HT_1_2/custom_op_ht_1_2.cpp
  src/HT_1_2/load_wts.h
  src/HT_1_2/load_wts.cpp
  src/HT_1_2/txn_pkg_ht_1_2.hpp
  src/HT_1_2/txn_pkg_ht_1_2.cpp
  src/common/vaiml_client.h
  src/common/hw_runner.h
  src/common/hw_runner.cpp
  src/common/op_base.h
  src/common/op_buf.hpp
  src/common/op_defs.h
  src/common/op_init.hpp
  src/common/op_types.h
  src/common/timer.h
  src/common/bf16_utils.h
  src/common/bf16_utils.cpp
  src/GT/common/gen_gt_wts_common.h
  src/GT/GT_1_2/gen_gt_wts.h
  src/GT/GT_1_2/txn_pkg_gt_1_2.hpp
  src/GT/GT_1_2/txn_pkg_gt_1_2.cpp
  src/GT/GT_1_2/constant_fold_result.h
  src/GT/GT_1_2/custom_op_gt_1_2.hpp
  src/GT/GT_1_2/custom_op_gt_1_2.cpp
  src/GT/GT_1_3/custom_op_gt_1_3.hpp
  src/GT/GT_1_3/custom_op_gt_1_3.cpp
  src/GT/GT_1_3/txn_pkg_gt_1_3.cpp
  src/GT/GT_1_3/txn_pkg_gt_1_3.hpp
  )
if(BUILD_SHARED_LIBS)
  target_compile_definitions(vaip_custom_op_vaiml
                             PUBLIC -DVAIP_CUSTOM_OP_VAIML_USE_DLL=1)
else(BUILD_SHARED_LIBS)
  target_compile_definitions(vaip_custom_op_vaiml
                             PUBLIC -DVAIP_CUSTOM_OP_VAIML_USE_DLL=0)
endif(BUILD_SHARED_LIBS)
target_compile_definitions(vaip_custom_op_vaiml PUBLIC "-DVAIP_CUSTOM_OP=1")

target_include_directories(vaip_custom_op_vaiml PRIVATE ${zlib_SOURCE_DIR} ${zlib_BINARY_DIR})
target_link_libraries(
  vaip_custom_op_vaiml PRIVATE nlohmann_json::nlohmann_json glog::glog vaip::core ZLIB::ZLIB)

#if(ENABLE_XRT_SHARED_CONTEXT)
#    target_compile_definitions(vaip_custom_op_vaiml PRIVATE
#        "-DENABLE_XRT_SHARED_CONTEXT=1")
#    find_package(XRT REQUIRED)
#    target_link_libraries(vaip_custom_op_vaiml PRIVATE XRT::xrt_coreutil)
#    target_include_directories(vaip_custom_op_vaiml PRIVATE ${XRT_INCLUDE_DIRS})
#endif()

target_link_libraries(vaip_custom_op_vaiml PUBLIC ${LINK_LIBS})

set_target_properties(vaip_custom_op_vaiml PROPERTIES OUTPUT_NAME
                                                    "vaip_custom_op_VAIML")

add_definitions(-D_SILENCE_CXX17_RESULT_OF_DEPRECATION_WARNING)
add_definitions(-DXAIE_FEATURE_MSVC)
add_definitions(-w)
