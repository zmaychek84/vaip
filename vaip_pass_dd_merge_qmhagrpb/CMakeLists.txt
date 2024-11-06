#
#   The Xilinx Vitis AI Vaip in this distribution are provided under the following free 
#   and permissive binary-only license, but are not provided in source code form.  While the following free 
#   and permissive license is similar to the BSD open source license, it is NOT the BSD open source license 
#   nor other OSI-approved open source license.
#
# Copyright (C) 2022 Xilinx, Inc.
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
vai_add_library(NAME pass_dd_merge_qmhagrpb INCLUDE_DIR include SRCS src/pass_main.cpp)
target_link_libraries(pass_dd_merge_qmhagrpb PRIVATE vaip::core vart::util glog::glog vaip::vaip_pattern_zoo)


if(TARGET vaip::vaip_pattern_zoo)
  vai_add_library(NAME pass_dd_merge_qmhagrpb_mxgan INCLUDE_DIR include SRCS src/qmhagrpb_processor.hpp src/qmhagrpb_processor.cpp src/pass_main_mxgan.cpp)
  target_link_libraries(pass_dd_merge_qmhagrpb_mxgan PRIVATE vaip::core vart::util glog::glog vaip::vaip_pattern_zoo)
  target_include_directories(pass_dd_merge_qmhagrpb_mxgan PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../vaip_pattern_zoo/include)
  vai_add_debug_command(pass_dd_merge_qmhagrpb_mxgan
    COMMAND "$<TARGET_FILE:voe_py_pass>"
    ARGUMENTS "-i C:\\temp\\$ENV{USERNAME}\\vaip\\.cache\\e5fd1d2783dca2bdc9f40e4941e2a9b9\\onnx.onnx -o c:\\temp\\a.onnx -t c:\\temp\\a.txt -p vaip-pass_dd_merge_qmhagrpb_mxgan -c   C:\\temp\\$ENV{USERNAME}\\vaip\\.cache\\e5fd1d2783dca2bdc9f40e4941e2a9b9"
    ENVIRONMENT DEBUG_DD_MERGE_QMHAGRPB=1
  )

  
  vai_add_library(NAME pass_dd_merge_qmhagrpb_3_mxgan INCLUDE_DIR include SRCS src/qmhagrpb_processor.hpp src/qmhagrpb_processor.cpp src/pass_main_mxgan2.cpp)
  target_link_libraries(pass_dd_merge_qmhagrpb_3_mxgan PRIVATE vaip::core vart::util glog::glog vaip::vaip_pattern_zoo)
  target_include_directories(pass_dd_merge_qmhagrpb_3_mxgan PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../vaip_pattern_zoo/include)
  vai_add_debug_command(pass_dd_merge_qmhagrpb_3_mxgan
    COMMAND "$<TARGET_FILE:voe_py_pass>"
    ARGUMENTS "-i C:\\temp\\$ENV{USERNAME}\\vaip\\.cache\\08678ce4a4eab9eaa2cd8b3dcbdf5697\\onnx.onnx -o c:\\temp\\a.onnx -t c:\\temp\\a.txt -p vaip-pass_dd_merge_qmhagrpb_3_mxgan -c   C:\\temp\\$ENV{USERNAME}\\vaip\\.cache\\08678ce4a4eab9eaa2cd8b3dcbdf5697"
    ENVIRONMENT DEBUG_DD_MERGE_QMHAGRPB=1
  )


  vai_add_library(NAME pass_dd_merge_qmhagrpb_mxpzi INCLUDE_DIR include SRCS src/pass_main_mxpzi.cpp)
  target_link_libraries(pass_dd_merge_qmhagrpb_mxpzi PRIVATE vaip::core pass_dd_merge_qmhagrpb_mxgan vart::util glog::glog vaip::vaip_pattern_zoo)
  target_include_directories(pass_dd_merge_qmhagrpb_mxpzi PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../vaip_pattern_zoo/include)
  vai_add_debug_command(pass_dd_merge_qmhagrpb_mxpzi
    COMMAND "$<TARGET_FILE:voe_py_pass>"
    ARGUMENTS "-i C:\\temp\\$ENV{USERNAME}\\vaip\\.cache\\15d1515c86b40874ac954869798d0e77\\onnx.onnx -o c:\\temp\\a.onnx -t c:\\temp\\a.txt -p vaip-pass_dd_merge_qmhagrpb_mxpzi -c   C:\\temp\\$ENV{USERNAME}\\vaip\\.cache\\15d1515c86b40874ac954869798d0e77"
    ENVIRONMENT DEBUG_DD_MERGE_QMHAGRPB=1
  )
  message(STATUS "ADD pass_dd_merge_qmhagrpb_mxgan and pass_dd_merge_qmhagrpb_mxpzi because vaip::vaip::vaip_pattern_zoo is enabled")
else(TARGET vaip::vaip_pattern_zoo)
  message(WARNING "pass_dd_merge_qmhagrpb_mxgan and pass_dd_merge_qmhagrpb_mxpzi is disabled because vaip::vaip::vaip_pattern_zoo is NOT enabled")
endif(TARGET vaip::vaip_pattern_zoo)
