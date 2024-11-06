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

include_directories(${ONNXRUNTIME_SRC_DIR}/include/onnxruntime)
find_package(xir)
find_package(vart REQUIRED util runner)

vai_add_library(NAME vaip_custom_op_gather SRCS src/main.cpp src/custom_op.hpp
                src/custom_op.cpp)

if(BUILD_SHARED_LIBS)
  target_compile_definitions(vaip_custom_op_gather
                             PUBLIC -DVAIP_CUSTOM_OP_GATHER_USE_DLL=1)
else(BUILD_SHARED_LIBS)
  target_compile_definitions(vaip_custom_op_gather
                             PUBLIC -DVAIP_CUSTOM_OP_GATHER_USE_DLL=0)
endif(BUILD_SHARED_LIBS)
target_compile_definitions(vaip_custom_op_gather PUBLIC "-DVAIP_CUSTOM_OP=1")
target_link_libraries(vaip_custom_op_gather PRIVATE glog::glog vaip::core xir::xir
                                                 vart::runner vart::util)
set_target_properties(vaip_custom_op_gather PROPERTIES OUTPUT_NAME
                                                    "vaip_custom_op_GATHER")