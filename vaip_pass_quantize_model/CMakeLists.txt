#
# The Xilinx Vitis AI Vaip in this distribution are provided under the following
# free and permissive binary-only license, but are not provided in source code
# form.  While the following free and permissive license is similar to the BSD
# open source license, it is NOT the BSD open source license nor other
# OSI-approved open source license.
#
# Copyright (C) 2022 Xilinx, Inc.
# Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
#
# Redistribution and use in binary form only, without modification, is permitted
# provided that the following conditions are met:
#
# 1. Redistributions must reproduce the above copyright notice, this list of
#   conditions and the following disclaimer in the documentation and/or other
#   materials provided with the distribution.
#
# 1. The name of Xilinx, Inc. may not be used to endorse or promote products
#   redistributed with this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY XILINX, INC. "AS IS" AND ANY EXPRESS OR IMPLIED
# WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
# EVENT SHALL XILINX, INC. BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
# BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
# IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE
#

cmake_minimum_required(VERSION 3.17)
project(
  vaip
  VERSION 1.0.0
  LANGUAGES C CXX)

vai_add_library(NAME pass_quantize_model INCLUDE_DIR include SRCS
                src/pass_main.cpp)
vai_add_test(test_python_quantize_model INCLUDE_DIR include)
if(CMAKE_CROSSCOMPILING)
  set_property(
    TARGET pass_quantize_model
    APPEND
    PROPERTY INCLUDE_DIRECTORIES ${VAI_PYTHON_INCLUDE_DIRS})
  target_link_libraries(pass_quantize_model PRIVATE -l${VAI_PYTHON_LIB})
  set_property(
    TARGET test_python_quantize_model
    APPEND
    PROPERTY INCLUDE_DIRECTORIES ${VAI_PYTHON_INCLUDE_DIRS})
  target_link_libraries(test_python_quantize_model PRIVATE -l${VAI_PYTHON_LIB})
else(CMAKE_CROSSCOMPILING)
  target_link_libraries(pass_quantize_model PRIVATE ${PYTHON_LIBRARIES})
  set_property(
    TARGET pass_quantize_model
    APPEND
    PROPERTY INCLUDE_DIRECTORIES "${PYTHON_INCLUDE_DIRS}")
  target_link_libraries(test_python_quantize_model PRIVATE ${PYTHON_LIBRARIES})
  set_property(
    TARGET test_python_quantize_model
    APPEND
    PROPERTY INCLUDE_DIRECTORIES "${PYTHON_INCLUDE_DIRS}")
endif(CMAKE_CROSSCOMPILING)
if(USE_STATIC_PYTHON)
  target_link_libraries(test_python_quantize_model PRIVATE static_python::static_python)
endif()
target_link_libraries(pass_quantize_model PRIVATE pybind11::embed)
target_link_libraries(pass_quantize_model PRIVATE vaip::core vart::util
                                                  glog::glog)
target_link_libraries(test_python_quantize_model PRIVATE pybind11::embed)
# install(TARGETS test_python_quantize_model RUNTIME DESTINATION bin)
