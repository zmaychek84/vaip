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
  ort_vitis_ai_ep
  VERSION 1.0.0
  LANGUAGES C CXX)
option(FIND_FINGERPRINT "AUTO FIND FINGERPRINT BY VART dpu-controller" OFF)

find_package(xir REQUIRED)
find_package(vart REQUIRED util runner)

find_package(googletest)

vai_add_test(test_p REQUIRE ort_vitis_ai_ep)
# add_executable(test_p test/test_p.cpp) target_link_libraries(test_p onnx
# protobuf::libprotobuf)

# vai_add_test(onnx2xir REQUIRE ort_vitis_ai_ep xir::xir)
# target_link_libraries(onnx2xir ort_vitis_ai_ep onnx xir::xir)

if(googletest_FOUND)
  # vai_add_test( pattern_test REQUIRE ort_vitis_ai_ep onnx xir::xir
  # googletest::gtest onnxruntime::onnxruntime)

  # vai_add_test( pattern_cached_test REQUIRE ort_vitis_ai_ep onnx xir::xir
  # googletest::gtest onnxruntime::onnxruntime)

  # vai_add_test( xir_ops_test REQUIRE ort_vitis_ai_ep onnx xir::xir
  # googletest::gtest onnxruntime::onnxruntime)

  # vai_add_test( learn_partition REQUIRE ort_vitis_ai_ep onnx xir::xir
  # googletest::gtest onnxruntime::onnxruntime)

  # vai_add_test( rewrite_rule_test REQUIRE ort_vitis_ai_ep onnx xir::xir
  # googletest::gtest onnxruntime::onnxruntime)

  # vai_add_test( denotation_test REQUIRE ort_vitis_ai_ep onnx xir::xir
  # googletest::gtest onnxruntime::onnxruntime)

  # vai_add_test( replace_fix_test REQUIRE ort_vitis_ai_ep onnx xir::xir
  # googletest::gtest onnxruntime::onnxruntime)

  # vai_add_test( execution_provider_test REQUIRE ort_vitis_ai_ep onnx xir::xir
  # googletest::gtest onnxruntime::onnxruntime)

  # find_package(OpenCV COMPONENTS opencv_core opencv_video opencv_videoio
  # opencv_imgproc opencv_imgcodecs opencv_highgui)

  # vai_add_test( resnet50_pt REQUIRE ort_vitis_ai_ep onnx xir::xir googletest::gtest
  # onnxruntime::onnxruntime xcompiler::xcompiler-core ${OpenCV_LIBS})

  # vai_add_test( test_onnx_runner REQUIRE ort_vitis_ai_ep onnx xir::xir
  # googletest::gtest onnxruntime::onnxruntime xcompiler::xcompiler-core)

  vai_add_test(xmodel_topo REQUIRE xir::xir)
  vai_add_test(dump_const REQUIRE xir::xir)

  vai_add_test(test_dpu_runner REQUIRE xir::xir vart::runner)

endif(googletest_FOUND)
