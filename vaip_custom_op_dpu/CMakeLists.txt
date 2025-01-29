#
#  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
#  Licensed under the MIT License.
#


cmake_minimum_required(VERSION 3.17)
project(
  vaip
  VERSION 1.0.0
  LANGUAGES C CXX)

find_package(vart REQUIRED util trace runner)
find_package(Eigen3 REQUIRED)
find_package(nlohmann_json REQUIRED)

vai_add_library(
  NAME
  vaip_custom_op_dpu
  SRCS
  src/graph_holder.cpp
  src/graph_holder.hpp
  src/main.cpp
  src/custom_op.hpp
  src/custom_op.cpp
  src/ort_tensor_buffer.hpp
  src/ort_tensor_buffer.cpp
  src/schedule.hpp
  src/schedule.cpp
  src/dlanalyzer.cpp)

if(BUILD_SHARED_LIBS)
  target_compile_definitions(vaip_custom_op_dpu
                             PUBLIC -DVAIP_CUSTOM_OP_DPU_USE_DLL=1)
else(BUILD_SHARED_LIBS)
  target_compile_definitions(vaip_custom_op_dpu
                             PUBLIC -DVAIP_CUSTOM_OP_DPU_USE_DLL=0)
endif(BUILD_SHARED_LIBS)
target_compile_definitions(vaip_custom_op_dpu PUBLIC "-DVAIP_CUSTOM_OP=1")

target_link_libraries(
  vaip_custom_op_dpu PRIVATE glog::glog vaip::core xir::xir vart::runner
                             vart::util vart::trace vaip::encryption
                             nlohmann_json::nlohmann_json)

if(ENABLE_XRT_SHARED_CONTEXT)
    target_compile_definitions(vaip_custom_op_dpu PRIVATE
        "-DENABLE_XRT_SHARED_CONTEXT=1")
    find_package(XRT REQUIRED)
    target_link_libraries(vaip_custom_op_dpu PRIVATE XRT::xrt_coreutil)    
    target_include_directories(vaip_custom_op_dpu PRIVATE ${XRT_INCLUDE_DIRS})
endif()

set_target_properties(vaip_custom_op_dpu PROPERTIES OUTPUT_NAME
                                                    "vaip_custom_op_DPU")
