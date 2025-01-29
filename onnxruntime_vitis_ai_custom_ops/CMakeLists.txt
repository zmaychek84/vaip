#
#  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
#  Licensed under the MIT License.
#


cmake_minimum_required(VERSION 3.17)
project(
  vaip
  VERSION 1.0.0
  LANGUAGES C CXX)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)

set (XRT_DIR $ENV{XRT_PATH})
find_package(xir)
find_package(Eigen3 REQUIRED)
find_package(vart REQUIRED util runner)
find_package(XRT REQUIRED PATHS ${XRT_DIR})

#if(WIN32)
#find_package(DynamicDispatch REQUIRED)
#endif()

include_directories(${ONNXRUNTIME_SRC_DIR}/include/onnxruntime ${XRT_INCLUDE_DIRS})
include_directories(${CMAKE_INSTALL_PREFIX}/include/)

include_directories(${transformers_SOURCE_DIR}/include/)

include_directories(${CMAKE_CURRENT_LIST_DIR}/../vaip_summary_report/)
if(NOT WIN32)
include_directories(${CMAKE_INSTALL_PREFIX}/include/ryzenai/dynamic_dispatch)
set(LINK_DIRS
    ${CMAKE_INSTALL_PREFIX}/lib/
)
endif()
vai_add_library(NAME onnxruntime_vitis_ai_custom_ops
  SRCS SHARED
  src/main.cpp
  src/custom_op_mha.hpp
  src/custom_op_mha.cpp
  src/custom_op_prefill_gqa.hpp
  src/custom_op_prefill_gqa.cpp
  src/custom_op_gqa.hpp
  src/custom_op_gqa.cpp
  src/custom_op_gqo.hpp
  src/custom_op_gqo.cpp
  src/custom_op_sslrn.hpp
  src/custom_op_sslrn.cpp
  src/custom_op_ssmlp.hpp
  src/custom_op_ssmlp.cpp
  src/custom_op_rope.hpp
  src/custom_op_rope.cpp
  src/custom_op_slrn.hpp
  src/custom_op_slrn.cpp)

target_compile_definitions(onnxruntime_vitis_ai_custom_ops
                             PUBLIC -DVAIP_CUSTOM_OP_MHA_USE_DLL=1)
if(NOT WIN32)
  target_compile_options(onnxruntime_vitis_ai_custom_ops PRIVATE -mavx512bf16 -mavx512f -mfma -mavx512vl )
endif()

target_compile_definitions(onnxruntime_vitis_ai_custom_ops PUBLIC "-DVAIP_CUSTOM_OP=1")

if(WIN32)
  target_link_libraries(onnxruntime_vitis_ai_custom_ops PRIVATE glog::glog vaip::core xir::xir XRT::xrt_coreutil
                                                 vart::runner vart::util
                                                 dyn_dispatch_core
                                                 Eigen3::Eigen)
else()
  target_link_libraries(onnxruntime_vitis_ai_custom_ops PRIVATE glog::glog vaip::core xir::xir XRT::xrt_coreutil vart::runner vart::util ${CMAKE_INSTALL_PREFIX}/lib/libdyn_dispatch_core.so Eigen3::Eigen)
endif()

set_target_properties(onnxruntime_vitis_ai_custom_ops PROPERTIES OUTPUT_NAME "onnxruntime_vitis_ai_custom_ops")


#if(WIN32)
set(ONNXRUNTIME_CUSTOM_OP_LIB_LINK_FLAG "-DEF:${PROJECT_SOURCE_DIR}/src/onnxruntime_vitis_ai_custom_ops.def")
set_property(TARGET onnxruntime_vitis_ai_custom_ops APPEND_STRING PROPERTY LINK_FLAGS ${ONNXRUNTIME_CUSTOM_OP_LIB_LINK_FLAG})
#endif()

## TODO BEGIN fix warnings
set_property(TARGET onnxruntime_vitis_ai_custom_ops PROPERTY COMPILE_WARNING_AS_ERROR OFF)
message(WARNING "ignore compilation errors for onnxruntime_vitis_ai_custom_ops")
target_compile_definitions(onnxruntime_vitis_ai_custom_ops PRIVATE "-D_CRT_SECURE_NO_WARNINGS=1")
## TODO END
