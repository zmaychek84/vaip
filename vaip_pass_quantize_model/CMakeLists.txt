#
#  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
#  Licensed under the MIT License.
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
