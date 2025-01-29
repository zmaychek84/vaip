#
#  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
#  Licensed under the MIT License.
#

cmake_minimum_required(VERSION 3.17)
project(
  vaip
  VERSION 1.0.0
  LANGUAGES C CXX)

vai_add_library(NAME pass_graph_label_example INCLUDE_DIR include SRCS
                pass_graph_label_example.cpp)

if(CMAKE_CROSSCOMPILING)
  set_property(
    TARGET pass_graph_label_example
    APPEND
    PROPERTY INCLUDE_DIRECTORIES ${VAI_PYTHON_INCLUDE_DIRS})
  target_link_libraries(pass_graph_label_example PRIVATE -l${VAI_PYTHON_LIB})
else(CMAKE_CROSSCOMPILING)
  target_link_libraries(pass_graph_label_example PRIVATE ${PYTHON_LIBRARIES})
  set_property(
    TARGET pass_graph_label_example
    APPEND
    PROPERTY INCLUDE_DIRECTORIES "${PYTHON_INCLUDE_DIRS}")
endif(CMAKE_CROSSCOMPILING)
target_link_libraries(pass_graph_label_example PRIVATE pybind11::embed)
target_link_libraries(pass_graph_label_example PRIVATE vaip::core vart::util
                                                       glog::glog)
