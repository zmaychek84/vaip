#
#  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
#  Licensed under the MIT License.
#


cmake_minimum_required(VERSION 3.17)
project(
  vaip
  VERSION 1.0.0
  LANGUAGES C CXX)

vai_add_library(
  NAME
  pass_graph_output_add_node
  INCLUDE_DIR
  include
  SRCS
  src/pass_main.cpp
  src/graph_output_add_node.cpp
  src/graph_output_add_node.hpp)

target_link_libraries(pass_graph_output_add_node PRIVATE vaip::core vart::util
                                                         glog::glog)
target_compile_definitions(pass_graph_output_add_node
                           PRIVATE "-DVAIP_USE_DEPRECATED_API=1")
