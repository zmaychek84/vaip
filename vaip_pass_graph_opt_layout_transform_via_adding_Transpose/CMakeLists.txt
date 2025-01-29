#
#  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
#  Licensed under the MIT License.
#


cmake_minimum_required(VERSION 3.17)
project(
  vaip
  VERSION 1.0.0
  LANGUAGES C CXX)

vai_add_library(NAME pass_graph_opt_layout_transform_via_adding_Transpose
                INCLUDE_DIR include SRCS src/pass_main.cpp)

target_link_libraries(pass_graph_opt_layout_transform_via_adding_Transpose
                      PRIVATE vaip::core vart::util glog::glog)
