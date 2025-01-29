#
#  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
#  Licensed under the MIT License.
#
cmake_minimum_required(VERSION 3.5)
project(
  vaip
  VERSION 1.0.0
  LANGUAGES C CXX)
if (MSVC)
  add_compile_options(/bigobj)
else ()
  add_compile_options(-Wa,-mbig-obj)
endif ()
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../vaip/src")
vai_add_library(NAME pass_vaiml_partition INCLUDE_DIR
                include
                SRCS
                src/vaiml_partition_load_wts.h
                src/gt_inner_map_1_2.h
                src/ht_inner_map_1_2.h
                src/gt_inner_map_1_3.h
                src/pass_main.cpp
                src/vaiml_subgraph_processor.h
                src/vaiml_subgraph_processor.cc)

find_package(nlohmann_json REQUIRED)
target_link_libraries(pass_vaiml_partition
                      PRIVATE vaip::core nlohmann_json::nlohmann_json xir::xir vart::util glog::glog)
