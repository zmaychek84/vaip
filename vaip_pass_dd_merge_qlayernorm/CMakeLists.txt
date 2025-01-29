#
#  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
#  Licensed under the MIT License.
#


cmake_minimum_required(VERSION 3.17)
project(
  vaip
  VERSION 1.0.0
  LANGUAGES C CXX)

vai_add_library(NAME pass_dd_merge_qlayernorm INCLUDE_DIR include SRCS
                src/pass_main.cpp)

target_link_libraries(pass_dd_merge_qlayernorm PRIVATE vaip::core vart::util
                                                        glog::glog vaip::vaip_pattern_zoo)

                                                        
vai_add_library(NAME pass_dd_merge_qlayernorm_1 INCLUDE_DIR include SRCS
                src/pass_main_1.cpp)

target_link_libraries(pass_dd_merge_qlayernorm_1 PRIVATE vaip::core vart::util
                                                        glog::glog vaip::vaip_pattern_zoo)