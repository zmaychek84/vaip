#
#  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
#  Licensed under the MIT License.
#


cmake_minimum_required(VERSION 3.17)
project(
  vaip
  VERSION 1.0.0
  LANGUAGES C CXX)

vai_add_library(NAME pass_dd_merge_iconv_0 INCLUDE_DIR include SRCS
                src/pattern_0.cpp)

target_link_libraries(pass_dd_merge_iconv_0 PRIVATE vaip::core vart::util
                                                        glog::glog vaip::vaip_pattern_zoo)

vai_add_library(NAME pass_dd_merge_iconv_1 INCLUDE_DIR include SRCS
                src/pattern_1.cpp)

target_link_libraries(pass_dd_merge_iconv_1 PRIVATE vaip::core vart::util
                                                        glog::glog vaip::vaip_pattern_zoo)

vai_add_library(NAME pass_dd_merge_iconv_2 INCLUDE_DIR include SRCS
                src/pattern_2.cpp)

target_link_libraries(pass_dd_merge_iconv_2 PRIVATE vaip::core vart::util
                                                        glog::glog vaip::vaip_pattern_zoo)

vai_add_library(NAME pass_dd_merge_iconv_3 INCLUDE_DIR include SRCS
                src/pattern_3.cpp)

target_link_libraries(pass_dd_merge_iconv_3 PRIVATE vaip::core vart::util
                                                        glog::glog vaip::vaip_pattern_zoo)