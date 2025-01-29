#
#  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
#  Licensed under the MIT License.
#


cmake_minimum_required(VERSION 3.17)
project(
  vaip
  VERSION 1.0.0
  LANGUAGES C CXX)

vai_add_library(NAME pass_dd_merge_dqadd INCLUDE_DIR include SRCS
                src/pass_main.cpp)

target_link_libraries(pass_dd_merge_dqadd PRIVATE vaip::core vart::util
                                                        glog::glog vaip::vaip_pattern_zoo)




vai_add_library(
  NAME
  pass_dd_merge_actconstadd
  INCLUDE_DIR
  include
  SRCS
  src/pass_main_actconstadd.cpp)

target_link_libraries(pass_dd_merge_actconstadd PRIVATE vaip::core vart::util glog::glog vaip::vaip_pattern_zoo)
vai_add_debug_command(pass_dd_merge_actconstadd
    COMMAND "$<TARGET_FILE:voe_py_pass>"
    ARGUMENTS "-i C:\\temp\\$ENV{USERNAME}\\vaip\\.cache\\CUR\\onnx.onnx -o c:\\temp\\a.onnx -t c:\\temp\\a.txt -p vaip-pass_dd_merge_dqadd -c  C:\\temp\\$ENV{USERNAME}\\vaip\\.cache\\CUR"
    ENVIRONMENT DEBUG_DD_MERGE_DQADD=1
    )
target_compile_definitions(pass_dd_merge_actconstadd PUBLIC "-DENABLE_VAIP_PASS_DD_MERGE_DQADD=1")
