#
#  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
#  Licensed under the MIT License.
#
vai_add_library(
  NAME
  pass_dd_merge_attentionprepro
  INCLUDE_DIR
  include
  SRCS
  src/pass_main_mdsqr.cpp)

target_link_libraries(pass_dd_merge_attentionprepro PRIVATE vaip::core vart::util glog::glog vaip::vaip_pattern_zoo)
vai_add_debug_command(pass_dd_merge_attentionprepro
    COMMAND "$<TARGET_FILE:voe_py_pass>"
    ARGUMENTS "-i C:\\temp\\$ENV{USERNAME}\\vaip\\.cache\\CUR\\onnx.onnx -o c:\\temp\\a.onnx -t c:\\temp\\a.txt -p vaip-pass_dd_merge_attentionprepro -c  C:\\temp\\$ENV{USERNAME}\\vaip\\.cache\\CUR"
    ENVIRONMENT DEBUG_DD_MERGE_ATTENTIONPREPRO=1
    )
    
target_compile_definitions(pass_dd_merge_attentionprepro PUBLIC "-DENABLE_VAIP_PASS_DD_MERGE_ATTENTIONPREPRO=1")

vai_add_library(
  NAME
  pass_dd_merge_attentionprepro_mxgan
  INCLUDE_DIR
  include
  SRCS
  src/pass_main_mxgan.cpp)

target_link_libraries(pass_dd_merge_attentionprepro_mxgan PRIVATE vaip::core vart::util glog::glog vaip::vaip_pattern_zoo)
target_compile_definitions(pass_dd_merge_attentionprepro_mxgan PUBLIC "-DENABLE_VAIP_PASS_DD_MERGE_ATTENTIONPREPRO=1")

vai_add_library(
  NAME
  pass_dd_merge_attentionprepro_mxpzi
  INCLUDE_DIR
  include
  SRCS
  src/pass_main_mxpzi.cpp)

target_link_libraries(pass_dd_merge_attentionprepro_mxpzi PRIVATE vaip::core vart::util glog::glog vaip::vaip_pattern_zoo)
target_compile_definitions(pass_dd_merge_attentionprepro_mxpzi PUBLIC "-DENABLE_VAIP_PASS_DD_MERGE_ATTENTIONPREPRO=1")

vai_add_library(
  NAME
  pass_dd_merge_attentionprepro_psw
  INCLUDE_DIR
  include
  SRCS
  src/pass_main_psw.cpp)

target_link_libraries(pass_dd_merge_attentionprepro_psw PRIVATE vaip::core vart::util glog::glog vaip::vaip_pattern_zoo)
target_compile_definitions(pass_dd_merge_attentionprepro_psw PUBLIC "-DENABLE_VAIP_PASS_DD_MERGE_ATTENTIONPREPRO=1")