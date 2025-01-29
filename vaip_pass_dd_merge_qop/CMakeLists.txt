#
#  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
#  Licensed under the MIT License.
#
vai_add_library(
  NAME
  pass_dd_merge_qop
  SRCS
  src/pass_main.cpp)

target_link_libraries(pass_dd_merge_qop PRIVATE vaip::core vart::util glog::glog vaip::vaip_pattern_zoo)
vai_add_debug_command(pass_dd_merge_qop
    COMMAND "$<TARGET_FILE:voe_py_pass>"
    ARGUMENTS "-i C:\\temp\\$ENV{USERNAME}\\vaip\\.cache\\CUR\\onnx.onnx -o c:\\temp\\a.onnx -t c:\\temp\\a.txt -p vaip-pass_dd_merge_qop -c  C:\\temp\\$ENV{USERNAME}\\vaip\\.cache\\CUR"
    ENVIRONMENT DEBUG_DD_MERGE_QOP=1
    )
target_compile_definitions(pass_dd_merge_qop PUBLIC "-DENABLE_VAIP_PASS_DD_MERGE_QOP=1")

vai_add_library(
  NAME
  pass_dd_merge_qop_onnx
  SRCS
  src/pass_main_onnx.cpp)

target_link_libraries(pass_dd_merge_qop_onnx PRIVATE vaip::core vart::util glog::glog vaip::vaip_pattern_zoo)
vai_add_debug_command(pass_dd_merge_qop_onnx
    COMMAND "$<TARGET_FILE:voe_py_pass>"
    ARGUMENTS "-i C:\\temp\\$ENV{USERNAME}\\vaip\\.cache\\CUR\\onnx.onnx -o c:\\temp\\a.onnx -t c:\\temp\\a.txt -p vaip-pass_dd_merge_qop_onnx -c  C:\\temp\\$ENV{USERNAME}\\vaip\\.cache\\CUR"
    ENVIRONMENT DEBUG_DD_MERGE_QOP_onnx=1
    )
target_compile_definitions(pass_dd_merge_qop_onnx PUBLIC "-DENABLE_VAIP_PASS_DD_MERGE_QOP_ONNX=1")
