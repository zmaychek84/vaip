#
#  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
#  Licensed under the MIT License.
#
vai_add_library(
  NAME
  pass_dd_merge_qsoftmax
  INCLUDE_DIR
  include
  SRCS
  src/pass_main.cpp)

target_link_libraries(pass_dd_merge_qsoftmax PRIVATE vaip::core vart::util glog::glog)
vai_add_debug_command(pass_dd_merge_qsoftmax
    COMMAND "$<TARGET_FILE:voe_py_pass>"
    ARGUMENTS "-i C:\\temp\\$ENV{USERNAME}\\vaip\\.cache\\CUR\\onnx.onnx -o c:\\temp\\a.onnx -t c:\\temp\\a.txt -p vaip-pass_dd_merge_qsoftmax -c  C:\\temp\\$ENV{USERNAME}\\vaip\\.cache\\CUR"
    ENVIRONMENT DEBUG_DD_MERGE_QSOFTMAX=1
    )
target_compile_definitions(pass_dd_merge_qsoftmax PUBLIC "-DENABLE_VAIP_PASS_DD_MERGE_QSOFTMAX=1")
