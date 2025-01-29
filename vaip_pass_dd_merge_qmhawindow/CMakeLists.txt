#
#  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
#  Licensed under the MIT License.
#
if(TARGET vaip::vaip_pattern_zoo)
  vai_add_library(
    NAME
    pass_dd_merge_qmhawindow
    INCLUDE_DIR
    include
    SRCS
    src/pass_main.cpp)

  target_link_libraries(pass_dd_merge_qmhawindow PRIVATE vaip::core vaip::pass_dd_merge_qmhachannel vaip::vaip_pattern_zoo vart::util glog::glog)
  target_compile_definitions(pass_dd_merge_qmhawindow PUBLIC "-DENABLE_VAIP_PASS_DD_MERGE_QMHAWINDOW=1")
  vai_add_debug_command(pass_dd_merge_qmhawindow
    COMMAND "$<TARGET_FILE:voe_py_pass>"
    ARGUMENTS "-i C:\\temp\\$ENV{USERNAME}\\vaip\\.cache\\72651ec6ae2fe552dd0604527d73c1e5\\onnx.onnx -o c:\\temp\\a.onnx -t c:\\temp\\a.txt -p vaip-pass_dd_merge_qmhawindow -c   C:\\temp\\$ENV{USERNAME}\\vaip\\.cache\\72651ec6ae2fe552dd0604527d73c1e5"
    ENVIRONMENT DEBUG_DD_MERGE_QMHACHANNEL=1
  )

else(TARGET vaip::vaip_pattern_zoo)
  message(WARNING "ENABLE_PATTERN_ZOO is OFF, pass_dd_merge_qmhawindow is disalbed also")
endif(TARGET vaip::vaip_pattern_zoo)
