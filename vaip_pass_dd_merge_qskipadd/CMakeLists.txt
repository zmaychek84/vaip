#
#  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
#  Licensed under the MIT License.
#
vai_add_library(
  NAME
  pass_dd_merge_qskipadd
  SRCS
  src/pass_main.cpp)

target_link_libraries(pass_dd_merge_qskipadd PRIVATE vaip::core vart::util glog::glog vaip::vaip_pattern_zoo)
target_compile_definitions(pass_dd_merge_qskipadd PUBLIC "-DENABLE_VAIP_PASS_DD_MERGE_QSKIPADD=1")
