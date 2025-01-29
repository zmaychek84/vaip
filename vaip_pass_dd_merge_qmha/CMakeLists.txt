#
#  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
#  Licensed under the MIT License.
#
if(TARGET vaip::pass_dd_merge_qmhachannel)
vai_add_library(
  NAME
  pass_dd_merge_qmha
  INCLUDE_DIR
  include
  SRCS
  src/pass_main.cpp)

target_link_libraries(pass_dd_merge_qmha PRIVATE vaip::core vaip::pass_dd_merge_qmhachannel vart::util glog::glog vaip::vaip_pattern_zoo)
target_compile_definitions(pass_dd_merge_qmha PUBLIC "-DENABLE_VAIP_PASS_DD_MERGE_QMHA=1")
endif(TARGET vaip::pass_dd_merge_qmhachannel)
