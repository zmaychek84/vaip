#
#  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
#  Licensed under the MIT License.
#
vai_add_library(
  NAME
  pass_dd_merge_qgroupnorm_0
  SRCS
  src/pattern_0.cpp)

target_link_libraries(pass_dd_merge_qgroupnorm_0 PRIVATE vaip::core vart::util glog::glog vaip::vaip_pattern_zoo)
target_compile_definitions(pass_dd_merge_qgroupnorm_0 PUBLIC "-DENABLE_VAIP_PASS_DD_MERGE_QGROUPNORM=1")

vai_add_library(
  NAME
  pass_dd_merge_qgroupnorm_1
  SRCS
  src/pattern_1.cpp)

target_link_libraries(pass_dd_merge_qgroupnorm_1 PRIVATE vaip::core vart::util glog::glog vaip::vaip_pattern_zoo)
target_compile_definitions(pass_dd_merge_qgroupnorm_1 PUBLIC "-DENABLE_VAIP_PASS_DD_MERGE_QGROUPNORM=1")