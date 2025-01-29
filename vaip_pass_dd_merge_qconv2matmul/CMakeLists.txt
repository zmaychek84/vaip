#
#  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
#  Licensed under the MIT License.
#
vai_add_library(
  NAME
  pass_dd_merge_qconv2matmul_0
  SRCS
  src/pattern_0.cpp)

target_link_libraries(pass_dd_merge_qconv2matmul_0 PRIVATE vaip::core vart::util glog::glog vaip::vaip_pattern_zoo)
target_compile_definitions(pass_dd_merge_qconv2matmul_0 PUBLIC "-DENABLE_VAIP_PASS_DD_MERGE_QCONV2MATMUL=1")

vai_add_library(
  NAME
  pass_dd_merge_qconv2matmul_1
  SRCS
  src/pattern_1.cpp)

target_link_libraries(pass_dd_merge_qconv2matmul_1 PRIVATE vaip::core vart::util glog::glog vaip::vaip_pattern_zoo)
target_compile_definitions(pass_dd_merge_qconv2matmul_1 PUBLIC "-DENABLE_VAIP_PASS_DD_MERGE_QCONV2MATMUL=1")

vai_add_library(
  NAME
  pass_dd_merge_qconv2matmul_2
  SRCS
  src/pattern_2.cpp)

target_link_libraries(pass_dd_merge_qconv2matmul_2 PRIVATE vaip::core vart::util glog::glog vaip::vaip_pattern_zoo)
target_compile_definitions(pass_dd_merge_qconv2matmul_2 PUBLIC "-DENABLE_VAIP_PASS_DD_MERGE_QCONV2MATMUL=1")

vai_add_library(
  NAME
  pass_dd_merge_qconv2matmul_3
  SRCS
  src/pattern_3.cpp)

target_link_libraries(pass_dd_merge_qconv2matmul_3 PRIVATE vaip::core vart::util glog::glog vaip::vaip_pattern_zoo)
target_compile_definitions(pass_dd_merge_qconv2matmul_3 PUBLIC "-DENABLE_VAIP_PASS_DD_MERGE_QCONV2MATMUL=1")

vai_add_library(
  NAME
  pass_dd_merge_qconv2matmul_4
  SRCS
  src/pattern_4.cpp)

target_link_libraries(pass_dd_merge_qconv2matmul_4 PRIVATE vaip::core vart::util glog::glog vaip::vaip_pattern_zoo)
target_compile_definitions(pass_dd_merge_qconv2matmul_4 PUBLIC "-DENABLE_VAIP_PASS_DD_MERGE_QCONV2MATMUL=1")