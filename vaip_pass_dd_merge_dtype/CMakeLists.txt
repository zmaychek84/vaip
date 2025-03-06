#
#  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
#  Licensed under the MIT License.
#
vai_add_library(
  NAME
  pass_dd_merge_dtype
  INCLUDE_DIR
  include
  SRCS
  src/pass_main.cpp)

target_link_libraries(pass_dd_merge_dtype PRIVATE vaip::core vart::util glog::glog)
vai_add_debug_command(pass_dd_merge_dtype
    COMMAND "$<TARGET_FILE:voe_py_pass>"
    ARGUMENTS "-i C:\\temp\\$ENV{USERNAME}\\vaip\\.cache\\CUR\\onnx.onnx -o c:\\temp\\a.onnx -t c:\\temp\\a.txt -p vaip-pass_dd_merge_dtype -c  C:\\temp\\$ENV{USERNAME}\\vaip\\.cache\\CUR"
    ENVIRONMENT DEBUG_DD_MERGE_DTYPE=1
    )
target_compile_definitions(pass_dd_merge_dtype PUBLIC "-DENABLE_VAIP_PASS_DD_MERGE_DTYPE=1")

vai_add_library(
  NAME
  pass_dd_merge_dtype_mzdk5
  INCLUDE_DIR
  include
  SRCS
  src/mzdk5_pass_main.cpp)
target_link_libraries(pass_dd_merge_dtype_mzdk5 PRIVATE vaip::core vart::util glog::glog)
target_compile_definitions(pass_dd_merge_dtype_mzdk5 PUBLIC "-DENABLE_VAIP_PASS_DD_MERGE_DTYPE=1")

vai_add_library(
  NAME
  pass_dd_merge_dtype_infer
  INCLUDE_DIR
  include
  SRCS
  src/dtype_infer_pass_main.cpp)
target_link_libraries(pass_dd_merge_dtype_infer PRIVATE vaip::core vart::util glog::glog)
target_compile_definitions(pass_dd_merge_dtype_infer PUBLIC "-DENABLE_VAIP_PASS_DD_MERGE_DTYPE=1")

vai_add_library(
  NAME
  pass_dd_merge_shape_mzdk5
  INCLUDE_DIR
  include
  SRCS
  src/mzdk5_shape_pass.cpp)
target_link_libraries(pass_dd_merge_shape_mzdk5 PRIVATE vaip::core vart::util glog::glog)
target_compile_definitions(pass_dd_merge_shape_mzdk5 PUBLIC "-DENABLE_VAIP_PASS_DD_MERGE_DTYPE=1")

vai_add_library(
  NAME
  pass_dd_merge_DDMergeShape_psw
  INCLUDE_DIR
  include
  SRCS
  src/psw_pass_main.cpp)

target_link_libraries(pass_dd_merge_DDMergeShape_psw PRIVATE vaip::core vart::util glog::glog)

vai_add_library(
  NAME
  pass_dd_merge_EndPass_ds
  INCLUDE_DIR
  include
  SRCS
  src/ds_pass_main.cpp)

target_link_libraries(pass_dd_merge_EndPass_ds PRIVATE vaip::core vart::util glog::glog)

vai_add_library(
  NAME
  pass_dd_merge_EndPass_psu
  INCLUDE_DIR
  include
  SRCS
  src/psu_pass_main.cpp)

target_link_libraries(pass_dd_merge_EndPass_psu PRIVATE vaip::core vart::util glog::glog)

vai_add_library(
  NAME
  pass_dd_merge_qdq_adjustment
  INCLUDE_DIR
  include
  SRCS
  src/psr_qdq_adjustment.cpp)

target_link_libraries(pass_dd_merge_qdq_adjustment PRIVATE vaip::core vart::util glog::glog vaip::vaip_pattern_zoo)
vai_add_debug_command(pass_dd_merge_qdq_adjustment
    COMMAND "$<TARGET_FILE:voe_py_pass>"
    ARGUMENTS "-i C:\\temp\\$ENV{USERNAME}\\vaip\\.cache\\CUR\\onnx.onnx -o c:\\temp\\a.onnx -t c:\\temp\\a.txt -p vaip-pass_dd_merge_qdq_adjustment -c  C:\\temp\\$ENV{USERNAME}\\vaip\\.cache\\CUR"
    ENVIRONMENT DEBUG_DD_MERGE_QGELU_QDQ=1
    )
target_compile_definitions(pass_dd_merge_qdq_adjustment PUBLIC "-DENABLE_VAIP_PASS_DD_MERGE_QDQ_ADJUSTMENT=1")