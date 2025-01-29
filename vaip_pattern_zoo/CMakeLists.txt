#
#  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
#  Licensed under the MIT License.
#


vai_add_library(NAME vaip_pattern_zoo
  INCLUDE_DIR include
  SRCS
  include/vaip/pattern_zoo.hpp
  src/all_pattern.cpp
)

target_link_libraries(vaip_pattern_zoo
  PRIVATE vaip::core vart::util glog::glog)

target_compile_definitions(vaip_pattern_zoo PRIVATE "-DVAIP_EXPORT_DLL=1"  PUBLIC "-DENABLE_PATTERN_ZOO=1")
