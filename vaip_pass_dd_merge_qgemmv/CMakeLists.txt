#
#  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
#  Licensed under the MIT License.
#
vai_add_library(NAME pass_dd_merge_qgemmv INCLUDE_DIR include SRCS src/pass_main.cpp)
target_link_libraries(pass_dd_merge_qgemmv PRIVATE vaip::core vaip::vaip_pattern_zoo)
