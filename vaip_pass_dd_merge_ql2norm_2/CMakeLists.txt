#
#  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
#  Licensed under the MIT License.
#
find_package(nlohmann_json REQUIRED)
vai_add_library(NAME pass_dd_merge_ql2norm_2 INCLUDE_DIR include SRCS src/pass_main.cpp)
target_link_libraries(pass_dd_merge_ql2norm_2 PRIVATE vaip::core nlohmann_json::nlohmann_json vaip::vaip_pattern_zoo)
vai_add_library(NAME pass_dd_merge_qlpnorm_2 INCLUDE_DIR include SRCS src/pass_main_lpnorm.cpp)
target_link_libraries(pass_dd_merge_qlpnorm_2 PRIVATE vaip::core nlohmann_json::nlohmann_json vaip::vaip_pattern_zoo)
