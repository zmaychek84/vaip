#
#  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
#  Licensed under the MIT License.
#
find_package(nlohmann_json REQUIRED)
vai_add_library(NAME pass_dd_merge_qconcateops INCLUDE_DIR include SRCS src/pass_main.cpp)
vai_add_library(NAME pass_dd_merge_qconcateops_1_1 INCLUDE_DIR include SRCS src/pass_main_1_1.cpp)
vai_add_library(NAME pass_dd_merge_qconcateops_1_2 INCLUDE_DIR include SRCS src/pass_main_1_2.cpp)
vai_add_library(NAME pass_dd_merge_qconcateops_1_3 INCLUDE_DIR include SRCS src/pass_main_1_3.cpp)
vai_add_library(NAME pass_dd_merge_qconcateops_1_2_1 INCLUDE_DIR include SRCS src/pass_main_1_2_1.cpp)
target_link_libraries(pass_dd_merge_qconcateops PRIVATE vaip::core nlohmann_json::nlohmann_json vaip::vaip_pattern_zoo)
target_link_libraries(pass_dd_merge_qconcateops_1_1 PRIVATE vaip::core nlohmann_json::nlohmann_json vaip::vaip_pattern_zoo)
target_link_libraries(pass_dd_merge_qconcateops_1_2 PRIVATE vaip::core nlohmann_json::nlohmann_json vaip::vaip_pattern_zoo)
target_link_libraries(pass_dd_merge_qconcateops_1_2_1 PRIVATE vaip::core nlohmann_json::nlohmann_json vaip::vaip_pattern_zoo)
target_link_libraries(pass_dd_merge_qconcateops_1_3 PRIVATE vaip::core nlohmann_json::nlohmann_json vaip::vaip_pattern_zoo)
