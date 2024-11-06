# Copyright (C) 2022 Xilinx Inc.
# Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
#
cmake_minimum_required(VERSION 3.17)
project(
  vaip
  VERSION 1.0.0
  LANGUAGES C CXX)

vai_add_library(NAME pass_convert_ending_blacklist_ops_to_unknown_op INCLUDE_DIR
                include SRCS src/pass_main.cpp)

target_link_libraries(pass_convert_ending_blacklist_ops_to_unknown_op
                      PRIVATE vaip::core vart::util glog::glog)
