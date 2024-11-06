/*
 *  Copyright (C) 2022 Xilinx, Inc. All rights reserved.
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 **/
#include "reserved_symbols.cpp1.inc"
/// @file vaip_symbols.cpp
/// @brief This file contains the list of reserved symbols for VAIP passes
/**  Why we need this?
 *
 * To prevent linker from doing garbage collection.
 *
 * It is not so good that every new pass or customize op to update this file. It
 * leads too many git conflicts.
 *
 * Instead, every pass/custom_op need to create a text file named
 * `symbols.txt` in its root
 * directory. onnxruntime_vitisai_ep/CMakeLists.txt parses
 * `symbols.txt` and update this file accordingly.
 *
 */
#ifdef ENABLE_PATTERN_ZOO
#  include <vaip/pattern_zoo.hpp>
#endif
typedef void* void_ptr_t;
// clang-format off
void_ptr_t vaip_reserved_symbols[] = {
#include "reserved_symbols.cpp2.inc"
};
// clang-format on