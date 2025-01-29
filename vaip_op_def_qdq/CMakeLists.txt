#
#  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
#  Licensed under the MIT License.
#


cmake_minimum_required(VERSION 3.17)
project(vaip VERSION 1.0.0 LANGUAGES C CXX)
vai_add_library(NAME op_def_qdq SRCS src/opdef_main.cpp DEPENDS vaip::core)
# op_def_qdq needs to be written to op_def.txt 
