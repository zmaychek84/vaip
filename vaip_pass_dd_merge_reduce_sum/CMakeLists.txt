#
#
# The Xilinx Vitis AI Vaip in this distribution are provided under the following
# free and permissive binary-only license, but are not provided in source code
# form.  While the following free and permissive license is similar to the BSD
# open source license, it is NOT the BSD open source license nor other
# OSI-approved open source license.
#
# Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
#
# Redistribution and use in binary form only, without modification, is permitted
# provided that the following conditions are met:
#
# 1. Redistributions must reproduce the above copyright notice, this list of
#   conditions and the following disclaimer in the documentation and/or other
#   materials provided with the distribution.
#
# 2. The name of Xilinx, Inc. may not be used to endorse or promote products
#   redistributed with this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY XILINX, INC. "AS IS" AND ANY EXPRESS OR IMPLIED
# WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
# EVENT SHALL XILINX, INC. BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
# BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
# IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE
#
vai_add_library(
  NAME
  pass_dd_merge_reduce_sum
  INCLUDE_DIR
  include
  SRCS
  src/pass_main.cpp)

target_link_libraries(pass_dd_merge_reduce_sum PRIVATE vaip::core vart::util glog::glog vaip::vaip_pattern_zoo)
vai_add_debug_command(pass_dd_merge_reduce_sum
    COMMAND "$<TARGET_FILE:voe_py_pass>"
    ARGUMENTS "-i C:\\temp\\$ENV{USERNAME}\\vaip\\.cache\\CUR\\onnx.onnx -o c:\\temp\\a.onnx -t c:\\temp\\a.txt -p vaip-pass_dd_merge_reduce_sum -c  C:\\temp\\$ENV{USERNAME}\\vaip\\.cache\\CUR"
    ENVIRONMENT DEBUG_DD_MERGE_REDUCE_SUM=1
    )
target_compile_definitions(pass_dd_merge_reduce_sum PUBLIC "-DENABLE_VAIP_PASS_DD_MERGE_REDUCE_SUM=1")
