#!/bin/bash
#
#   The Xilinx Vitis AI Vaip in this distribution are provided under the following free 
#   and permissive binary-only license, but are not provided in source code form.  While the following free 
#   and permissive license is similar to the BSD open source license, it is NOT the BSD open source license 
#   nor other OSI-approved open source license.
#
#    Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
#
#    Redistribution and use in binary form only, without modification, is permitted provided that the following conditions are met:
#
#    1. Redistributions must reproduce the above copyright notice, this list of conditions and the following disclaimer in 
#    the documentation and/or other materials provided with the distribution.
#
#    2. The name of Xilinx, Inc. may not be used to endorse or promote products redistributed with this software without specific 
#    prior written permission.
#
#    THIS SOFTWARE IS PROVIDED BY XILINX, INC. "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, 
#    THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL XILINX, INC. 
#    BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
#    PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY 
#    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT 
#    OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE
#

source /opt/xilinx/xrt/setup.sh
### test p0 model group ###
#export MODEL_ZOO=nightly_48_vaiq_u8s8_p0
#export MODEL_ZOO=weekly_97_vaiq_u8s8_p0
#if only test one  p0 model like resnet18, you can open this environment
#export CASE_NAME=resnet18

### test p1 model group
export MODEL_ZOO=ipu_benchmark_real_data
#if only test one  p1 model like resnest14d, you can open this environment
export CASE_NAME="A3"
#export CASE_NAME="A3 C2 C4 E L"
export USER_ENV="ACCURACY_TEST=true"
#normally we run mismatch test
export TEST_MODE=accuracy
export OUTPUT_CHECKING=cpu_ep,cpu_runner
#compile onnx model not using cache
export XLNX_ENABLE_CACHE=0
export XLNX_TARGET_NAME=AMD_AIE2P_4x4_Overlay
#use latest xcoartifactory xclbin
#export XLNX_VART_FIRMWARE=https://xcoartifactory.xilinx.com/artifactory/PHX_test_case_package/xclbin/latest/1x4.xclbin
#use local xclbin
#export XLNX_VART_FIRMWARE=/proj/xsjhdstaff6/$USER/workspace/vaip.hui/hi/1x4.xclbin

#set your cache store directory
export XLNX_CACHE_DIR=`pwd`/cache

export DPU_SUBGRAPH_NUM="500"

#if you want incremental test, open this option.
#export INCREMENTAL_TEST=true

python ci/ci_main.py test_modelzoo -f ipu_ci.env -j 1
