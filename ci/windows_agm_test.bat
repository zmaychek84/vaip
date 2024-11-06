::
::   The Xilinx Vitis AI Vaip in this distribution are provided under the following free 
::   and permissive binary-only license, but are not provided in source code form.  While the following free 
::   and permissive license is similar to the BSD open source license, it is NOT the BSD open source license 
::   nor other OSI-approved open source license.
::
::    Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
::
::    Redistribution and use in binary form only, without modification, is permitted provided that the following conditions are met:
::
::    1. Redistributions must reproduce the above copyright notice, this list of conditions and the following disclaimer in 
::    the documentation and/or other materials provided with the distribution.
::
::    2. The name of Xilinx, Inc. may not be used to endorse or promote products redistributed with this software without specific 
::    prior written permission.
::
::    THIS SOFTWARE IS PROVIDED BY XILINX, INC. "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, 
::    THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL XILINX, INC. 
::    BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
::    PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY 
::    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT 
::    OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE
::
::test performance model group
set MODEL_ZOO=ipu_benchmark
::if only test one model like resnet50, you can open this environment
set CASE_NAME=resnet50

::performance test
set TEST_MODE=performance

::compile onnx model not using cache
set XLNX_ENABLE_CACHE=0

::use latest xcoartifactory xclbin
set XLNX_VART_FIRMWARE=https://xcoartifactory.xilinx.com/artifactory/PHX_test_case_package/xclbin/latest_strix/4x4.xclbin
::set XLNX_VART_FIRMWARE=https://xcoartifactory.xilinx.com/artifactory/PHX_test_case_package/xclbin/latest/4x4.xclbin
::use local xclbin
::set XLNX_VART_FIRMWARE=/proj/xsjhdstaff6/$USER/workspace/vaip.hui/hi/1x4.xclbin

::set your cache store directory
::set XLNX_CACHE_DIR=/proj/xsjhdstaff6/$USER/workspace/vaip.hui/cache

::forbid case
::set FORBID_CASE=F2 resnet50_v1.5 yolov8_DetectionModel resnet50_fp32 Apollo Iris Themis ECBSR720 Capcub_jianying_pc ECBSR720_NHWC UNet MobileNetEdgeTPU MobileDet_SSD Mosaic esrgan yolov3 resnet50 mobilenetv3 inceptionv4 deeplabv3

::user_env
set OPT_LEVEL=3
set NUM_OF_DPU_RUNNERS=1
set THREAD=1
set XLNX_ENABLE_STAT_LOG=0
set SKIP_CPU_EP=true
set TEST_TIME=60
set TARGET_TYPE=STRIX
::set TARGET_TYPE=PHOENIX
::set PERF_XMODEL=true
::set OUTPUT_CHECKING=true

::agm capture for generating .csv file, visualizer for getting BW etc.
set AGM_CAPTURE=true
set AGM_VISUALIZER=true


::if you want incremental test, open this option.
::set INCREMENTAL_TEST=true

python ci/ci_main.py test_modelzoo -f ipu_ci.env -j 1
