#
#   The Xilinx Vitis AI Vaip in this distribution are provided under the following free 
#   and permissive binary-only license, but are not provided in source code form.  While the following free 
#   and permissive license is similar to the BSD open source license, it is NOT the BSD open source license 
#   nor other OSI-approved open source license.
#
#    Copyright (C) 2022 Xilinx, Inc. All rights reserved.
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


set -e
onnx_model=$1;shift
xir_model=$1;shift

md5=$(md5sum $onnx_model | cut -f 1 -d ' ')
model_dir=$HOME/.local/onnx/models/$md5
target_name=DPUCZDX8G_ISA1_B4096

mkdir -p $model_dir;

env \
ENABLE_SAVE_GRAPH_TXT=1 \
DEBUG_COMPILE_MODEL=1 \
XLNX_ENABLE_CACHE=0 \
XLNX_ENABLE_DUMP_XIR_MODEL=1   \
XLNX_ENABLE_DUMP_ONNX_MODEL=1  \
XLNX_ENABLE_DUMP_COMPILED_MODEL=1 \
XLNX_TARGET_NAME=$target_name \
DUMMY_RUNNER_BATCH_SIZE=1 \
DEBUG_VITIS_AI_EP_DUMMY_RUNNER=1 \
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.local/Ubuntu.20.04.x86_64.Debug/lib \
    $HOME/build/build.Ubuntu.20.04.x86_64.Debug/vaip/test_onnx_runner "$onnx_model" &

pid1=$!

if [ ! -f "$model_dir/compiled.ref.xmodel" ]; then
    $HOME/.local/Ubuntu.20.04.x86_64.Debug/bin/xcompiler \
        -i "$xir_model" -o "$model_dir/compiled.ref.xmodel" -t $target_name
fi

wait $pid1
