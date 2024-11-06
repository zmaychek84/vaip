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
prefix=/home/public/zhaolin/pytorch

echo "run models begin ..."
cd /workspace/aisw/vaip/test/;
exec 3> report.txt
while read onnx_model xir_model
do
    echo "onnx_model:" $onnx_model;
    echo "xir_model:" $xir_model;
    if bash -e run_model.sh  $prefix/$onnx_model $prefix/$xir_model </dev/null; then
        (echo "# OK $onnx_model "
         echo "# $(md5sum $prefix/$onnx_model) "
         echo "    bash -e run_model.sh  $prefix/$onnx_model $prefix/$xir_model " )>&3
    else
        (
            md5=$(md5sum $prefix/$onnx_model | cut -f 1 -d ' ')
            echo "# FAIL $onnx_model "
            echo "# $md5 "
            echo "    bash -e run_model.sh  $prefix/$onnx_model $prefix/$xir_model"
            echo "    bash -e compile_model.sh  $prefix/$onnx_model $prefix/$xir_model"
            echo "    sshpass -p root ssh root@10.176.179.252 /home/test_dpu_runner /home/$md5/compiled.ref.xmodel /home/$md5/compiled.xmodel"
        ) >&3
    fi
done < model_zoo.list

exec 3>&-
echo "run models end"
