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


model_dir=$1;shift;
md5=$1;shift;

ip=10.176.179.103
vaip_model="$model_dir/compiled.xmodel"
ref_model="$model_dir/compiled.ref.xmodel"

echo "copying model to $ip"
md5sum "$vaip_model" "$ref_model"
if ( cd $model_dir/..; tar -czf - $md5/compiled.xmodel $md5/compiled.ref.xmodel ) |
       ssh xbjlabdpsvr16 "cat >/tmp/$USER.$md5.tar.gz" ; then
    echo " /tmp/$USER.$md5.tar.gz on xbjlabdpsvr16 is ready"
else
    echo " file to create /tmp/$USER.$md5.tar.gz on xbjlabdpsvr16"
    exit 1
fi

ssh -n xbjlabdpsvr16 "cat /tmp/$USER.$md5.tar.gz | sshpass -p root ssh root@$ip tar -xvzf - -C /home "
sshpass -p root ssh -n root@$ip "md5sum /home/$md5/compiled.ref.xmodel /home/$md5/compiled.xmodel ; xdputil xmodel -m /home/$md5/compiled.ref.xmodel; env XLNX_ENABLE_FINGERPRINT_CHECK=0 /home/test_dpu_runner /home/$md5/compiled.ref.xmodel /home/$md5/compiled.xmodel; rm -rf /home/$md5"
ssh -n xbjlabdpsvr16 "rm -rf /tmp/$USER.$md5.tar.gz"
