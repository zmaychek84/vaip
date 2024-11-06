<!-- Copyright (C) 2022 Xilinx, Inc. All rights reserved.
    Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License. -->
# 2D_UNet_LW
## copy models from XCD
```
% scp -r localhost:/proj/rdi/staff/zhaolin/code/regression/internal-cooperation-models/pytorch/2D_UNet_LW/code/quantize_result /workspace/aisw/onnx_models/2D_UNet_LW
% md5sum /workspace/aisw/onnx_models/2D_UNet_LW/unet_int.onnx



% cd /workspace/aisw/vaip/; ../Vitis-AI-Library/cmake.sh  --project vaip && env DUMMY_RUNNER_BATCH_SIZE=1 LD_LIBRARY_PATH=$HOME/.local/Ubuntu.20.04.x86_64.Debug/lib XLNX_ENABLE_DUMP_MODEL=1 XLNX_ENABLE_DUMP_GRAPH_TXT=1  DEBUG_VITIS_AI_EP=1 DEBUG_VITIS_AI_EP_DUMMY_RUNNER=1 DEBUG_DUMP_WEIGHTS=1 ENABLE_SKIP_PASS=0 DEBUG_EXPORT_TO_XIR=0 DEBUG_REPLACE_FIX=0 DEBUG_REWRITE_RULE=0 DEBUG_DPU_CONTROLLER=0   DEBUG_COMPILE_MODEL=1 DEBUG_VAIP_PASS=2 ~/build/build.Ubuntu.20.04.x86_64.Debug/vaip/test_onnx_runner /workspace/aisw/onnx_models/2D_UNet_LW/unet_int.onnx /workspace/aisw/001.JPEG 2>&1 | tee /tmp/test.log

```

F0607 01:48:28.257586 23566 replace_fix.cpp:74] Check failed: std::exp2f(fix_point) * x_scale_val == 1 (-nan vs. 1) x_scale_val inf fix_point -2147483648
*** Check failure stack trace: ***

DequantizerLinear(QuantizerLinear(input) y_scale value is "infinity"

## compare transposed_conv2d weights
```
% md5sum *.0.dat


% md5sum /tmp/283.*.dat
## e01bde99271f3636424ea32bd6f8343e
% ~/build/build.Ubuntu.20.04.x86_64.Debug/xir/tools/xir dump_txt /tmp/vaip.xir.xmodel /tmp/unet_onnx_fix.txt


% ~/build/build.Ubuntu.20.04.x86_64.Debug/xir/tools/xir dump_txt /workspace/aisw/onnx_models/2D_UNet_LW/unet_int.xmodel /tmp/unet_vai_fix.txt
% md5sum /tmp/*.0.dat | while read md5 name ; do  if [ `grep  -c $md5 /tmp/unet_vai_fix.txt` -ne 1 ]; then echo $md5 $name ; fi  ; done

% grep e01bde99271f3636424ea32bd6f8343e /tmp/unet_vai_fix.txt

% md5sum /tmp/343.*.dat
## b41c801e6b79fecc730398e97d8e0d25
% grep b41c801e6b79fecc730398e97d8e0d25 /tmp/unet_onnx_fix.txt


```
### test on board
```
% env LD_LIBRARY_PATH=/home/mingyue/.local/Ubuntu.20.04.x86_64.Debug/lib /home/mingyue/.local/Ubuntu.20.04.x86_64.Debug/bin/xcompiler -i /workspace/aisw/onnx_models/2D_UNet_LW/unet_int.xmodel -o /tmp/unet_vai.xmodel -t DPUCZDX8G_ISA1_B4096

% scp /tmp/vaip.compiled.xmodel root@10.176.179.103:/home/root/mingyue/unet/unet_onnx.xmodel
% scp /tmp/unet_vai.xmodel root@10.176.179.103:/home/root/mingyue/unet/unet_vai.xmodel

% ssh root@10.176.179.103
% xdputil xmodel -l /home/root/mingyue/unet/unet_onnx.xmodel
% xdputil xmodel -l /home/root/mingyue/unet/unet_vai.xmodel
% xdputil run -i 1 /home/root/mingyue/unet/unet_onnx.xmodel /home/root/mingyue/bcc/upload.bin
% xdputil run -i 1 /home/root/mingyue/unet/unet_vai.xmodel /home/root/mingyue/bcc/upload.bin
```




# VCK190
```
% rsync -avr /opt/petalinux/2022.1/sysroots/cortexa72-cortexa53-xilinx-linux/install/Debug mingyue@xbjlabdpsvr16:/group/xbjlab/dphi_software/software/workspace/mingyue/d/

% ssh root@10.176.179.74
% mount -t nfs -o nolock 10.176.178.33:/group_xbjlab/ /group/xbjlab/
% export LD_LIBRARY_PATH=/group/xbjlab/dphi_software/software/workspace/mingyue/d/Debug/lib
% export PATH=/group/xbjlab/dphi_software/software/workspace/mingyue/d/Debug/bin:$PATH

% /group/xbjlab/dphi_software/software/workspace/mingyue/d/Debug/bin/resnet50_pt /home/root/mingyue/ResNet_int.onnx /home/root/mingyue/sample_classification.jpg
```





# 3D_detection pointpillars-nuscenes
```
% scp -r localhost:/proj/rdi/staff/zhaolin/code/regression/internal-cooperation-models/pytorch/3d_detection/pointpillars-nuscenes/quantized/ /workspace/aisw/onnx_models/MVXFasterRCNN
% md5sum /workspace/aisw/onnx_models/MVXFasterRCNN/MVXFasterRCNN_quant_int.onnx

% cd /workspace/aisw/vaip/; ../Vitis-AI-Library/cmake.sh  --project vaip && env DUMMY_RUNNER_BATCH_SIZE=1 LD_LIBRARY_PATH=$HOME/.local/Ubuntu.20.04.x86_64.Debug/lib XLNX_ENABLE_DUMP_MODEL=1 XLNX_ENABLE_DUMP_GRAPH_TXT=1  DEBUG_VITIS_AI_EP=1 DEBUG_VITIS_AI_EP_DUMMY_RUNNER=1 DEBUG_DUMP_WEIGHTS=1 ENABLE_SKIP_PASS=0 DEBUG_EXPORT_TO_XIR=0 DEBUG_REPLACE_FIX=0 DEBUG_REWRITE_RULE=0 DEBUG_DPU_CONTROLLER=0   DEBUG_COMPILE_MODEL=1  DEBUG_VAIP_PASS=2 ~/build/build.Ubuntu.20.04.x86_64.Debug/vaip/test_onnx_runner /workspace/aisw/onnx_models/MVXFasterRCNN/MVXFasterRCNN_quant_int.onnx

```


# RenNet_BCC
```
% scp -r localhost:/proj/rdi/staff/zhaolin/code/regression/internal-cooperation-models/pytorch/Bayesian-Crowd-Counting/quantize_result /workspace/aisw/onnx_models/ResNet_BCC
% md5sum /workspace/aisw/onnx_models/ResNet_BCC/ResNet_BCC_int.onnx

% cd /workspace/aisw/vaip/; ../Vitis-AI-Library/cmake.sh  --project vaip && env DUMMY_RUNNER_BATCH_SIZE=1 LD_LIBRARY_PATH=$HOME/.local/Ubuntu.20.04.x86_64.Debug/lib XLNX_ENABLE_DUMP_MODEL=1 XLNX_ENABLE_DUMP_GRAPH_TXT=1  DEBUG_VITIS_AI_EP=1 DEBUG_VITIS_AI_EP_DUMMY_RUNNER=1 DEBUG_DUMP_WEIGHTS=1 ENABLE_SKIP_PASS=0 DEBUG_EXPORT_TO_XIR=0 DEBUG_REPLACE_FIX=0 DEBUG_REWRITE_RULE=0 DEBUG_DPU_CONTROLLER=0   DEBUG_COMPILE_MODEL=1 DEBUG_VAIP_PASS=2 ENABLE_SAVE_GRAPH_TXT=1   ~/build/build.Ubuntu.20.04.x86_64.Debug/vaip/test_onnx_runner /workspace/aisw/onnx_models/ResNet_BCC/ResNet_BCC_int.onnx


% gdb ~/build/build.Ubuntu.20.04.x86_64.Debug/vaip/test_onnx_runner
% set env DUMMY_RUNNER_BATCH_SIZE 1
% set env LD_LIBRARY_PATH /home/mingyue/.local/Ubuntu.20.04.x86_64.Debug/lib
% set env XLNX_ENABLE_DUMP_MODEL 1
% set env XLNX_ENABLE_DUMP_GRAPH_TXT 1
% set env DEBUG_VITIS_AI_EP 1
% set env DEBUG_VITIS_AI_EP_DUMMY_RUNNER 1
% set env DEBUG_VAIP_PASS 2
% run /workspace/aisw/onnx_models/ResNet_BCC/ResNet_BCC_int.onnx
```
## compile model usr xcompiler
```
% env LD_LIBRARY_PATH=/home/mingyue/.local/Ubuntu.20.04.x86_64.Debug/lib /home/mingyue/.local/Ubuntu.20.04.x86_64.Debug/bin/xcompiler -i /workspace/aisw/onnx_models/ResNet_BCC/ResNet_BCC_int.xmodel -o /tmp/bcc.xmodel -t DPUCZDX8G_ISA1_B4096

```

## dump fix model to txt
```
% ~/build/build.Ubuntu.20.04.x86_64.Debug/xir/tools/xir dump_txt /tmp/vaip.xir.xmodel /tmp/bcc_onnx_fix.txt
% ~/build/build.Ubuntu.20.04.x86_64.Debug/xir/tools/xir dump_txt /workspace/aisw/onnx_models/ResNet_BCC/ResNet_BCC_int.xmodel /tmp/bcc_vai_fix.txt
```
## dump compiled xmodel to txt
```
% ~/build/build.Ubuntu.20.04.x86_64.Debug/xir/tools/xir dump_txt /tmp/vaip.compiled.xmodel /tmp/bcc.txt
% ~/build/build.Ubuntu.20.04.x86_64.Debug/xir/tools/xir dump_txt /tmp/bcc.xmodel /tmp/bcc_pt.txt
### mc_code value: bytes = 2441588 md5sum = 7d1783605390755d0be4b39082a5a5f5
### value: bytes = 12655489 md5sum = ac9812af95219a25be11fe296ca09514

% ~/build/build.Ubuntu.20.04.x86_64.Debug/xir/tools/xir dump_reg /tmp/vaip.compiled.xmodel
% ~/build/build.Ubuntu.20.04.x86_64.Debug/xir/tools/xir dump_reg /tmp/bcc.xmodel
### weights different
% ~/build/build.Ubuntu.20.04.x86_64.Debug/xir/tools/xir dump_bin /tmp/vaip.compiled.xmodel /tmp/a
% ~/build/build.Ubuntu.20.04.x86_64.Debug/xir/tools/xir dump_bin /tmp/bcc.xmodel /tmp/b
% md5sum /tmp/a/REG_0.bin /tmp/b/REG_0.bin

```
## copy model to board
```
% scp root@10.176.179.103:/usr/share/vitis_ai_library/models/bcc_pt/bcc_pt.xmodel /workspace/aisw/onnx_models/ResNet_BCC/
% ~/build/build.Ubuntu.20.04.x86_64.Debug/xir/tools/xir dump_txt /workspace/aisw/onnx_models/ResNet_BCC/bcc_pt.xmodel /tmp/bcc_pt.txt
###mc_code value: bytes = 2409988 md5sum = faa13814c0421d51e6502ee680d71850
###REG_0 value: bytes = 12655489 md5sum = f9be4a586cb8d68bfd215762a249e500
% scp /tmp/vaip.compiled.xmodel root@10.176.179.103:/home/root/resnet_bcc.xmodel
```

## test on board
```
% scp /tmp/vaip.compiled.xmodel root@10.176.179.103:/home/root/mingyue/bcc/bcc_onnx.xmodel
% scp /tmp/vaip.compiled.xmodel root@10.176.179.103:/home/root/mingyue/bcc/bcc_onnx_old.xmodel
% scp /tmp/bcc.xmodel root@10.176.179.103:/home/root/mingyue/bcc/bcc_vai.xmodel
% ssh root@10.176.179.103

% cd /home/root/mingyue/bcc
% ls
% xdputil xmodel -l /home/root/mingyue/bcc/bcc_onnx.xmodel
% xdputil xmodel -l /home/root/mingyue/bcc/bcc_vai.xmodel
% xdputil run -i 1 /home/root/mingyue/bcc/bcc_onnx.xmodel /home/root/mingyue/bcc/upload.bin && md5sum 0.582.bin
% xdputil run -i 1 /home/root/mingyue/bcc/bcc_onnx_old.xmodel /home/root/mingyue/bcc/upload.bin && md5sum 0.582.bin
% xdputil run -i 1 /home/root/mingyue/bcc/bcc_vai.xmodel /home/root/mingyue/bcc/upload.bin && 0.ResNet_BCC__ResNet_BCC_Sequential_reg_layer__Conv2d_4__3970_fix.bin

% cp /usr/share/vitis_ai_library/models/bcc_pt/bcc_pt.prototxt /home/root/mingyue/bcc/bcc_onnx.prototxt
% cp /usr/share/vitis_ai_library/models/bcc_pt/bcc_pt.prototxt /home/root/mingyue/bcc/bcc_vai.prototxt
% /home/root/Vitis-AI/examples/Vitis-AI-Library/samples/bcc/test_jpeg_bcc /home/root/mingyue/bcc/bcc_onnx.xmodel /home/root/Vitis-AI/examples/Vitis-AI-Library/samples/bcc/sample_bcc.jpg
% /home/root/Vitis-AI/examples/Vitis-AI-Library/samples/bcc/test_jpeg_bcc /home/root/mingyue/bcc/bcc_vai.xmodel /home/root/Vitis-AI/examples/Vitis-AI-Library/samples/bcc/sample_bcc.jpg

% env XLNX_ENABLE_DUMP=1 XLNX_ENABLE_DEBUG_MODE=1  XLNX_SHOW_DPU_COUNTER=1 xdputil run -i 1 /home/root/mingyue/bcc/bcc_onnx.xmodel /home/root/mingyue/bcc/upload.bin
%

```




# ENet_xilinx
```
% scp  -r mingyue@localhost:/proj/rdi/staff/zhaolin/code/regression/internal-cooperation-models/pytorch/ENet_xilinx/quantize_result /workspace/aisw/onnx_models/enet
% md5sum /workspace/aisw/onnx_models/enet/ENet_int.onnx

%cd /workspace/aisw/vaip/; ../Vitis-AI-Library/cmake.sh  --project vaip && env DUMMY_RUNNER_BATCH_SIZE=1 LD_LIBRARY_PATH=$HOME/.local/Ubuntu.20.04.x86_64.Debug/lib XLNX_ENABLE_DUMP_MODEL=1 XLNX_ENABLE_DUMP_GRAPH_TXT=1  DEBUG_VITIS_AI_EP=1 DEBUG_VITIS_AI_EP_DUMMY_RUNNER=1 DEBUG_DUMP_WEIGHTS=1 ENABLE_SKIP_PASS=0 DEBUG_EXPORT_TO_XIR=0 DEBUG_REPLACE_FIX=0 DEBUG_REWRITE_RULE=0 DEBUG_DPU_CONTROLLER=0   DEBUG_COMPILE_MODEL=1 DEBUG_VAIP_PASS=2 ~/build/build.Ubuntu.20.04.x86_64.Debug/vaip/test_onnx_runner /workspace/aisw/onnx_models/enet/ENet_int.onnx /workspace/aisw/001.JPEG 2>&1 | tee /tmp/test.log



% env LD_LIBRARY_PATH=/home/mingyue/.local/Ubuntu.20.04.x86_64.Debug/lib /home/mingyue/.local/Ubuntu.20.04.x86_64.Debug/bin/xcompiler -i /workspace/aisw/onnx_models/enet/ENet_int.xmodel -o /tmp/enet_vai.xmodel -t DPUCZDX8G_ISA1_B4096

% scp /tmp/vaip.compiled.xmodel root@10.176.179.103:/home/root/mingyue/enet/enet_onnx.xmodel
% scp /tmp/unet_vai.xmodel root@10.176.179.103:/home/root/mingyue/enet/enet_vai.xmodel

% ssh root@10.176.179.103
% xdputil xmodel -l /home/root/mingyue/enet/enet_onnx.xmodel
% xdputil xmodel -l /home/root/mingyue/enet/enet_vai.xmodel
% xdputil run -i 1 /home/root/mingyue/enet/enet_onnx.xmodel /home/root/mingyue/enet/enet_onnx.xmodel && md5sum 0.2139.bin
% xdputil run -i 1 /home/root/mingyue/enet/enet_vai.xmodel /home/root/mingyue/enet/enet_onnx.xmodel && md5sum 0.unet__unet_Conv2d_final__3152_fix.bin

```



# SA-Gate
```
% scp -r mingyue@localhost:/proj/rdi/staff/zhaolin/code/regression/internal-cooperation-models/pytorch/SA-Gate/code/quantize_result /workspace/aisw/onnx_models/sa_gate
% md5sum /workspace/aisw/onnx_models/sa_gate/DeepLab_int.onnx


%cd /workspace/aisw/vaip/; ../Vitis-AI-Library/cmake.sh  --project vaip && env DUMMY_RUNNER_BATCH_SIZE=1 LD_LIBRARY_PATH=$HOME/.local/Ubuntu.20.04.x86_64.Debug/lib XLNX_ENABLE_DUMP_MODEL=1 XLNX_ENABLE_DUMP_GRAPH_TXT=1  DEBUG_VITIS_AI_EP=1 DEBUG_VITIS_AI_EP_DUMMY_RUNNER=1 DEBUG_DUMP_WEIGHTS=1 ENABLE_SKIP_PASS=0 DEBUG_EXPORT_TO_XIR=0 DEBUG_REPLACE_FIX=0 DEBUG_REWRITE_RULE=0 DEBUG_DPU_CONTROLLER=0   DEBUG_COMPILE_MODEL=1 DEBUG_VAIP_PASS=2 ~/build/build.Ubuntu.20.04.x86_64.Debug/vaip/test_onnx_runner /workspace/aisw/onnx_models/sa_gate/DeepLab_int.onnx /workspace/aisw/001.JPEG 2>&1 | tee /tmp/test.log

% env LD_LIBRARY_PATH=/home/mingyue/.local/Ubuntu.20.04.x86_64.Debug/lib /home/mingyue/.local/Ubuntu.20.04.x86_64.Debug/bin/xcompiler -i /workspace/aisw/onnx_models/sa_gate/DeepLab_int.xmodel -o /tmp/sa_gate_vai.xmodel -t DPUCZDX8G_ISA1_B4096


% scp /tmp/vaip.compiled.xmodel root@10.176.179.103:/home/root/mingyue/sa_gate/sa_gate_onnx.xmodel
% scp /tmp/sa_gate_vai.xmodel root@10.176.179.103:/home/root/mingyue/sa_gate/sa_gate_vai.xmodel

% xdputil run -i 2 /home/root/mingyue/sa_gate/sa_gate_vai.xmodel /home/root/mingyue/sa_gate/sa_gate_vai.xmodel /home/root/mingyue/sa_gate/sa_gate_vai.xmodel
% xdputil run -i 2 /home/root/mingyue/sa_gate/sa_gate_onnx.xmodel /home/root/mingyue/sa_gate/sa_gate_vai.xmodel /home/root/mingyue/sa_gate/sa_gate_vai.xmodel




```

# personreid_resnet18
```
% onnx_model=/proj/rdi/staff/zhaolin/code/regression/internal-cooperation-models/pytorch/person_reid/personreid_resnet18/reid_resnet18_quant_result/Baseline_int.onnx

% onnx_model=/proj/rdi/staff/zhaolin/code/regression/internal-cooperation-models/pytorch/person_reid/personreid_resnet50/reid_resnet50_quant_result/Baseline_int.onnx
% xir_model=/proj/rdi/staff/zhaolin/code/regression/internal-cooperation-models/pytorch/person_reid/personreid_resnet50/reid_resnet50_quant_result/Baseline_int.xmodel


% onnx_model=/proj/rdi/staff/zhaolin/code/regression/internal-cooperation-models/pytorch/pmg/pmg_quant_result/PMG_int.onnx
% xir_model=/proj/rdi/staff/zhaolin/code/regression/internal-cooperation-models/pytorch/pmg/pmg_quant_result/PMG_int.xmodel


% md5sum $onnx_model



% target_name=DPUCZDX8G_ISA1_B4096
% model_dir=/tmp
% cd /workspace/aisw/vaip/; ../Vitis-AI-Library/cmake.sh  --project vaip && env ENABLE_SAVE_GRAPH_TXT=1 XLNX_ENABLE_DUMP_XIR_MODEL="$model_dir/xir.xmodel"   XLNX_ENABLE_DUMP_ONNX_MODEL="$model_dir/onnx.onnx"  XLNX_ENABLE_DUMP_COMPILED_MODEL="$model_dir/compiled.xmodel" XLNX_ENABLE_DUMP_ONNX_GRAPH_TXT="$model_dir/onnx.txt" XLNX_TARGET_NAME=$target_name DUMMY_RUNNER_BATCH_SIZE=1 DEBUG_VITIS_AI_EP_DUMMY_RUNNER=1 DEBUG_VAIP_PASS=2 DEBUG_DUMP_WEIGHTS=1 LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.local/Ubuntu.20.04.x86_64.Debug/lib $HOME/build/build.Ubuntu.20.04.x86_64.Debug/vaip/test_onnx_runner $onnx_model

% ~/build/build.Ubuntu.20.04.x86_64.Debug/xir/tools/xir dump_txt $xir_model /tmp/vai_fix.txt
% md5sum /tmp/*.0.dat | while read md5 name ; do  if [ `grep  -c $md5 /tmp/vai_fix.txt` -ne 1 ]; then echo $md5 $name ; fi  ; done


% env LD_LIBRARY_PATH=/home/mingyue/.local/Ubuntu.20.04.x86_64.Debug/lib /home/mingyue/.local/Ubuntu.20.04.x86_64.Debug/bin/xcompiler -i $xir_model -o /tmp/sa_gate_vai.xmodel -t DPUCZDX8G_ISA1_B4096




% cd /workspace/aisw/vaip/; ../Vitis-AI-Library/cmake.sh  --project vaip && env ENABLE_SAVE_GRAPH_TXT=1 XLNX_ENABLE_DUMP_XIR_MODEL="/tmp/xir.xmodel"   XLNX_ENABLE_DUMP_ONNX_MODEL="/tmp/onnx.onnx"  XLNX_ENABLE_DUMP_COMPILED_MODEL="/tmp/compiled.xmodel" XLNX_ENABLE_DUMP_ONNX_GRAPH_TXT="/tmp/onnx.txt" XLNX_TARGET_NAME=DPUCZDX8G_ISA1_B4096 DUMMY_RUNNER_BATCH_SIZE=1 DEBUG_VITIS_AI_EP_DUMMY_RUNNER=1 DEBUG_VAIP_PASS=2 DEBUG_DUMP_WEIGHTS=1 LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.local/Ubuntu.20.04.x86_64.Debug/lib $HOME/build/build.Ubuntu.20.04.x86_64.Debug/vaip/test_onnx_runner $onnx_model/proj/rdi/staff/zhaolin/code/regression/internal-cooperation-models/pytorch/pmg/pmg_quant_result/PMG_int.onnx




% cd /workspace/aisw/vaip/; ../Vitis-AI-Library/cmake.sh  --project vaip
% env ENABLE_SAVE_GRAPH_TXT=1 XLNX_ENABLE_DUMP_XIR_MODEL="/tmp/xir.xmodel"   XLNX_ENABLE_DUMP_ONNX_MODEL="/tmp/onnx.onnx"  XLNX_ENABLE_DUMP_COMPILED_MODEL="/tmp/compiled.xmodel" XLNX_ENABLE_DUMP_ONNX_GRAPH_TXT="/tmp/onnx.txt" XLNX_TARGET_NAME=DPUCZDX8G_ISA1_B4096 DUMMY_RUNNER_BATCH_SIZE=1 DEBUG_VITIS_AI_EP_DUMMY_RUNNER=1 DEBUG_VAIP_PASS=2 DEBUG_DUMP_WEIGHTS=1 DEBUG_TO_XIR_PASS=1 LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.local/Ubuntu.20.04.x86_64.Debug/lib $HOME/build/build.Ubuntu.20.04.x86_64.Debug/vaip/test_onnx_runner /workspace/pytorch/movenet/quantize_result_new/MoveNet_int.onnx



% export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.local/Ubuntu.20.04.x86_64.Debug/lib
% gdb ~/build/build.Ubuntu.20.04.x86_64.Debug/vaip/test_onnx_runner

% sen env ENABLE_SAVE_GRAPH_TXT 1
% set env XLNX_ENABLE_DUMP_XIR_MODEL "/tmp/xir.xmodel"
% set env XLNX_ENABLE_DUMP_ONNX_MODEL "/tmp/onnx.onnx"
% set env XLNX_ENABLE_DUMP_COMPILED_MODEL "/tmp/compiled.xmodel"
% set env XLNX_ENABLE_DUMP_ONNX_GRAPH_TXT "/tmp/onnx.txt"
% set env XLNX_TARGET_NAME DPUCZDX8G_ISA1_B4096
% set env DUMMY_RUNNER_BATCH_SIZE 1
% set env DEBUG_VITIS_AI_EP_DUMMY_RUNNER 1
% set env DEBUG_VAIP_PASS 2
% set env DEBUG_DUMP_WEIGHTS 1
% set env DEBUG_TO_XIR_PASS 1
% set env LD_LIBRARY_PATH /home/mingyue/.local/Ubuntu.20.04.x86_64.Debug/lib

% run /workspace/pytorch/movenet/quantize_result_new/MoveNet_int.onnx

```



# model 24  debug
```

cd /workspace/aisw/vaip/; ../Vitis-AI-Library/cmake.sh  --project vaip && env ENABLE_SAVE_GRAPH_TXT=1 XLNX_ENABLE_DUMP_XIR_MODEL="/tmp/xir.xmodel"   XLNX_ENABLE_DUMP_ONNX_MODEL="/tmp/onnx.onnx"  XLNX_ENABLE_DUMP_COMPILED_MODEL="/tmp/compiled.xmodel" XLNX_ENABLE_DUMP_ONNX_GRAPH_TXT="/tmp/onnx.txt" XLNX_TARGET_NAME=DPUCZDX8G_ISA1_B4096 DUMMY_RUNNER_BATCH_SIZE=1 DEBUG_VITIS_AI_EP_DUMMY_RUNNER=1 DEBUG_VAIP_PASS=2 DEBUG_DUMP_WEIGHTS=1 DEBUG_TO_XIR_PASS=1 LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.local/Ubuntu.20.04.x86_64.Debug/lib $HOME/build/build.Ubuntu.20.04.x86_64.Debug/vaip/test_onnx_runner /home/public/modelzoo/internal-cooperation-models/pytorch/ultra-fast/culane_quantize_result/UltraFast_int.onnx


% ~/build/build.Ubuntu.20.04.x86_64.Debug/xir/tools/xir dump_txt /home/public/modelzoo/internal-cooperati\on-models/pytorch/ultra-fast/culane_quantize_result/UltraFast_int_20220624.xmodel /tmp/vai_fix.txt
% md5sum /tmp/*.bin | while read md5 name ; do  if [ `grep  -c $md5 /tmp/vai_fix.txt` -ne 1 ]; then echo $md5 $name ; fi  ; done



% cd /tmp

% ~/build/build.Ubuntu.20.04.x86_64.Debug/vaip/dump_const /tmp/xir.xmodel

% ~/build/build.Ubuntu.20.04.x86_64.Debug/vaip/dump_const  /home/public/modelzoo/internal-cooperation-models/pytorch/ultra-fast/culane_quantize_result/UltraFast_int.xmodel

% md5sum /tmp/55.bin
% md5sum /tmp/61.bin

```

```

```



```

% env ENABLE_SAVE_GRAPH_TXT=1 XLNX_ENABLE_DUMP_XIR_MODEL="/tmp/xir.xmodel"  XLNX_ENABLE_DUMP_ONNX_MODEL="/tmp/onnx.onnx"  XLNX_ENABLE_DUMP_COMPILED_MODEL="/tmp/compiled.xmodel" XLNX_ENABLE_DUMP_ONNX_GRAPH_TXT="/tmp/onnx.txt" XLNX_TARGET_NAME=DPUCZDX8G_ISA1_B4096 DUMMY_RUNNER_BATCH_SIZE=1 DEBUG_VITIS_AI_EP_DUMMY_RUNNER=1 DEBUG_VAIP_PASS=2 DEBUG_DUMP_WEIGHTS=1 DEBUG_TO_XIR_PASS=1 LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.local/Ubuntu.20.04.x86_64.Debug/lib DEBUG_COMPILE_MODEL_USE_VAIP=1 gdb --args $HOME/build/build.Ubuntu.20.04.x86_64.Debug/vaip/ort_vitis_ai_ep/test_onnx_runner /home/public/zhaolin/pytorch/ENet_xilinx/quantize_result/ENet_int.onnx


% gdb $HOME/build/build.Ubuntu.20.04.x86_64.Debug/vaip/ort_vitis_ai_ep/test_onnx_runner
% sen env ENABLE_SAVE_GRAPH_TXT 1
% set env XLNX_ENABLE_DUMP_XIR_MODEL "/tmp/xir.xmodel"
% set env XLNX_ENABLE_DUMP_ONNX_MODEL "/tmp/onnx.onnx"
% set env XLNX_ENABLE_DUMP_COMPILED_MODEL "/tmp/compiled.xmodel"
% set env XLNX_ENABLE_DUMP_ONNX_GRAPH_TXT "/tmp/onnx.txt"
% set env XLNX_TARGET_NAME DPUCZDX8G_ISA1_B4096
% set env DUMMY_RUNNER_BATCH_SIZE 1
% set env DEBUG_VITIS_AI_EP_DUMMY_RUNNER 1
% set env DEBUG_VAIP_PASS 2
% set env DEBUG_DUMP_WEIGHTS 1
% set env DEBUG_TO_XIR_PASS 1
% set env LD_LIBRARY_PATH $LD_LIBRARY_PATH:/home/mingyue/.local/Ubuntu.20.04.x86_64.Debug/lib
% set env DEBUG_COMPILE_MODEL_USE_VAIP 1
% r



```
