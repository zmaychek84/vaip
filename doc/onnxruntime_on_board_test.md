<!--
    Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
    Licensed under the MIT License.
 -->

### cross compile and copy file to xbjlab
```
% unset LD_LIBRARY_PATH; source /opt/petalinux/2022.2/environment-setup-cortexa72-cortexa53-xilinx-linux
% python main.py --type debug  --dev-mode
% rsync -avr /opt/petalinux/2022.2/sysroots/cortexa72-cortexa53-xilinx-linux/install/Debug mingyue@xbjlabdpsvr16:/group/xbjlab/dphi_software/software/workspace/mingyue/d/

%scp /opt/petalinux/2022.2/sysroots/cortexa72-cortexa53-xilinx-linux/usr/lib/libcpuinfo.so mingyue@xbjlabdpsvr16:/group/xbjlab/dphi_software/software/workspace/mingyue/d/Debug/lib/
%scp /opt/petalinux/2022.2/sysroots/cortexa72-cortexa53-xilinx-linux/usr/lib/libre2.so mingyue@xbjlabdpsvr16:/group/xbjlab/dphi_software/software/workspace/mingyue/d/Debug/lib/

% scp ~/build/build.linux.2022.2.aarch64.Debug/vaip/onnxruntime_vitisai_ep/python/dist/voe-0.1.0-cp39-cp39-linux_x86_64.whl mingyue@xbjlabdpsvr16:/group/xbjlab/dphi_software/software/workspace/mingyue/d/Debug/voe-0.1.0-py3-none-any.whl
```
### access internet by ssh proxy
```
% ssh xcoengvm229033
% ssh -R9181:localhost:3128 -J xbjjmphost01, xbjlabdpsvr16 root@10.176.178.83
```

### test on board

```
% ssh root@10.176.178.83
% mount -t nfs -o nolock 10.176.178.33:/group_xbjlab/ /group/xbjlab

% export LD_LIBRARY_PATH=/group/xbjlab/dphi_software/software/workspace/mingyue/d/Debug/lib
% export PATH=/group/xbjlab/dphi_software/software/workspace/mingyue/d/Debug/bin:$PATH
% #export PYTHONPATH=/group/xbjlab/dphi_software/software/workspace/mingyue/d/Debug/lib/python3.9/site-packages


% export http_proxy=http://localhost:9181
% export https_proxy=http://localhost:9181
% curl -v www.google.com

% wget https://bootstrap.pypa.io/get-pip.py
% python get-pip.py
% pip install virtualenv
% virtualenv cross
% source cross/bin/activate
% pip install /group/xbjlab/dphi_software/software/workspace/mingyue/d/Debug/voe-0.1.0-py3-none-any.whl
% pip install numpy --force-reinstall


% /group/xbjlab/dphi_software/software/workspace/mingyue/d/Debug/bin/resnet50_pt /home/root/mingyue/ResNet_int.onnx /home/root/mingyue/sample_classification.jpg
```


### python sample
```
% scp -r /workspace/test_onnx_runner/resnet50_python mingyue@xbjlabdpsvr16:/group/xbjlab/dphi_software/software/workspace/mingyue/d/Debug/

% scp ~/build/build.linux.2022.2.aarch64.Debug/onnxruntime/Debug/dist/onnxruntime_vitisai-1.15.0-cp39-cp39-linux_x86_64.whl mingyue@xbjlabdpsvr16:/group/xbjlab/dphi_software/software/workspace/mingyue/d/Debug/onnxruntime_vitisai-1.15.0-py3-none-any.whl

%pip install /group/xbjlab/dphi_software/software/workspace/mingyue/d/Debug/onnxruntime_vitisai-1.15.0-py3-none-any.whl
%python  /group/xbjlab/dphi_software/software/workspace/mingyue/d/Debug/resnet50_python/test.py

```
