<!--  Copyright (C) 2022 Xilinx, Inc. All rights reserved. 
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
    
## generate node/model/data and test on XCD (EXECUTION_PROVIDER is "cpu")

``` console
% mkdir -p  /workspace/work/onnx_test/node_test/
% cd /workspace/work/onnx_test/node_test/
% whereis backend-test-tools
% backend-test-tools generate-data -o .
% $HOME/build/onnxruntime/Debug/onnx_test_runner ./node/
% $HOME/build/onnxruntime/Debug/onnx_test_runner ./simple/test_sign_model/
% $HOME/build/onnxruntime/Debug/onnx_test_runner ./simple/test_sign_model/ -v
```

## copy lib/node/model/data to zcu102 and test (EXECUTION_PROVIDER is "cpu")

``` console
% rsync -avr /opt/petalinux/2021.2/sysroots/cortexa72-cortexa53-xilinx-linux/install/Debug hawkwang@xbjlabdpsvr16:/home/hawkwang/d/
% cd /workspace/work/
% rsync -avr onnx_test root@10.176.179.62:~/wangming/
% ssh root@10.176.179.62
% echo $LD_LIBRARY_PATH
% ls /group/xbjlab/
% export LD_LIBRARY_PATH=/group/xbjlab/dphi_software/software/workspace/hawkwang/d/Debug/lib
% cd wangming/onnx_test/node_test/simple/
% /group/xbjlab/dphi_software/software/workspace/hawkwang/d/Debug/bin/onnx_test_runner ./test_sign_model
% /group/xbjlab/dphi_software/software/workspace/hawkwang/d/Debug/bin/onnx_test_runner ./test_sign_model -v
```

