<!--
    Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
    Licensed under the MIT License.
 -->

#### Prerequisites
1. SDK (version >= 2023.1) (Recommended for SDK2023.1)

### cross compile
```
1. unset LD_LIBRARY_PATH
2. source /opt/petalinux/2023.1/environment-setup-cortexa72-cortexa53-xilinx-linux
3. git clone git@gitenterprise.xilinx.com:VitisAI/vai-rt.git
4. cd vai-rt
5. env WITH_XCOMPILER=ON python main.py
   or env WITH_XCOMPILER=ON python main.py --type=release
   or env WITH_XCOMPILER=ON python main.py --dev-mode
   or env WITH_XCOMPILER=ON python main.py --dev-mode --type=release
```
comments(1): default is debug type, if add ```--type=release```, will build release type

comments(2): **maybe your source code folder has pervious source code, if you build with ```--dev-mode```, it will not auto pull the latest code and directly use the code in your source code folder, this means development mode, and in this mode, you can modify your code. if you build not with ```--dev-mode```, it will auto pull the latest code following the git url and commit id in ./recipy/.py, in this mode, you can't modify the source code, even though you modified, it will be covered anyway.**

comments(3): if you want build only one project, you can add ```--project```, for
example ```env WITH_XCOMPILER=ON python main.py --project vaip``` only build
vaip project.

### copy to board
```
1. copy /opt/petalinux/2023.1/sysroots/cortexa72-cortexa53-xilinx-linux/install/Debug or /opt/petalinux/2023.1/sysroots/cortexa72-cortexa53-xilinx-linux/install/Release to board, such as (board_ip):~/

2. copy /opt/petalinux/2023.1/sysroots/cortexa72-cortexa53-xilinx-linux/usr/lib/libcpuinfo.so to (board_ip):~/Debug/lib/ or (board_ip):~/Release/lib/

3. copy /opt/petalinux/2023.1/sysroots/cortexa72-cortexa53-xilinx-linux/usr/lib/libre2.so to (board_ip):~/Debug/lib/ or (board_ip):~/Release/lib/

4. copy and rename ~/build/build.linux.2023.1.aarch64.Debug/vaip/onnxruntime_vitisai_ep/python/dist/voe-0.1.0-cp310-cp310-linux_x86_64.whl to (board_ip):~/Debug/voe-0.1.0-py3-none-any.whl or ~/build/build.linux.2023.1.aarch64.Release/vaip/onnxruntime_vitisai_ep/python/dist/voe-0.1.0-cp310-cp310-linux_x86_64.whl to (board_ip):~/Release/voe-0.1.0-py3-none-any.whl

5. copy and rename ~/build/build.linux.2023.1.aarch64.Debug/onnxruntime/Release/dist/onnxruntime_vitisai-1.15.1-cp310-cp310-linux_x86_64.whl to (board_ip):~/Debug/onnxruntime_vitisai-1.15.1-py3-none-any.whl or ~/build/build.linux.2023.1.aarch64.Release/onnxruntime/Release/dist/onnxruntime_vitisai-1.15.1-cp310-cp310-linux_x86_64.whl to (board_ip):~/Release/onnxruntime_vitisai-1.15.1-py3-none-any.whl
```

### test on board

```
1. export LD_LIBRARY_PATH=~/Debug/lib or export LD_LIBRARY_PATH=~/Release/lib
2. export PATH=~/Debug/bin:$PATH or export PATH=~/Release/bin:$PATH
3. export PYTHONPATH=~/Debug/lib/python3.10/site-packages or export PYTHONPATH=~/Release/lib/python3.10/site-packages

4. pip install ~/Debug/voe-0.1.0-py3-none-any.whl or ~/Release/voe-0.1.0-py3-none-any.whl

5. pip install ~/Debug/onnxruntime_vitisai-1.15.1-py3-none-any.whl or ~/Release/onnxruntime_vitisai-1.15.1-py3-none-any.whl

6. C++ sample: ~/Debug/bin/resnet50_pt -c ~/Debug/bin/vaip_config.json ./ResNet_int.onnx ./sample_classification.jpg or ~/Release/bin/resnet50_pt -c ~/Release/bin/vaip_config.json ./ResNet_int.onnx ./sample_classification.jpg

7. python sample:
7.1 copy source code folder: /workspace/test_onnx_runner/resnet50_python to (board_ip):~/
7.2 copy resnet50_pt.onnx and ~/Debug/bin/vaip_config.json or ~/Release/bin/vaip_config.json to the correct path matching with line 28-31 in resnet50_python/test.py
7.3 python ~/resnet50_python/test.py
```
comments(1): if your board can't connect to internet, when you execute the above
4/5 pip install, it maybe report lack dependencies. another solution is copy
XCD package folder: /group/dphi_software/software/workspace/zhenzew/share/package to your board, then pip install add ```--find-links=package```
