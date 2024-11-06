<!--  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License. -->
## ONNXRuntime Vitis AI EP environment variables and Json config fields
### Environment Variables

| Env Var  | Default Value | Details |
| -------- | -------- | -------- |
| XLNX_ENABLE_DUMP_XIR_MODEL | 0   | Save the quantized xmodel(xir.xmodel) which is converted from ONNX model in cache directory or not.  |
| VAIP_COMPILE_RESERVE_CONST_DATA  | 0   | xcompiler option. Reserve data of const-fix operator in release mode or not. 1: `fat`  xmodel; 0: `thin` xmodel  |
| USE_CPU_RUNNER | 0 | Use CPU to simulate DPU operations |
| XLNX_TARGET_NAME | "" | DPU target name. On Adaptable SoCs, if not set, the DPU target name would be read automatically; On Windows, the default value is `AMD_AIE2_Nx4_Overlay` which is the DPU target name of the IPU PHX 1x4. `AMD_AIE2_5x4_Overlay` which is the DPU target name of IPU PHX 5x4 |
|XLNX_VART_FIRMWARE | "" | Configures the path location for the xclbin executable file that runs on the IPU. It is essential to configure this variable. Make sure the file name is aligned with `XLNX_TARGET_NAME` |
|XLNX_ONNX_EP_VERBOSE | 0 | 1 : show various component versions; 2 : show the values of environment variables which include DPU target name, xcompiler options and number of subgraphs assigned to the DPU. |
| XLNX_MINIMUM_NUM_OF_CONV | 2 | Filter by Number of Conv op for DPU compiler. If the number of Conv ops in the onnx model is less than XLNX_MINIMUM_NUM_OF_CONV, will not invoke xcompiler. |
| XLNX_ENABLE_STAT_LOG | 0 | 1 : show each DPU subgraph latency.  0 ： Not show each DPU subgraph latency.|
| XLNX_ENABLE_SUMMARY_LOG | 1 | 1 : show summary informations for number of operators and subgraphs .  0 ： Not show summary informations for number of operators and subgrahs.|
| XLNX_ENABLE_OLD_QDQ | 1 | 1 : fixnerun flow,  QuantizeLinear/DequantizeLinear convert to xir fix op.  0 : QDQ flow,QuantizeLinear/DequantizeLinear convert to xir quantize_linear/dequantize_linear op. |

### Json config fields (in vaip_config.json)

    - `passes[].passDpuParam.xcompilerAttrs` - xcompiler options.
        - debug_mode: "performance" or "debug"
        - opt_level: 0, Set optimization option level ranging from 0 to 3, the larger the number the higher optimization level. 0: Do not enable any optimization; 1: Enable mt_fusion optimization; 2: Enable mt_fusion, control_optimization and Shim-DMA; 3: Enable mt_fusion, control_optimization, Shim-DMA and at_fusion.
        - dpu_subgraph_num: 2, rollback all ops to CPU when the number of dpu subgraph exceeds the value of dpu_subgraph_num.

## Use Cases and instructions

### Run IPU PHX 1x4
 If you want to run xclbin in 1x4 mode, set the environment variable like this：
 ```
 set XLNX_TARGET_NAME=AMD_AIE2_Nx4_Overlay
 set XLNX_VART_FIRMWARE=${{ xclbin_dir }}/1x4.xclbin
 ```
 The XLNX_TARGET_NAME environment variable equals `AMD_AIE2_Nx4_Overlay` in default on Windows.

### Run IPU PHX 5x4
 If you want to  run xclbin in 5x4 mode, set the environment variable like this：
 ```
 set XLNX_TARGET_NAME=AMD_AIE2_5x4_Overlay
 set XLNX_VART_FIRMWARE=${{ xclbin_dir }}/5x4.xclbin
 ```
 Note: When switch between 1x4 and 5x4, one need to set `XLNX_TARGET_NAME` and `XLNX_VART_FIRMWARE` at the same time. Two file names must be aligned to one another.

### Run model using CPU simulation
If you want to use CPU simulation to run the DPU subgraph, set the environment variable like this：
```
set USE_CPU_RUNNER=1
set VAIP_COMPILE_RESERVE_CONST_DATA=1
```
Note: One need to set both `USE_CPU_RUNNER` and `VAIP_COMPILE_RESERVE_CONST_DATA`  equal 1 at the same time to run DPU simulator.

### Save quantized xmodel
If you want to save quantized xmodel converted from ONNX model, set environment variable like this：
```
set XLNX_ENABLE_DUMP_XIR_MODEL=1
```
`xir.xmodel` would be the name of the quantized xmodel saved under the cache directory.

### Show verbose info
You can change verbose level with change the `XLNX_ONNX_EP_VERBOSE`
```
set XLNX_ONNX_EP_VERBOSE=1
```
when `XLNX_ONNX_EP_VERBOSE` equals 1, which means print various component versions. Like the following:
```
[XLNX_ONNX_EP_VERBOSE] vaip.1.0.0: fc7a97d9277468b4e88ca32b70f2916879c67eed
[XLNX_ONNX_EP_VERBOSE] target-factory.3.5.0: e22f69d1e5a78dca91dcc512ef2eee660c466b27
[XLNX_ONNX_EP_VERBOSE] xcompiler.3.5.0: 4540ac3637a2e5165f75b568a288bba7b6ddfac0
[XLNX_ONNX_EP_VERBOSE] onnxruntime.1.15.1: baeece44ba075009c6bfe95891a8c1b3d4571cb3
[XLNX_ONNX_EP_VERBOSE] xir.3.5.0: e33d922f1f0a47f855e807ec235aeebd61969d22
```
when euqals 2, which means print the xcompiler options values of target such as enable_pdi, reserve_const_data, opt_level and number of subgraph, like the following:
```
[XLNX_ONNX_EP_VERBOSE] XLNX_TARGET_NAME = AMD_AIE2_Nx4_Overlay
[XLNX_ONNX_EP_VERBOSE] VAIP_COMPILE_RESERVE_CONST_DATA = 1
[XLNX_ONNX_EP_VERBOSE] opt_level: 0
[XLNX_ONNX_EP_VERBOSE] USE_CPU_RUNNER = 1
```

### Use xcompiler optimizetion
The optimization functionality is disabled by default. if you want to enable xcompiler optimization, you can change `opt_level` field in `vaip_config.json` configuration file. Like:
```
        "opt_level" : {
            "intValue" : 1
        }
```
or
```
       "opt_level" : {
            "intValue" : 2
        }
```
You can choose 0,1,2,3 as an alternative value for this field.
 * 0: Do not enable any optimization;
 * 1: Enable mt_fusion optimization;
 * 2: Enable mt_fusion, control_optimization and Shim-DMA;
 * 3: Enable mt_fusion, control_optimization, Shim-DMA and at_fusion.

 ### Set the maximum number of dpu subgraphs allowed

Maximum number of dpu subgraphs run on IPU. All ops will rollback to CPU when the number of dpu subgraph exceeds the value of dpu_subgraph_num. Default value is 2 . If you want control this number, you can change `dpu_subgraph_num` field in `vaip_config.json` confuguration file. Like:
```
    "dpu_subgraph_num" : {
            "intValue" : 5
    }
```
