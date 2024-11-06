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

## CI Related Parameters Introduction

### CI ENV LIST
```shell
# examples of setting ci env:
MODEL_TYPE=xmodel # set in CI_ENV if no selectable MODEL_TYPE parameter in job
OUTPUT_CHECKING=cpu_runner,onnx_ep # set in CI_ENV if no selectable OUTPUT_CHECKING parameter in job
TIMEOUT=300
VAITRACE_PROFILING=true
VAI_AIE_OVERCLOCK_MHZ=1810
TEST_TIME=60
TEST_HELLO_WORLD=false
SKIP_CPU_EP=true
ACCURACY_TEST_TIMEOUT=36000
CI_REBOOT_BOARD=true
DPM_LEVEL=7
LAYER_COMPARE=false
PERF_XMODEL=false
VART_PERF_CYCLES=500
VART_PERF_MULTI_FUNC=true
VART_PERF_FUNC_CYCLES=10
VART_PERF_FUNC_THREADS=2
IPUTRACE=false
```

#### 1. TEST_MODE
Select one from below:
1. `performance`: test model performance and profiling.
2. `accuracy`: test model accuracy.
3. `functionality`: test model functionality.

###### <font color="red">Note!!! CI doesn't support `vart_perf` mode anymore!!!</font>

#### 2. MODEL_TYPE
Select one from below, or set in CI_ENV if no selectable MODEL_TYPE parameter:
1. `onnx`
2. `xmodel`: for vart_perf flow

#### 3. OUTPUT_CHECKING
Compare results under different modes.

Select one from below, or set in CI_ENV if no selectable OUTPUT_CHECKING parameter:
1. In performance test: 
   1. set to `false` to skip comparison between CPU_RUNNER and VITISAI_EP.
   2. otherwise, only support `cpu_runner,onnx_ep` mode. And will use `test_xmodel_diff` to compare layer by layer if the result is "Mismatch" or "TIMEOUT".
2. In accuracy/functionality test:
   1. `cpu_ep,onnx_ep` 
   2. `cpu_ep.cpu_runner` 
   3. `cpu_runner,onnx_ep` 
   4. `cpu_ep,cpu_ep`: float vs quantized
    
#### 4. TIMEOUT
Limit of each model's test time. Will reboot board if exceeding this limit, and then incrementally test the remaining models.

Default is `3600` seconds.

#### 5. VAITRACE_PROFILING
Generate profiling excel after performance test if set to `true`.

#### 6. VAI_AIE_OVERCLOCK_MHZ
Clock frequency used by vaitrace. Should be clarified if `VAITRACE_PROFILING=true`.

#### 7. TEST_TIME
Default is `30` seconds.

#### 8. TEST_HELLO_WORLD
Compare VITISAI_EP with GRAPH_ENGINE if set to `true`.

Default is `true`.

#### 9. SKIP_CPU_EP
Skip CPU test if set to `true`.

Default is `true`.

#### 10. ACCURACY_TEST_TIMEOUT
Time limit of whole accuracy test. Will reboot board and exit test if exceeding this limit.

Default is 24 hours.

#### 11. CI_REBOOT_BOARD
Reboot board before test if set to `true`.

#### 12. DPM_LEVEL
Used to set dpm level, `1` for 1.0GHz, `7` for 1.8GHz.

Will automatically set ipu clock before test, and reset to 1.8GHz after test.

Default is `7`.

#### 13. LAYER_COMPARE
Use `test_xmodel_diff` to compare layer by layer.

#### 14. PERF_XMODEL
Use `vart_perf` to test performance.

#### 15. VART_PERF_CYCLES
Cycles that `vart_perf` runs.

Default is `3000`.

#### 16. VART_PERF_MULTI_FUNC
`vart_perf` supports multi-thread test. Set it to `true` to test functionality with multiple threads.

#### 17. VART_PERF_FUNC_CYCLES
Cycles that `vart_perf` runs with multi-thread.

Default is `10`.

#### 18. VART_PERF_FUNC_THREADS
Threads that `vart_perf` runs.

Default is `2`.

###### <font color="red">Note!!! Threads more than `2` may cause timeout or board hang!!!</font>

#### 19. IPUTRACE
Use iputrace tool to catch more information.

#### 20. CI_VAIP_CONFIG
Use user defined config file instead of default vaip_config.json.
```shell
# example
CI_VAIP_CONFIG=vaip_config_cxx.json
```

