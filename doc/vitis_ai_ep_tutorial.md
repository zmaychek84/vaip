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


# Tutorials - ONNXRuntime Vitis AI Execution Provider

ONNXRuntime Vitis AI Execution Provider (Vitis AI EP) is provided to hardware-accelerated AI inference with AMD's DPU . It allows user to directly run the quantized ONNX model on the target board. The current Vitis AI EP inside ONNXRuntime enables acceleration of Neural Network model inference using embedded devices such as Zynq® UltraScale+™ MPSoC, Versal, Versal AI Edge and Kria cards.

Vitis AI ONNXRuntime Engine, short for VOE, which is the implementation library of Vitis AI EP.


## How to build

### On Edge
On Edge, you only need to download the image and installation package related to the developemnt board, burn the image, and install the required package according to the [Target Setup Instructions](https://xilinx.github.io/Vitis-AI/docs/board_setup/board_setup.html).


### On Windows

#### Prerequisites
1. Visual Studio 2019
2. cmake (version >= 3.26)
3. python (version >= 3.9) (Recommended for python 3.9.13 64bit)

#### Install ONNX Runtime

Option 1: download a prebuilt package
You may get a prebuilt onnxruntime package from [TBD]() . For example, you may download `onnxruntime-win-x64-***.zip` and unzip it to any folder. The folder you unzip it to will be your ONNXRUNTIME_ROOTDIR path.

Option 2: build from source
If you'd like to build from source, the full build instructions are [here](https://www.onnxruntime.ai/docs/build/).

Please note you need to include the "--use_vitisai --build_shared_lib" flags in your build command.

e.g. from Developer Command Prompt or Developer PowerShell for the Visual Studio version you are going to use, build ONNX Runtime at least these options:
```
.\build.bat --use_vitisai --build_shared_lib --parallel --config Release
```
If you want to use python API, you need to include the `--build_wheel` flag in your build comand.
```
.\build.bat --use_vitisai --build_shared_lib --parallel --config Release --build_wheel
```

By default the build output will go in the .\build\Windows\<config> folder, python whl file will go in the `.\build\Windows\<config>\<config>\dist\onnxruntime_vitisai--{version}-cp39-cp39-win_amd64.whl`, and "C:\Program Files\onnxruntime" will be the installation location.

You can override the installation location by specifying CMAKE_INSTALL_PREFIX via the cmake_extra_defines parameter.
e.g.
```
.\build.bat --use_vitisai --build_shared_lib --parallel --config Release --cmake_extra_defines CMAKE_INSTALL_PREFIX=D:\onnxruntime
```
Run the below command to install onnxruntime. If installing to "C:\Program Files\onnxruntime" you will need to run the command from an elevated command prompt.
```
cmake --install build\Windows\Release --config Release
```

#### Install VOE

Download the prebuilt package from [TBD](). You may downlaod `voe-{version}-win_amd64.zip` and unzip it to any folder.
The voe-{version}-win_amd64.zip file show as follows.
```
├── bin
│   └── onnxruntime_vitisai_ep.dll
├── vaip_config.json
├── version_info.txt
└── voe-{version}-cp39-cp39-win_amd64.whl
```
- `onnxruntime_vitisai_ep.dll` : The implementation of Vitis AI EP.
- `vaip_config.json` : Vitis AI EP's configuration file. When creating onnxruntime session , need to configure this file path to `config_file` provider option.
- `version_info.txt` : The version of the Vitis AI libraries.
- `voe-{version}-cp39-cp39-win_amd64.whl` : VOE python modules.

If you use C++ API,  first please copy `onnxruntime_vitisai_ep` to the onnxruntime's installation location , `C:\Program Files\onnxruntime\bin`.
```
copy "bin\onnxruntime_vitisai_ep.dll" "C:\Program Files\onnxruntime\bin"
```
second plese install VOE's python modules:
```
pip install voe-{version}-cp39-cp38-win_amd64.whl
```

If you use python API , you need install VOE and onnxruntime_vitisai python modules:
```
pip install voe-{version}-cp39-cp38-win_amd64.whl
pip install onnxruntime_vitisai-{version}-cp39-cp39-win_amd64.whl
```

## Build and Run sample

### On Windows
#### C++ sample Prerequisites
1. opencv (version=4.6.0)
It is recommended to build form source code and use static build. the default installation localtion is "{build_dir}\install" , the installation location specify to  "D:\opencv" as an example.
```
git clone https://github.com/opencv/opencv.git -b 4.6.0
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_CONFIGURATION_TYPES=Release -A x64 -T host=x64 -G 'Visual Studio 16 2019' '-DCMAKE_INSTALL_PREFIX=D:\opencv' '-DCMAKE_PREFIX_PATH=D:\opencv' -DCMAKE_BUILD_TYPE=Release -DBUILD_opencv_python2=OFF -DBUILD_opencv_python3=OFF -DBUILD_WITH_STATIC_CRT=OFF -B build -S opencv
cmake --build build --config Release
cmake --install build --config Release
```


#### Build and Run C++ sample

Download the prebuilt package from [TBD](). You may downlaod `vitis_ai_ep_cxx_sample.zip` and unzip it to any folder.
The vitis_ai_ep_cxx_sample.zip file show as follows.
```
viti_ai_ep_cxx_sample
└── resnet50
```
- `resnet50` C++ sample source code directory which contains test image and test ONNX modle.

##### Build
Open Developer Command Prompt or Developer PowerShell for the Visual Studio version you are going to use, change your current directory to `viti_ai_ep_cxx_sample`, and run the below command.

- You may omit "-DONNXRUNTIME_ROOTDIR=..." if you installed to "C:\Program Files\onnxruntime", otherwise adjust the value to match your ONNX Runtime install location.

```
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_CONFIGURATION_TYPES=Release -A x64 -T host=x64 -DCMAKE_INSTALL_PREFIX=. -DCMAKE_PREFIX_PATH=. -B build -S resnet50  -DOpenCV_DIR="D:/opencv"
```

You can open and build and install the solution using Visual Studio,
```
devenv build\resnet50.sln
```
By default the build output will go in the .\bin\resnet50.exe

##### Run

```
.\bin\resnet50_pt.exe resnet50\resnet50_pt.onnx vaip_config.json resnet50\resnet50.jpg
```


#### Run Python sample

Download the prebuilt package from [TBD](). You may downlaod `vitis_ai_ep_py_sample.zip` and unzip it to any folder.
The vitis_ai_ep_py_sample.zip file show as follows.
```
vitis_ai_ep_py_sample
├── resnet50_python
├── requirements.txt
├── test_ort.py
└── vaip_config.json
```
- `resnet50_python`  Sample source code directory which contains test images and test ONNX model.
- `requirements.txt` The python library that the sample code depends on.
- `test_ort.py` Support run any quantized ONNX model. Usage is `python test_ort.py [onnx_model]`. If the model ends normally, `OK` will be the end of console log.

Run renset50 by the follwing commands. Note: Please make sure you have installed `voe-{version}-cp39-cp39-win_amd64.whl` and  `onnxruntime_vitisai-{version}-cp39-cp39-win_amd64.whl` before running the example.

```
python -m pip install -r requirements.txt
python test_ort.py ResNet_int.onnx
```
Run it for the first time and you will see the process of modle compilation. And detection result will be the end of console log.
```
============ Top 5 labels are: ============================
n01917289 brain coral 0.9998714923858643
n01496331 electric ray, crampfish, numbfish, torpedo 7.484220986953005e-05
n02655020 puffer, pufferfish, blowfish, globefish 1.6699555999366567e-05
n09256479 coral reef 1.6699555999366567e-05
n07754684 jackfruit, jak, jack 1.0128791473107412e-05
===========================================================
```

## Sample code
If you are using C++, you can use the following example as a reference:
```C++
// ...
#include <onnxruntime_cxx_api.h>
// include other header files
// ...

auto onnx_model_path = "resnet50_pt.onnx"
Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "resnet50_pt");
auto session_options = Ort::SessionOptions();

auto options = std::unorderd_map<std::string,std::string>({});
options["config_file"] = "/etc/vaip_config.json";
// optional, eg: cache path : /tmp/my_cache/abcdefg
options["cacheDir"] = "/tmp/my_cache";
options["cacheKey"] = "abcdefg";

session_options.AppendExecutionProvider_VitisAI(options);

auto session = Ort::Session(env, onnx_model_path, session_options);

Ort::AllocatorWithDefaultOptions allocator;
//Get name & shape of graph inputs
auto input_count = session.GetInputCount();
auto input_shapes = std::vector<std::vector<int64_t>>();
auto input_names_ptr = std::vector<Ort::AllocatedStringPtr>();
auto input_names = std::vector<const char*>();
input_shapes.reserve(input_count);
input_names_ptr.reserve(input_count);
input_names.reserve(input_count);
for (size_t i = 0; i < input_count; i++) {
    input_shapes.push_back(
        session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
    auto name = session.GetInputNameAllocated(i, allocator);
    input_names.push_back(name.get());
    input_names_ptr.push_back(std::move(name));
}

// Get name & shape of graph outputs
auto output_count = session.GetOutputCount();
auto output_shapes = std::vector<std::vector<int64_t>>();
auto output_names_ptr = std::vector<Ort::AllocatedStringPtr>();
auto output_names = std::vector<const char*>();
output_shapes.reserve(output_count);
output_names_ptr.reserve(output_count);
output_names.reserve(output_count);

for (size_t i = 0; i < output_count; i++) {
    auto shape =
        session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
    output_shapes.push_back(shape);
    auto name = session.GetOutputNameAllocated(i, allocator);
    output_names.push_back(name.get());
    output_names_ptr.push_back(std::move(name));
}


// create input tensors and fillin input data
std::vector<Ort::Value> input_tensors;
input_tensors.reserve(input_count);
auto input_tensor_values = std::vector<std::vector<char>>(input_count);
// preprocess input data
// ...

Ort::MemoryInfo info =
    Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
input_tensors.push_back(Ort::Value::CreateTensor(
    info, input_tensor_values[i].data(), input_tensor_values[i].size(),
    input_shape.data(), input_shape.size(), input_type));


auto output_tensors =
        session.Run(Ort::RunOptions(), input_names.data(), input_tensors.data(),
                    input_count, output_names.data(), output_count);

// postprocess output data
// ...
```

If you are using python, you can use the following example as a reference:
```python
import onnxruntime

# Add other imports
# ...

# Load inputs and do preprocessing
# ...

# Create an inference session using the Vitis-AI execution provider

session = onnxruntime.InferenceSession(
    '[model_file].onnx',
    providers=["VitisAIExecutionProvider"],
    provider_options=[{"config_file":"/etc/vaip_config.json"}])

input_shape = session.get_inputs()[0].shape
input_name = session.get_inputs()[0].name

# Load inputs and do preprocessing by input_shape
input_data = [...]
result = session.run([], {input_name: input_data})
```
