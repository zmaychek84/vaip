<!--
    Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
    Licensed under the MIT License.
 -->

# VAIP


Vitis AI partitioner.


## Getting start


#### prepare dev environment

##### Linux
All development environments of VAIP are integrated into a docker container.
Access the container via xdock-vitis-ai-sw
clone xdock-vitis-ai-sw repo.
```shell
# cd to your workspace
cd xdock-vitis-ai-sw
make run
```
After running the above command, you are in the docker terminal.

##### Windows
Developers need to install the dependent environment in advance as follows.
1. Visual Studio 2019, with individual componment "spectre" installed,
2. cmake (version >= 3.26),
3. python (version == 3.9), 3.11 would cause some packaging issues,
4. XRT,
   1. Download the zip(either debug or release, depends on your build)
   2. Extract it and copy the xrt folder inside to `~/.local/{prefix}.{build_type}/`
      1. `{prefix}` is the output of `python -c 'import platform ; print(\".\".join([platform.system(),platform.version(),platform.machine()]))'`
      2. `{build_type}` is Debug or Release, depends on your build
   3. The directory after successful installation is as follows.
      ```
      dir ~\.local\Windows.10.0.14393.AMD64.Release\

          Directory: ~\.local\Windows.10.0.14393.AMD64.Release

      Mode                LastWriteTime         Length Name
      ----                -------------         ------ ----
      d-----        5/18/2023  11:48 AM                xrt
      ```

#### compile
Use vai-rt to build vaip and dependencies.
clone vai-rt repo.
```shell
cd vai-rt
python main.py --type=release --dev-mode
```
The script will automate the installation of all dependencies.
`--type=debug` means that the compiler type is the debug.
`--dev-mode` forces compilation to use the latest code,your local change will be coverage.

#### How to run an onnx model
```shell
test_onnx_runner onnx/model/path/onnx.onnx
```


#### pull large file
There are some pull processes for large files that need to be done manually.
```shell
cd /workspace/vaip
git lfs fetch
git lfs pull
```

If you don't have git-lfs installed, follow these steps to install it:
```
wget  https://github.com/git-lfs/git-lfs/releases/download/v3.3.0/git-lfs-linux-amd64-v3.3.0.tar.gz
tar -zxvf git-lfs-linux-amd64-v3.3.0.tar.gz
cd git-lfs-3.3.0
sudo ./install.sh
git lfs install
```
## Cross compile
For cross compilation, can refer to [onnxruntime_on_board_test](./doc/onnxruntime_on_board_test.md).



# tips for developpers.

put the following line in your `~/.bashrc` might make many command shorter.

on Linux.

```bash
    export PREFIX=$HOME/.local/Ubuntu.20.04.x86_64.Debug
    export BUILD=$HOME/build/build.Ubuntu.20.04.x86_64.Debug
    export W=/workspace
```

for `git-bash` on Windows.

```bash
export W=$HOME/workspace
export BUILD=$HOME/build/build.Windows.10.0.17763.AMD64.Debug
export PREFIX=$HOME/.local/Windows.10.0.17763.AMD64.Debug
```

create a shell command  `build` in one of your $PATH, for example, `$HOME/.local/bin`.

```bash
#!/bin/bash
if [ -z "$1" ]; then
    project=vaip
    # Default logic or value goes here
else
    project=$1
fi
echo "build project $project"
export BUILD=$HOME/build/build.Ubuntu.20.04.x86_64.Debug
# on windows
# export BUILD=$HOME/build/build.Windows.10.0.17763.AMD64.Debug
cmake --build $BUILD/$project -j $(nproc) && cmake --install $BUILD/$project
```

Then for example, you can run `build vaip` quickly to build the vaip project.

## to create a new pass

```
python vaip/python/voe/tools/create_cxx_pass.py --name dd change op dtype --enable
python vaip/python/voe/tools/create_cxx_pass.py --help # for more details
```

## convert an onnx model to text format

Sometime, for debugging purpose, it is very useful to view the onnx model in the text format

```
$BUILD/vaip/onnxruntime_vitisai_ep/onnx_dump_txt -i a.onnx -o a.txt
$BUILD/vaip/onnxruntime_vitisai_ep/onnx_dump_txt -h # for more details
```

## generate a new pattern


```
$BUILD/vaip/onnxruntime_vitisai_ep/onnx_pattern_gen env \
 IGNORE_CONSTANT=1 \
 ENABLE_CONSTNAT_SHARING=0 \
 $BUILD/vaip/onnxruntime_vitisai_ep/onnx_pattern_gen \
 -i value/Add_output_0_QuantizeLinear_Output \
 -i key/MatMul_output_0_QuantizeLinear_Output \
 -i query/Add_output_0_QuantizeLinear_Output \
 -i Mul_output_0_convert_QuantizeLinear_Output \
 -o Reshape_4_output_0_QuantizeLinear_Output \
 -f vaip/.cache/acd89c9415eba62a3623a3af2e7e8227/onnx.onnx\
 -c ../../vaip_pattern_zoo/src/QMHAGRPB_0.h.inc
 -m ../../vaip_pattern_zoo/src/QMHAGRPB_0.mmd
 ```

 1. `IGNORE_CONSTANT` when it is 0, constant initializers are not
    shown in the generated mermaid diagram. Usually it makes diagrams
    cleaner.
 2. `ENABLE_CONSTNAT_SHARING=0` when it is 0, the generated pattern
    does not try to share a common constant initializer, which make
    the generated pattern potentially match wider range of nodes. If
    `ENABLE_CONSTNAT_SHARING=1`, the generated pattern is stricter to
    match a certain of subgraph which also share these constants. It
    does not match subgraphs which do not share constant initializers.
 3. `-i` specify subgraph input, if there are more than one inputs, we
    need to set multiple `-i` options.
 3. `-o` the subgraph output, only a single output is allowed.
 4. `-f` the sample onnx model as a template to genrate a pattern.
 5. `-c` the source file for generated c++ code.
 5. `-m` the file name for generated mermaid diagram.
 6. `-h` for see sample usage.


##  Extract subgraph from the ONNX model

Sometimes, the model is very large.  For debugging purpose, it is very useful to extract subgraph as unit test.
The "onnx_knife" tool can extracts a subgraph from an ONNX model based on specified input and output nodes.
Support multiple input and single output.
The tool takes command-line arguments to specify the input ONNX model file, the output path for the subgraph, and the input and output node arg names.

```
$BUILD/vaip/onnxruntime_vitisai_ep/onnx_knife -i 111 -o 118 -I pt_resnet50.onnx -O subgraph.onnx
$BUILD/vaip/onnxruntime_vitisai_ep/onnx_knife -h # for more detail
```
