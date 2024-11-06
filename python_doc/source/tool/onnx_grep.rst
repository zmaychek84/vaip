.. 
   Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc.
   
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

.. _onnx_grep-label:

onnx_grep
=========
| Onnx_grep is a simple tool to debug and test your pattern matching.
| It ususally installed under ~/.local/{build_version}/bin/vaip.
| There is also a onnx_grep.py in the src folder that do the exact same thing that is written in Python.
| It takes three mandatory arguments which are model file, pattern file and config file.

For JSON pattern, use the following command to execute:

.. code-block:: bash

    ./onnx_grep -f /workspace/test_onnx_runner/data/pt_resnet50.onnx -p /workspace/vaip/onnxruntime_vitisai_ep/test/pattern1.json /workspace/vaip/vaip/etc/vaip_config.json

For python pattern, use the following command to execute:

.. code-block:: bash

    ./onnx_grep -f /workspace/test_onnx_runner/data/pt_resnet50.onnx -p /workspace/vaip/onnxruntime_vitisai_ep/test/pattern1.py /workspace/vaip//vaip/etc/vaip_config.json

| The files listed above do exists assuming you cloned all repositories under /workspace directory.
| To learn how to write a python pattern, please checkout :ref:`pass-label` and :ref:`pattern-label`.
| There is a subtle difference between production code and testing code where the python pattern your provide to onnx_grep must define a non-member function called pattern() as the entry point.
| The model file may not be identical with the graph in production becasue the graph maybe transformed before it reaches current pass.
| In addtion, onnxruntime would transform the graph even before the first pass. So, it is recommended to test it against the .onnx files generated under the cache directory.

The output string is if the pattern matched against each single node within the model provided.
Here are the example output if the node is matched/unmatched.

.. code-block:: bash

    I20230831 06:10:24.999087 55896 onnx_grep.cpp:103] find node: @431 [1188:(ty=1,shape=[1,512,7,7])] Conv [1175:(ty=1,shape=[1,512,7,7]),1181:(ty=1,shape=[512,512,3,3]),1187:(ty=1,shape=[512])]
    I20230831 06:10:24.999179 55896 onnx_grep.cpp:105] node is not match @432 [1189:(ty=1,shape=[1,512,7,7])] Relu [1188:(ty=1,shape=[1,512,7,7])]
    I20230831 06:10:24.999248 55896 onnx_grep.cpp:105] node is not match @433 [1192:(ty=3,shape=[1,512,7,7])] QuantizeLinear [1189:(ty=1,shape=[1,512,7,7]),1190:(ty=1,shape=[]),1191:(ty=3,shape=[])]
