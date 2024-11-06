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

.. _test-onnx-runner-label:

test_onnx_runner
================
| Test_onnx_runner is a tool to run your model with random generated inputs.
| So, the end user can test his/her model without the need to write preprocessing and postprocessing.
| It is installed under  ~/.local/{build_version}/bin.

| On windows, it is recommended to call the executable with run_test_onnx_runner.bat so it can avoid pathing issues.
| On Linux, several paths must be set. To keep things simple, I paste the command below.

.. code-block:: bash

    env `cat /workspace/vaip/debug_env.txt` LD_LIBRARY_PATH=/opt/xilinx/xrt/lib:/usr/lib:/usr/lib/x86_64-linux-gnu:/usr/local/lib:/home/{username}/.local/{build_version}/lib:/home/{username}/.local/{build_version}/opt/xilinx/xrt/lib  ~/.local/{build_version}/bin/test_onnx_runner {model_path}

| env cat is just setting the variable listed in debug_env.txt which contains several debugging environment variables.
| On Linux, everything is dynamic linked. So setting LD_LIBRARY_PATH is a must.
| Finally, the executable takes a model path as the bare minimum.
| The program should output "done" if the execution is finished without any error.

There are a few options that worth mentioning:
| Use -c if you want to set the config file instead of using the default path listed in the debug_env.txt.
| Use -e if you want to encrypt your model with a 256-bit key.

