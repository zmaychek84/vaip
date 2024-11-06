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

.. _const-label:

Read/Load Const Data
====================
Reading/Writing const data is quite straightforward.
To read a const data in action(), const_data() can be called on a matched node like the following.

.. code-block:: python

    def action(self, w_scale, **kwargs):
        w_scale.const_data()[0]

| We recommend use a binary file to store const data as converting a float number to string may lead to precision loss.
| The file location can be fixed or unique for each model by calling self.cache_dir()

.. code-block:: python

    wts_file = self.cache_dir() + "/" + name + ".bin"
    file = open(wts_file, "wb")
    weight = np.array(input_w.const_data(), dtype=np.int8).reshape(
        input_w.shape()[0], input_w.shape()[1]
    )
    file.write(np.array(weight).tobytes())
    file.close()

| To make reading the file easier, the meta_def comes with a <str, str> map can be set on python side and get on the C++ side.
| You can save the file name like the following.

.. code-block:: python

     meta_def.set_generic_param("wts_file", str(wts_file))
