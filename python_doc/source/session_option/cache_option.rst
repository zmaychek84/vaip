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

.. _cache-option-label:

Cache Option
============

| To avoid recompiling the onnx model, the compiled models are saved under :ref:`cache directory <cache_dir_api-label>`.
| Next time when the session is created, the cache hit would allow us to skip compilation and go straight to computation.
| But, the default path is not always desirable when we want to deploy the model.
| So, we could specify a model's cache directory in the session option like the following:

.. code-block:: python

    session = onnxruntime.InferenceSession(
        '[model_file].onnx',
        providers=["VitisAIExecutionProvider"],
        provider_options=[{
            "config_file":"xxx",
            "cacheKey": "3f97582a85b8ae1be41127f326bbf9b2",
            "cacheDir": "/some_path"
    }])

Note: the cache directory is now set at the directory {cacheDir + cacheKey}.