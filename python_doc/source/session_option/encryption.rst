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

Encryption
==========

Model Encryption
----------------

To protect customers’ intellectual property, encryption is supported as a session option.
With this enabled, all the xir and compiled models generated would be encrypted using AES256 algorithm.

To enable encryption, you need to pass in the encryption key like following:

.. code-block:: python

    session = onnxruntime.InferenceSession(
        '[model_file].onnx',
        providers=["VitisAIExecutionProvider"],
        provider_options=[{
            "config_file":"xxx",
            "encryptionKey": "89703f950ed9f738d956f6769d7e45a385d3c988ca753838b5afbc569ebf35b2"
    }])

| The key is 256-bit which is represented as a 64-digit string.
| The cached models under :ref:`cache directory <cache_dir_api-label>` are now unabled to be opened with Netron.
| There is a side effect as well, dumping would be disabled as dumping would leak out much information about the model.

It is also supported as a built-in -e option at :ref:`test onnx runner <test-onnx-runner-label>`.



Deployment
----------


| To deploy the model elsewhere else, the onnx model must be passed in when creating a session as well.
| One way to hide the original model is to decrypt the onnx model into memory and pass the data pointer like the following:

.. code-block:: python

    model_byte = your_decrypt_func('[encrypted_model_file].onnx', your_decrypt_key)
    session = onnxruntime.InferenceSession(
        model_byte,
        providers=["VitisAIExecutionProvider"],
        provider_options=[{
            "config_file":"xxx",
            "encryptionKey": "89703f950ed9f738d956f6769d7e45a385d3c988ca753838b5afbc569ebf35b2"
    }])


| To avoid recompile the onnx model at deployed environment, see :ref:`cache option <cache-option-label>`.