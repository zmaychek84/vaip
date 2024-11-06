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

.. _customized-option-label:

Customized Option
=================

| To support more plugins, we added the support for customized session options.
| For example, if we pass in "foo":"bar", we can check and read the value inside a pass like the following:
| Please note the session options can only be accessed after pattern().
| See :ref:`session option API <sesion_option_api-label>` for details.

.. code-block:: python

    session = onnxruntime.InferenceSession(
        '[model_file].onnx',
        providers=["VitisAIExecutionProvider"],
        provider_options=[{
            "foo":"bar",
    }])

    class fuse_foo(Rule):
        def action(self, **kwargs):
            inputs = [...]
            outputs = [...]
            if self.has_session_option("foo"):
                device = self.get_session_option("foo")
                meta_def = self.try_fuse(device, inputs, outputs, [], device)
                return meta_def.fuse()
