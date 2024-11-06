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

.. _pass-label:

Pass
====
| A pass is the procedure call on the graph that can consist of multiple rules.
| Each rule would try to find the node that matches with the description from pattern().
| This description specifies a node’s inputs and possibly its inputs’ inputs.
| The node matched is then filtered by an optional where().
| Usually, the user would filter out some nodes with unwanted constant data within the node in this step.
| Afterwards, the remaining matched node would then be transformed or read for internal data at action().

Whether a pass would be called is defined in a JSON file with path controlled by environment variable, VITISAI_EP_JSON_CONFIG.
Normally, this variable is defined as vaip_config.json.
To understand how to create or modify a pass, fuse_NMS would be taken as an example for this tutorial.
In the vaip_config.json, the fuse_NMS is defined as following:

.. literalinclude:: ../../vaip/etc/vaip_config.json
   :language: JSON
   :lines: 8-14

| "name" is the name of the pass. It can be aribitrary.
| "plugin" should always be "vaip-pass_py_ext" if the pass is implemented in Python.
| "disable" line should be removed if the pass is enabled.
| "pyExt" is required for a Python pass.
| "moduleName" should match the file path of your Python script.
| "methodName" is the entry point of your Python script.

.. literalinclude:: ../../vaip/python/voe/passes/fuse_NMS.py
   :language: python
   :pyobject: rules
   :dedent:

Here the function name matched methodName defined in JSON file.
It should return the list of classes needed to be created for this pass.

.. literalinclude:: ../../vaip/python/voe/passes/fuse_NMS.py
   :language: python
   :pyobject: fuse_NMS.pattern
   :dedent:

| node() creates a pattern that matchs a node with four node inputs and an optype of NonMaxSuppression.
| locals() would return all local variables as dictionary.
| So, nms.build(locals()) is equvalent to nms.build({"p_input": p_input, "score": score, ...}).
| In the end, the function returns a JSON string representation of the pattern created by nms.build().
| Note: The pattern would be matched against the transformed graph output by previous pass.
| Note2: The onnxruntime would transform the graph before any passes. So, checkout the intermediate model saved under :ref:`cache directory <cache_dir_api-label>`.
| Note3: This part can be debugged using :ref:`onnx_grep-label`.

.. _simple-fuse-label:

.. literalinclude:: ../../vaip/python/voe/passes/fuse_NMS.py
   :language: python
   :pyobject: fuse_NMS.action
   :dedent:

| The action function acts on the matched pattern.
| Each matched node or node input is passed as key value pair. So, the parameter name should match the keys passed at nms.build().
| This example showed how to fuse NonMaxSuppression into a super_layer.
| An action should return a bool to indicate if the graph is modified. Anything other than False and None are casted into True on the C++ side.
| In general, an action can fuse a graph, create a new node and record const data in meta_def on matched pattern.
| How to implement these would be explained in next few sections.
