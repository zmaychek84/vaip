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

Getting Started
===============
In order to provide a consistent user experience, this tutorial is ran using :ref:`test-onnx-runner-label` with model resnet50_pt from test_onnx_runner repo.
The tutorial contains the basic building blocks to write a pass in Python.
The :ref:`pass-label` contains the general layout of a pass.

| To create a pass, a required :ref:`pattern-label`, :ref:`action-label` and optional :ref:`where-label` must be implemented.
| To read out const data from the graph and load it at computation, check out :ref:`const-label`.
| To fuse a graph, check out :ref:`fuse-label`.

.. toctree::
   pass
   pattern
   where
   action
   const
   graph_fuse
   anchor_point