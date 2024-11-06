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

.. _fuse-label:

Fuse
====
| Previous, we briefly mentioned an example on graph fusing at :ref:`pass <simple-fuse-label>`.
| To see the API description, check :ref:`try_fuse API <fuse-api-label>`.

try_fuse()
----------

| try_fuse() is called to check if a graph fusing on the inputs and outputs inside action() is possible.
| The algorithm would traverses from the outputs all the way to the inputs to detect if there is a leakage.
| A leakage is defined as a node that the an output of the fused node depends but not lists as an input.
| If a leakage occurs, it returns None to indicate that such a fuse is impossible. Otherwise, it returns a meta def.
| Please note the meta_def's name must be unique.

fuse()
-------
| It fuses the nodes between inputs and outputs inclusively into a new node named super_layer.
| Please note action() takes a boolean as return to indicate if a graph transformation is done in this matched pattern.
| So please return the return value of action().
