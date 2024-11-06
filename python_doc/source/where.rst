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

.. _where-label:

Where
===============

| where() is used to describe relationship other than topological relationship.
| For example, if we only need to act on a node only if the node's input has 4 dimensions, we would return false on where() when this condition is not met.  
| It is optional. If it is undefined, the pass would just skip this function.
| In general, it should have the same arguments as the action().

.. literalinclude:: ../../vaip/python/voe/passes/merge_mul.py
   :language: python
   :pyobject: MergeMul.where
   :dedent:

| In this example, the first Mul node is checked for if the consumers(how many nodes take it as a input) are greater than 1.
| Then two consts' must have an identical shape.
| Otherwise, the action() would not be called.