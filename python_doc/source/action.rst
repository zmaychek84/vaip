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

.. _action-label:

Action
===============
| Action is called after pattern() and where() if it where() is undefined or retured true.
| Usually,  :ref:`graph fusing <fuse-label>` and :ref:`data IO <const-label>` are done at this function.
| It returns a boolean to indicate if the graph transformation is done for the pattern matched.
| For now, anything other than false and None is casted into true.
| If true is returned, the graph would try to call Resolve() to remove unconnected nodes.