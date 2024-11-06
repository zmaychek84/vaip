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

VAIP Internal Design Doc
========================

Implement ONNX Runtime Execution Provider
-----------------------------------------

The onnxruntime ExecutionProvider interface is defined at 
`ORT ExecutionProviders`_

..  code-block:: cpp

    class ExecutionProviders {
        ...
        // step 1. partition
        virtual std::vector<std::unique_ptr<ComputeCapability>>
        GetCapability(const onnxruntime::GraphViewer& graph_viewer,
                const IKernelLookup& kernel_lookup) const;
        // step 2. real computation
        virtual common::Status Compile(const std::vector<FusedNodeAndGraph>& fused_nodes_and_graphs,
                                 std::vector<NodeComputeInfo>& node_compute_funcs);
        ...
    }

.. _ORT ExecutionProviders: https://github.com/microsoft/onnxruntime/blob/b7ae293be05c89a0cb623feec4d2d2cbf006e4c3/include/onnxruntime/core/framework/execution_provider.h#L60

VAIP plugin does not need to implement the above to APIs, they can simply implement 

Graph Partition
---------------

`GetCapability` returns a bunch of `ComputeCapability` objects and each object represents a single subgraph.

In order to cache the partition result in JSON, we translate it into protobuf defination as blow. 
Protobuf API supports serialize and deserialize a protobuf message to and from a JSON object.

.. literalinclude:: ../../vaip/src/capability.proto
    :start-at: message MetaDefProto 
    :language: protobuf
    :linenos:

#. `id` any string, it must be unique name for the corresponding subgraph. 
#. `inputs` is a list of input names, its element must be a graph input, a graph constant initializer or an output of a node. refer to `ONNX IR Node`_
#. `outputs` is a list of output names, its element must be a output of a node.
#. `nodes` is a list of nodes, its element must be a output of a node, when a node has multiple output, the first output is used.
#. `constant_initializers` is list of graph constant initializers.
#. `device`, this is an extension field used by VAIP, this field is used to find the custom op impemention plugin, see <TODO>


.. _ONNX IR Node:  https://github.com/onnx/onnx/blob/main/docs/IR.md#nodes


see this is a sample 
:cpp:class:`vaip_core::AnchorPoint`

.. doxygenclass:: vaip_core::AnchorPoint
    :members:
