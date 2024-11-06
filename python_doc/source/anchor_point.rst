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


.. _anchor_point-label:

Anchor Point
===============
| A node can be ordered, replaced and created during each graph transformation.
| To preserve the invariant property of each graph transformation, the input and output of a subgraph should be identical before and after each transformation.
| Take a concrete example, if the custom ops are implemented using NCWH layout. But the inputs and outputs of the original graph is using NCHW layout,
| the subgraph should be transformed from NCWH back NCHW so the onnxruntime can continue with the output that has the correct layout.
| 
| Thus, anchor point is used to record such connection such as tranposing, converting data in the new graph.
| It can also be thought as a way the trace the origin of a created node which we will use during computation and debugging.
|
| We will use a few examples to demonstrate how to use set_anchor_point.
| Note calling create_node would indirectly calls set_anchor_point as well.
| For more information, see :ref:`builder-label`.

The first example is to combining a float2fix and fix2float into a new subgraph.

.. graphviz::

   digraph {
        "Input" -> "float2fix"
        "float2fix" -> "fix2float";
   }

To preserve the invariant property, the output should be as same as the fix2float's output.
So, we should set the anchor point to fix2float.

.. literalinclude:: ../../vaip/python/voe/passes/merge_fix.py
   :language: python
   :pyobject: MergeFix.action
   :dedent:

| This is an extremely simple case. The output of newly created node would replace the output of last node(fix2float) in previous graph.
| Here is a slightly complicated case.
| Suppose, we want to create a new graph that moves the transpose to the bottom from the following graph.
| So, it is easier to combine two transposes into one transpose in the following passes.

.. graphviz::

   digraph {
        "Input" -> "Transpose"
        "Transpose" -> "ReLU";
   }

Obviously, we need to reset the anchor point for every node. The following showed how to do it with anchor_point.


.. code-block:: python

   class swap_ReLU_transpose(Rule):
      def pattern(self):
         input = wildcard()
         transpose = node("Transpose", input)
         relu = node("Relu", transpose)
         return relu.build(locals())

    def action(self, input, relu, transpose, **kwargs):
        order = t.attr("perm")
        shape = convert_shape(input.shape(), order)
        new_relu = self.create_node("Relu", attrs=same_as(relu), data_type=same_as(relu), inputs=[input], anchor_point=(relu, transpose(shape)))
        new_transpose = self.create_node("Transpose", attrs=same_as(transpose), data_type=same_as(transpose), inputs=[new_relu], anchor_point=relu)
        return new_transpose

.. graphviz::

   digraph {
        "Input" -> "ReLU"
        "ReLU" -> "Transpose";
   }

| The newly created transpose is the output of the graph. So it should replace the output of last node(ReLU) in the previous graph.
| It also takes the output of newly created ReLU, hence the inputs are set to equal to [input].
| This is similar to last example.
| The interesting point is setting the anchor point of newly created ReLU which we will refer as new_relu.
| What is relationship between new_relu and the ReLU it derived from?
| The output of the ReLU is tranposed where the new_relu is not tranposed yet!
| So by calling set_anchor_point with a tuple of an argument we can described the relationship as relu + a tranpose = new_relu.


