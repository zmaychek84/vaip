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

.. _pattern-label:

Pattern
===============

Multi-Node pattern
------------------

| In the :ref:`pass-label` section, a pattern of node with four inputs and op type of NonMaxSuppression.
| This is the simplest pattern possible. More complex pattern is used in the real-world.
| Let use a slightly more involved pattern as an exmaple.

.. literalinclude:: ../../vaip/python/voe/passes/merge_fix.py
   :language: python
   :pyobject: MergeFix.pattern
   :dedent:

.. graphviz::

   digraph {
        "*" -> "float2fix"
        "float2fix" -> "fix2float";
   }

From the diagram above, we can see that instead of patterning to a single node from previous example, now the pattern matches several nodes.
Each node passes its previous node as its node input.
Note: Only the nodes that are direct/indrect inputs to node being built would be used. Other nodes would be discarded.

Optional node input
-------------------

Optional argument is essential part of pattern matching. By checking https://github.com/onnx/onnx/blob/main/docs/Operators.md, we can see
there are many operations that have optional arguments, such as "RESIZE".
[] is determined to denote optional argument. Although this is normally used to denote list, but it is repurposed as such.


.. code-block:: python

    def pattern(self):
        p_input = wildcard()
        roi = wildcard()
        scale = wildcard()
        sizes = wildcard()
        resize = node("Resize", p_input, [roi], [scale], [sizes])
        return resize.build(locals())


There isn't much of difference besides added [] at pattern function.
However, be careful at action() function.
If you add optional argument as parameter to action function. It would throw key-not-found error.
So, iterating the kwargs would be the approach to detect which node input is matched.

.. code-block:: python

    def action(self, resize, **kwargs):
        inputs = []
        for key in kwargs:
            inputs.append(kwargs[key])

Subpattern
----------
| A pattern can be consisted of multiple identical components. Writing such a pattern in a single function can clumsy and error-prone.
| Thus, we introduce the concept of subpattern. Just like a wildcard() function, it can be used as a node's input.
| Take the following graph as an example, we can see many identical components.

.. graphviz::

   digraph {
        "x" -> "DequantizeLinear";
        "x_scale" -> "DequantizeLinear";
        "x_zero_point" -> "DequantizeLinear";

        "x " -> "DequantizeLinear ";
        "x_scale " -> "DequantizeLinear ";
        "x_zero_point " -> "DequantizeLinear ";

        "x  " -> "DequantizeLinear  ";
        "x_scale  " -> "DequantizeLinear  ";
        "x_zero_point  " -> "DequantizeLinear  ";

        "DequantizeLinear"-> "Conv";
        "DequantizeLinear "-> "Conv";
        "DequantizeLinear  "-> "Conv";
        "Conv"-> "Relu";
        "Relu"-> "QuantizeLinear";
        "scale"-> "QuantizeLinear";
        "quantize"-> "QuantizeLinear";
   }

| Thus, to create a subpattern of DequantizeLinear, we write the function as the following:

.. literalinclude:: ../../vaip/python/voe/generate_test_cases_pattern/p50_dp_conv_relu_q.py
   :language: python
   :pyobject: dequant
   :dedent:

| There are three differences between a subpattern and a pattern.
| First, the subpattern only return a node which is an input of another node in the pattern which we will present next.
| Second, the subpattern must somehow give the locals() information back to the pattern.
| So, update_subpattern_env() is called as such with locals().
| Finally, the subpattern must accept the env and prefix as its arguments and use these later in update_subpattern_env()

.. literalinclude:: ../../vaip/python/voe/generate_test_cases_pattern/p50_dp_conv_relu_q.py
   :language: python
   :pyobject: pattern
   :dedent:

| There are a few differences are well.
| the build is called with env instead of locals() because now we need every variables' information in the subpattern as well.
| The env has such information after calling update_subpattern_env().
| It also has to be created using create_subpattern_env().
| Each subpattern is called with a prefix so the every variable can be accessed in the action().
| To access of the s_scale with prefix "X" in action() function, add X_s_scale on the argument of action().

Multi-Consumer Node
-------------------
| The node's consumer is never specified on the pattern() function.
| So, to match for a multi-consumer Add node like following, the pattern() is identical to a single-consumer Add node.

.. graphviz::

   digraph {
        "Add" -> "MatMul";
        "Add" -> "MatMul ";
        "Add" -> "MatMul  ";
   }

| But, we need to filter out the Add node with multi-consumer using get_consumers() in where().
| Since get_consumers return a list of nodes, we can iterate each one of them to do some further filtering if wanted. 

.. code-block:: python

   def pattern(self):
      add = node("Add", wildcard())
      return add.build(locals())

   def where(self, add):
      consumer = add.get_consumers()
      is_multi_consumer = len(consumer) == 3
      if not is_multi_consumer:
         return False
      for n in consumer:
         if n.op_type() != "MatMul":
            return False
      return True

If a graph fuse is needed for this pattern, the input should be Add node. the output should be the consumers. Like the following:

.. code-block:: python

   def action(self, add):
      inputs = [add]
      outputs = add.get_consumers()
      meta_def = self.try_fuse("node name", inputs, outputs, [], "device name")
