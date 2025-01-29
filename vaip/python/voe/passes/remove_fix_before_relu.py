##
##  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
##  Licensed under the MIT License.
##

from voe.pattern import node, wildcard
from voe.rule_ext import Node, Rule, same_as

"""
  test case model enc_0423_4096x2160

  remove fix before relu
  From : relu(fix(input))
  To  : relu(input)
"""


class RemoveFixBeforeRelu(Rule):
    def pattern(self):
        before_fix = node("com.xilinx:fix", wildcard())
        relu = node("com.xilinx:relu", before_fix)
        return relu.build(locals())

    def where(self, before_fix: Node, relu: Node, **kwargs):
        consumers = relu.get_consumers()
        for c in consumers:
            if "fix" != c.op_type() or before_fix.attr("fix_point") != c.attr(
                "fix_point"
            ):
                return False
        return True

    def action(self, relu: Node, before_fix: Node, **kwargs):
        return self.create_node(
            op_type=same_as(relu),
            inputs=same_as(before_fix),
            attrs=same_as(relu),
            anchor_point=relu,
        )


def rules():
    return [RemoveFixBeforeRelu()]
