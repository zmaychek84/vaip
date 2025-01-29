##
##  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
##  Licensed under the MIT License.
##
import sys

from voe.pattern import node, wildcard
from voe.rule_ext import Rule, same_as
from voe.anchor_point import CAST


class ConvertSoftmaxRule(Rule):
    def pattern(self):
        sfm_in = wildcard()
        softmax = node("com.xilinx:softmax", sfm_in)
        return softmax.build(locals())

    def action(self, sfm_in, softmax, **_others):
        attrs = {
            "axis": softmax.attr("axis"),
            "type": "poly",
        }
        cast_1 = self.create_node(
            op_type="com.xilinx:cast",
            inputs=[sfm_in],
            data_type="bfloat16",
            shape=softmax.shape(),
            anchor_point=(sfm_in, CAST),
        )
        hsoftmax = self.create_node(
            op_type="com.xilinx:hard_softmax",
            inputs=[cast_1],
            attrs=attrs,
            data_type="bfloat16",
            shape=softmax.shape(),
            anchor_point=(sfm_in, CAST),
        )
        cast_2 = self.create_node(
            op_type="com.xilinx:cast",
            inputs=[hsoftmax],
            data_type="float32",
            shape=softmax.shape(),
            anchor_point=softmax,
        )
        return cast_2


def rules():
    return [ConvertSoftmaxRule()]
