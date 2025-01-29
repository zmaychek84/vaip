##
##  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
##  Licensed under the MIT License.
##

import sys

import glog as log
import numpy as np
from voe.anchor_point import FLOAT2FIX, scale_to_fixpoint
from voe.pattern import node, wildcard
from voe.rule_ext import Rule, same_as

"""
  test case model 112

  merge mul pass
  From : FixNeuron(input)
  To  : DequantizeLinear(QuantizeLinear(input))
"""


class ConvertFixNeuron(Rule):
    def pattern(self):
        input = wildcard()
        scale = wildcard()
        zero_point = wildcard()
        fixneuron = node("ai.onnx.contrib:FixNeuron", input, scale, zero_point)
        return fixneuron.build(locals())

    def action(self, input, scale, zero_point, fixneuron, **_others):
        quantizelinear = self.create_node(
            op_type="ai.onnx:QuantizeLinear",
            inputs=[input, scale, zero_point],
            data_type="int8",
            shape=same_as(fixneuron),
            anchor_point=(
                fixneuron,
                FLOAT2FIX(scale_to_fixpoint(scale.const_data()[0])),
            ),
        )
        return self.create_node(
            op_type="ai.onnx:DequantizeLinear",
            inputs=[quantizelinear, scale, zero_point],
            data_type=same_as(fixneuron),
            shape=same_as(fixneuron),
            anchor_point=fixneuron,
        )


def rules():
    return [ConvertFixNeuron()]
