##
##  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
##  Licensed under the MIT License.
##

from voe.pattern import node, wildcard
from voe.rule_ext import Rule


class GraphLabelBaseRule(Rule):
    def action(self, **kwargs):
        pass


class ReluConv(GraphLabelBaseRule):
    """
    Match : Relu(Conv(x,w))
    """

    def pattern(self):
        X = wildcard()
        W = wildcard()
        B = wildcard()
        conv = node("Conv", X, W, [B])
        relu = node("Relu", conv)
        return relu.build(locals())

    def where(self, **_kwargs):
        return True


class Conv(GraphLabelBaseRule):
    """
    Match : Conv(x,w)
    """

    def pattern(self):
        X = wildcard()
        W = wildcard()
        B = wildcard()
        conv = node("Conv", X, W, [B])
        return conv.build(locals())

    def where(self, **_kwargs):
        return True


def rules():
    return [ReluConv(), Conv()]
