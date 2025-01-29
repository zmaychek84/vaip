##
##  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
##  Licensed under the MIT License.
##
import sys

import glog as log
import numpy as np
from voe.anchor_point import CONST
from voe.pattern import node, wildcard, xir_const_op
from voe.rule_ext import Rule, same_as


class fuse_NMS(Rule):
    def pattern(self):
        p_input = wildcard()
        score = wildcard()
        max_output = wildcard()
        threshold = wildcard()
        nms = node("NonMaxSuppression", p_input, score, max_output, threshold)
        return nms.build(locals())

    def action(self, score, p_input, nms, **kwargs):
        inputs = [p_input, score]
        outputs = [nms]
        meta_def = self.try_fuse("NMS", inputs, outputs, [], "NMS")
        return meta_def.fuse()


def rules():
    return [fuse_NMS()]
