##
##  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
##  Licensed under the MIT License.
##
import sys

from voe.anchor_point import NCHW2NHWC
from voe.pattern import graph_input, node, wildcard


# testcase 100
def action(vaip_pass, graph, transpose, fix):
    # yapf: disable
    (graph.builder(vaip_pass)
     .clone_op_type(fix.node()) # fix
     .set_input_nodes([graph.builder(vaip_pass)
                       .clone_op_type(transpose.node()) # transpose
                       .clone_inputs(fix.node())
                       .clone_attrs(transpose.node())
                       .clone_data_type(transpose.node())
                       .clone_shape(transpose.node())
                       .set_anchor_point2(fix.node(), NCHW2NHWC)
                       .build()])
     .clone_attrs(fix.node())
     .set_anchor_point1(transpose.node())
     .build()
     )
    return True


def pattern():
    fix = node("com.xilinx:fix", graph_input())
    transpose = node("com.xilinx:transpose", fix)
    return transpose.build(locals())


def process(vaip_pass, graph):
    # yapf: disable
    return (pattern(), action)
