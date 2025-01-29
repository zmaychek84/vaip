##
##  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
##  Licensed under the MIT License.
##
import sys

from voe.anchor_point import NCHW2NHWC
from voe.pattern import graph_input, node, wildcard

## test case : # 20


def action(vaip_pass, graph, transpose, fix, add, left_fix, right_fix):
    # yapf: disable
    (graph.builder(vaip_pass)
     .clone_op_type(fix.node()) # fix
     .set_input_nodes([graph.builder(vaip_pass)
                       .clone_op_type(add.node()) # add
                       .set_input_nodes([graph.builder(vaip_pass)
                                         .clone_op_type(left_fix.node()) #fix
                                         .set_input_nodes([
                                             graph.builder(vaip_pass)
                                             .clone_op_type(transpose.node()) # transpose
                                             .clone_inputs(left_fix.node())
                                             .clone_attrs(transpose.node())
                                             .clone_shape(transpose.node())
                                             .clone_data_type(transpose.node())
                                             .set_anchor_point2(left_fix.node(),NCHW2NHWC)
                                             .build()])
                                         .clone_attrs(left_fix.node())
                                         .clone_shape(transpose.node())
                                         .clone_data_type(transpose.node())
                                         .set_anchor_point2(left_fix.node(),NCHW2NHWC)
                                         .build()
                                         ,
                                         graph.builder(vaip_pass)
                                         .clone_op_type(right_fix.node()) # fix
                                         .set_input_nodes([
                                             graph.builder(vaip_pass)
                                             .clone_op_type(transpose.node()) # transpose
                                             .clone_inputs(right_fix.node())
                                             .clone_attrs(transpose.node())
                                             .clone_shape(transpose.node())
                                             .clone_data_type(transpose.node())
                                             .set_anchor_point2(right_fix.node(),NCHW2NHWC)
                                             .build()])
                                         .clone_attrs(right_fix.node())
                                         .clone_shape(transpose.node())
                                         .clone_data_type(transpose.node())
                                         .set_anchor_point2(right_fix.node(),NCHW2NHWC)
                                         .build()])
                       .clone_attrs(add.node())
                       .clone_data_type(transpose.node())
                       .clone_shape(transpose.node())
                       .set_anchor_point2(add.node(), NCHW2NHWC)
                       .build()])
     .clone_attrs(fix.node())
     .set_anchor_point1(transpose.node())
     .build()
     )
    return True


def pattern():
    left_fix = node("com.xilinx:fix", graph_input())
    right_fix = node("com.xilinx:fix", graph_input())
    add = node("com.xilinx:add", left_fix, right_fix)
    fix = node("com.xilinx:fix", add)
    transpose = node("com.xilinx:transpose", fix)
    return transpose.build(locals())


def process(vaip_pass, graph):
    # yapf: disable
    return (pattern(), action)
