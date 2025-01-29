##
##  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
##  Licensed under the MIT License.
##
import sys


def action(graph, transpose):
    print(str(abc))
    print(str(abc.node()))
    print(str(abc.node_arg()))
    # builder = graph.builder()
    # builder.set_input_args([i.node_arg()]).set_op_type(
    #     "relu", "com.xilinx").set_anchor_point1(abc.node()).build()
    return False


# ("com.xinlinx:transpose",
#                ("com.xilinx:fix",
#                 ("com.xilinx:add",
#                  ("com.xilinx:fix", "input"),
#                  ("com.xilinx:fix", "input"))))


def pattern():
    fix = node("com.xilinx:fix", graph_input())
    transpose = node("com.xilinx:transpose", fix)
    return transpose.build(locals())


def process(pass_obj, graph):
    return (pattern(), action)
