##
# Copyright (C) 2022 Xilinx, Inc.
##  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc.
##
##  Licensed under the Apache License, Version 2.0 (the "License");
##  you may not use this file except in compliance with the License.
##  You may obtain a copy of the License at
##
##  http://www.apache.org/licenses/LICENSE-2.0
##
##  Unless required by applicable law or agreed to in writing, software
##  distributed under the License is distributed on an "AS IS" BASIS,
##  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
##  See the License for the specific language governing permissions and
##  limitations under the License.
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
