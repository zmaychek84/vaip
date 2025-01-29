##
##  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
##  Licensed under the MIT License.
##
from voe.pattern import graph_input, is_pattern, node, wildcard


def pattern():
    input = graph_input()
    weight = wildcard()
    conv = node("Conv", input, weight, wildcard())
    return conv.build(locals())


pat = pattern()
print(pattern())
print(f"is pattern?{is_pattern(pat)}")
