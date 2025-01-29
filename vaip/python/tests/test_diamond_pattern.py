##
##  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
##  Licensed under the MIT License.
##
from voe.pattern import is_pattern, node, wildcard


def pattern():
    input = wildcard()
    sin = node("Sin", input)
    cos = node("Cos", input)
    add = node("Add", sin, cos)
    return add.build(locals())


pat = pattern()
print(pattern())
print(f"is pattern?{is_pattern(pat)}")
