##
##  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
##  Licensed under the MIT License.
##
def pattern():
    input = wildcard()
    weight = wildcard()
    conv = node("Conv", input, weight, wildcard())
    return conv.build(locals())
