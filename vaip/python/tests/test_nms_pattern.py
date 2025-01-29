##
##  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
##  Licensed under the MIT License.
##
from voe.pattern import node, wildcard


def pattern():
    pat_input = wildcard()
    pat_score = wildcard()
    pat_max_output = wildcard()
    pat_threshold = wildcard()
    pat_nms = node(
        "NonMaxSuppression", pat_input, pat_score, pat_max_output, pat_threshold
    )
    return pat_nms.build(locals())


print(pattern())
