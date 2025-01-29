##
##  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
##  Licensed under the MIT License.
##

from voe.pattern import node, wildcard, create_subpattern_env, update_subpattern_env


def dequant(env, prefix):
    x = wildcard()
    x_scale = wildcard()
    x_zero_point = wildcard()
    dequantizeLinear = node(
        "DequantizeLinear",
        x,
        x_scale,
        [x_zero_point],
    )
    update_subpattern_env(env, locals(), prefix)
    return dequantizeLinear


def pattern():
    env = create_subpattern_env()

    X = dequant(env, "X")
    W = dequant(env, "W")
    B = dequant(env, "B")
    conv = node("Conv", X, W, [B])
    rule = node("Relu", conv)

    quantize_y_scale4 = wildcard()
    quantize_y_zp4 = wildcard()
    y4 = node("QuantizeLinear", rule, quantize_y_scale4, [quantize_y_zp4])
    update_subpattern_env(env, locals())
    return y4.build(env)
