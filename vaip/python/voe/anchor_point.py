##
##  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
##  Licensed under the MIT License.
##
import json
import math


class _AnchorPointTag(str):
    pass


def transpose(order: _AnchorPointTag) -> _AnchorPointTag:
    return _AnchorPointTag(
        json.dumps(
            {"opType": "transpose", "attribute": {"transposeAttr": {"order": order}}},
            sort_keys=True,
            indent=4,
        )
    )


def fix(type: str, fixpoint: str) -> _AnchorPointTag:
    return _AnchorPointTag(
        json.dumps(
            {"opType": type, "attribute": {"fixAttr": {"fixPoint": fixpoint}}},
            sort_keys=True,
            indent=4,
        )
    )


def cast() -> _AnchorPointTag:
    return _AnchorPointTag(
        json.dumps(
            {"opType": "cast"},
            sort_keys=True,
            indent=4,
        )
    )


def const() -> _AnchorPointTag:
    return _AnchorPointTag(json.dumps({"opType": "const"}, sort_keys=True, indent=4))


def FLOAT2FIX(fixpoint: int) -> _AnchorPointTag:
    return fix("float2fix", fixpoint)


def FIX(fixpoint: int) -> _AnchorPointTag:
    return fix("fix", fixpoint)


def FIX2FLOAT(fixpoint: int) -> _AnchorPointTag:
    return fix("fix2float", fixpoint)


def is_anchor_point(obj) -> bool:
    return isinstance(obj, _AnchorPointTag)


def scale_to_fixpoint(scale) -> int:
    return int(math.log(scale, 1 / 2))


NCHW2NHWC = transpose([0, 2, 3, 1])
NCCHW2NHWCC = transpose([0, 3, 4, 1, 2])
CONST = const()
CAST = cast()
