##
##  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
##  Licensed under the MIT License.
##
from typing import Any


class _SameAs(object):
    def __init__(self, value: Any) -> None:
        self._value = value

    def value(self) -> Any:
        return self._value


def same_as(x) -> _SameAs:
    return _SameAs(x)


def isa_SameAs(obj: Any):
    return isinstance(obj, _SameAs)
