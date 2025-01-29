##
##  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
##  Licensed under the MIT License.
##
import voe.voe_cpp2py_export as v
from .graph import *


class Model(object):
    def __init__(self, filename: str) -> None:
        self._model = v.model_load(filename)

    def get_main_graph(self) -> Graph:
        return Graph(self._model.get_main_graph())
