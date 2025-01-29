##
##  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
##  Licensed under the MIT License.
##

import voe.voe_cpp2py_export as v


class Node(object):
    def __init__(self, graph: v.GraphWrapper, node: v.Node) -> None:
        self._graph = graph
        self._node = node

    def __str__(self) -> str:
        return self._node.__str__()
