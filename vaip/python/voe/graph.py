##
##  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
##  Licensed under the MIT License.
##

from .node import *
from typing import List
import voe.voe_cpp2py_export as v


class Graph(object):
    def __init__(self, cpp_graph: v.GraphWrapper) -> None:
        self._graph = cpp_graph

    def resolve(self, is_force: bool) -> None:
        self._graph.resolve(is_force)

    def get_node_in_topoligical_order(self) -> List[int]:
        return self._graph.get_node_in_topoligical_order()

    def get_node(self, index: int) -> Node:
        return Node(self._graph, self._graph.get_node(index))
