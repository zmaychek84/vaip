##
##  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
##  Licensed under the MIT License.
##
import voe.voe_cpp2py_export as v
from typing import List


class MetaDef(object):
    def __init__(self, vaip_pass, graph, meta_def):
        self._graph = graph
        self._meta_def = meta_def
        self._pass = vaip_pass

    def fuse(self) -> v.Node:
        """
        Create a super_layer based on the infomation from meta_def.
        Can only be called once.

        :return: The newly created node if fuse succeeded.
        :rtype: Node
        """
        if self._meta_def == None:
            raise RuntimeError("The Metadef has been fused")
        ret = self._pass.fuse(self._graph, self._meta_def)
        self._meta_def = None
        return ret

    @property
    def outputs(self) -> List[v.NodeArg]:
        """
        Return the fused node's outputs.
        Can only be called before fuse().

        :return: fused node's outputs.
        :rtype: list
        """
        if self._meta_def == None:
            raise RuntimeError("The Metadef is invalid now")
        return self._meta_def.get_outputs(self._graph)

    def set_generic_param(self, key, value):
        """
        set generic param for information used at custom op.
        Can only be called before fuse().
        """

        if self._meta_def == None:
            raise RuntimeError("The Metadef is invalid now")
        return self._meta_def.set_generic_param(key, value)
