#
#   The Xilinx Vitis AI Vaip in this distribution are provided under the following free
#   and permissive binary-only license, but are not provided in source code form.  While the following free
#   and permissive license is similar to the BSD open source license, it is NOT the BSD open source license
#   nor other OSI-approved open source license.
#
#    Copyright (C) 2022 Xilinx, Inc. All rights reserved.
#    Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
#
#    Redistribution and use in binary form only, without modification, is permitted provided that the following conditions are met:
#
#    1. Redistributions must reproduce the above copyright notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the distribution.
#
#    2. The name of Xilinx, Inc. may not be used to endorse or promote products redistributed with this software without specific
#    prior written permission.
#
#    THIS SOFTWARE IS PROVIDED BY XILINX, INC. "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
#    THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL XILINX, INC.
#    BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
#    PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
#    OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE
#


import logging
import os
import platform
from pathlib import Path
import pathlib
from . import shell
from .imp import workspace as _impl


class Workspace(object):
    def __init__(self, filename=None):
        super().__init__()
        self._environ = _impl.init_environ(self)
        if not filename is None:
            self.update_environment_from_file(filename)
        self._recipe_json = _impl.load_recipe_json(self)
        self._ref_json = _impl.load_ref_json(self)

    @staticmethod
    def is_windows():
        return _impl.is_windows()

    @staticmethod
    def user():
        if Workspace.is_windows():
            return os.environ["USERNAME"]
        else:
            return os.environ["USER"]

    @staticmethod
    def home():
        workspace_home = (
            pathlib.Path.home()
            if not os.environ.get("JOB_BASE_NAME", "")
            else Path(r"C:\\IPU_workspace") / os.environ["JOB_BASE_NAME"]
        )
        if platform.system() == "Linux":
            workspace_home = Path("/home") / os.environ["USER"]
        return workspace_home

    def workspace(self):
        return (
            self.home() / "workspace"
            if not os.environ.get("JOB_BASE_NAME", "")
            else self.home() / "vaip_regression"
        )

    def build_dir(self):
        return (
            self.home() / "build"
            if not os.environ.get("JOB_BASE_NAME", "")
            else self.home()
        )

    def build_type(self):
        return os.environ.get("BUILD_TYPE", "Debug")

    def install_prefix(self):
        return _impl.install_prefix(self)

    def environ(self):
        """return environement variable for running commands"""
        return self._environ

    def update_environment_from_file(self, filename):
        return _impl.update_environment_from_file(self, filename)

    def update_environment_from_dict(self, var_dict):
        return _impl.update_environment_from_dict(self, var_dict)

    def save_workspace(self):
        _impl.save_workspace(self)

    @staticmethod
    def load_json(file):
        return _impl._load_json(file)
