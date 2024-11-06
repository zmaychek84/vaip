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


import hashlib
import json
import logging
import os
import pathlib
import subprocess
import sys
import time
import traceback

from . import shell, workspace
from .imp import recipe as _impl


class Recipe(object):
    def __init__(self, workspace):
        super().__init__()
        self._workspace = workspace
        self._cwd = pathlib.Path(os.getcwd())
        self._dry_run = False
        self._log_file = self.build_dir() / ("build.log")

    def name(self):
        """unique name of this recipe"""
        return ""

    def build_dir(self):
        """default build directory for regression test"""
        return self._workspace.build_dir() / self.name()

    def src_dir(self):
        """default source code directory for regression test"""
        return self._workspace.workspace() / self.name()

    def log_file(self):
        return self._log_file

    def report(self):
        return _impl.report(self)

    def initialize_build(self):
        return _impl.initialize_build(self)

    def finalize_build_on_sucess(self):
        pass

    def finalize_build_on_failure(self):
        tb = traceback.format_exc()
        if tb is not None:
            print(tb, flush=True)

    def show_log_file(self):
        return _impl.show_log_file(self)

    def run(self, args, dry_run=False, **kwargs):
        return _impl.run(self, args, dry_run, **kwargs)

    def steps(self):
        """concrate steps to make this recipe"""
        pass

    def make_all(self):
        try:
            self.report()
            start = time.time()
            strf_start = time.strftime("%d-%m-%Y %H:%M:%S", time.localtime())
            self.initialize_build()
            self.steps()
        except subprocess.CalledProcessError as e:
            logging.error(
                f"!!! failure :( !!!  build [{self.name()}] failed!. cmd= {e.cmd} log {self.log_file()} are show as below:"
            )
            self.finalize_build_on_failure()
            sys.exit(1)
        else:
            end = time.time()
            strf_end = time.strftime("%d-%m-%Y %H:%M:%S", time.localtime())
            elapse = "{:.2f}".format(end - start)
            logging.info(
                f"=== end :) ==== build [{self.name()}] {elapse} seconds done.\n\tplease read {self.log_file()} for details"
            )
            with open(self.log_file(), "a+") as f:
                logging.info("save elapse time")
                time_info = f"TEST_START_TIME={strf_start}\nTEST_END_TIME={strf_end}\nBuild elapse seconds: {elapse}"
                f.write(time_info)
                f.flush()
            f.close()
            self.finalize_build_on_sucess()
        finally:
            pass

    @property
    def dry_run(self):
        return self._dry_run

    @dry_run.setter
    def dry_run(self, value):
        self._dry_run = value
