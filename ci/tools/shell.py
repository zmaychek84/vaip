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
import shlex
import subprocess
import sys
import traceback


def write_to_log(output, msg):
    try:
        logging.info(msg)
        file = open(output, "a")
        file.write(msg)
        file.write("\n")
        file.flush()
        file.close()
    except Exception as e:
        logging.error(f"write to log failure: {msg}")


def run(
    args, dry_run=False, quiet=True, output=None, env={}, cwd=os.getcwd(), timeout=None
):
    msg = f"running@[ {cwd} ] : " + " ".join([shlex.quote(str(arg)) for arg in args])
    if not quiet:
        logging.info(msg)
    logging.info(f"output = {output}")
    if not dry_run:
        if output:
            file = open(output, "a")
            file.write(msg)
            file.write("\n")
            file.flush()
            my_stdout = file
        else:
            my_stdout = None
        try:
            return subprocess.run(
                args,
                shell=False,
                check=True,
                stdout=my_stdout,
                stderr=my_stdout,
                env=env,
                cwd=cwd,
                timeout=timeout,
            )
        except subprocess.CalledProcessError as e:
            logging.error(f"command failure: {e.cmd} @[ {cwd} ]")
            raise


class Cwd(object):
    def __init__(self, recipe, cwd):
        self._recipe = recipe
        self._cwd = cwd

    def __enter__(self):
        self._oldcwd = self._recipe._cwd
        self._recipe._cwd = self._cwd
        return self

    def __exit__(self, exc_type, exc_value, tb):
        self._recipe.cwd = self._oldcwd
        if exc_type is not None:
            # traceback.print_exception(exc_type, exc_value, tb)
            return False
        return True


def is_dir(dir_name):
    return os.path.isdir(dir_name)


def is_file(file_name):
    return os.path.isfile(file_name)
