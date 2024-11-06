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

from .. import shell


def report(recipe):
    logging.info(f"=== begin ===  start to build [{recipe.name()}]")
    logging.info(f"\tlog_file={recipe.log_file()}")
    logging.info(f"\tbuild_dir={recipe.build_dir()}")
    logging.info(f"\tsrc_dir={recipe.src_dir()}")


def _run(recipe, args, dry_run=False, **kwargs):
    if "env" not in kwargs:
        return shell.run(
            args,
            output=recipe.log_file(),
            dry_run=recipe._dry_run,
            env=recipe._workspace.environ(),
            cwd=recipe._cwd,
            **kwargs,
        )
    else:
        return shell.run(
            args,
            output=recipe.log_file(),
            dry_run=recipe._dry_run,
            cwd=recipe._cwd,
            **kwargs,
        )


def run(recipe, args, dry_run=False, **kwargs):
    msg = f"[{recipe.name()}] running@[ {recipe._cwd} ] : " + " ".join(
        [shlex.quote(str(arg)) for arg in args]
    )
    logging.info(msg)
    try:
        return _run(recipe, args, **kwargs)
    except subprocess.CalledProcessError as e:
        logging.error(f"[{recipe.name()}] command failure: {e.cmd}")
        raise


def initialize_build(recipe):
    if os.path.exists(recipe.log_file()):
        os.remove(recipe.log_file())
    os.makedirs(recipe.log_file().parent, exist_ok=True)
    if recipe.src_dir():
        os.makedirs(recipe.src_dir(), exist_ok=True)
    os.makedirs(recipe.build_dir(), exist_ok=True)
    with open(recipe.log_file(), "w") as file:
        file.write("start to build " + recipe.name())
    file.close()


def show_log_file(recipe):
    with open(recipe.log_file(), "r") as f:
        logging.info(f.read())
    f.close()
