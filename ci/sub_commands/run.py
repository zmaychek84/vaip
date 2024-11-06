#!/usr/bin/env python
# coding=utf-8

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


import argparse
import json
import logging
import os
import pathlib

import tools.vaip_model_recipe as recipe
from tools import workspace as workspace


def main(args):
    w = workspace.Workspace()
    all_recipes = recipe.create_receipes_from_args(w, args)
    if args.env:
        w.update_environment_from_file(args.env)
    else:
        w.update_environment_from_file(
            [pathlib.Path(__file__).parent.parent.parent / "debug_env.txt"]
        )
    for r in all_recipes:
        r._step_download_onnx()

    for r in all_recipes:
        r._log_file = None
        r._step_compile_onnx()

    if "DUMP_RESULT" in w.environ():
        all_result = []
        focus_keys = [
            "id",
            "onnx_model",
            "md5sum",
            "md5_compare",
            "l2norm",
            "tolerance",
            "result_download_onnx",
            "result_compile_onnx",
            "result_run_onnx",
            "result",
        ]
        for r in all_recipes:
            elem = {}
            for focus_key in focus_keys:
                if focus_key in r._json.keys():
                    elem[focus_key] = r._json[focus_key]
                else:
                    elem[focus_key] = None
            all_result.append(elem)
        with open(w.environ()["DUMP_RESULT"], "w") as f:
            json.dump(all_result, f, indent=4)
        print(f"run result >> {w.environ()['DUMP_RESULT']}")


def help(subparsers):
    parser = subparsers.add_parser("run", help="test model")
    parser.add_argument("case", nargs="*", help="run case with id(could be multiple)")
    parser.add_argument(
        "-e", "--env", nargs="*", help="read enviroment variables from a file"
    )
    parser.set_defaults(func=main)
