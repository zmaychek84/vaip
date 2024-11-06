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
import concurrent.futures
import json
import logging
import pathlib
import sys
import os

import tools.vaip_model_recipe as recipe
from tools import workspace as workspace


def main(args):
    if hasattr(args, "env"):
        w = workspace.Workspace(args.env)
    else:
        w = workspace.Workspace(
            pathlib.Path(__file__).parent.parent / "regression_test.txt"
        )

    all_recipes = recipe.create_receipes_from_args(w, args)

    executor = concurrent.futures.ThreadPoolExecutor(args.job_count)
    futures = [executor.submit(lambda x: x.make_all(), r) for r in all_recipes]
    for i in range(len(futures)):
        try:
            result = futures[i].result()
        except:
            logging.info(f"error for {all_recipes[i].name()}")
            # all_recipes[i]._json['result'] = 'FAILED@main'
        finally:
            all_recipes[i]._step_compare_with_ref()
    w.save_workspace()

    if "DUMP_RESULT" in w.environ():
        dump_file = w.environ()["DUMP_RESULT"]
        all_result = []
        focus_keys = [
            "id",
            "onnx_model",
            "md5sum",
            "md5_compare",
            "snr_target",
            "psnr_target",
            "l2norm_target",
            "lpips_target",
            "snr",
            "psnr",
            "l2norm",
            "l2norm_hw",
            "snr_hw",
            "psnr_hw",
            "l2norm_detail",
            "accuracy_input",
            "tolerance",
            "result_download_onnx",
            "result_compile_onnx",
            "result_run_onnx",
            "result",
            "acc_type",
        ]
        for r in all_recipes:
            elem = {}
            for focus_key in focus_keys:
                if focus_key in r._json.keys():
                    elem[focus_key] = r._json[focus_key]
                else:
                    elem[focus_key] = None
            if len(all_recipes) == 1:
                if os.path.exists(dump_file):
                    with open(dump_file, "r") as j:
                        all_result = json.load(j)
                    j.close()
            all_result.append(elem)
        with open(dump_file, "w") as f:
            json.dump(all_result, f, indent=4)
        f.close()
        print(os.getcwd(), flush=True)
        print(f"run result >> {dump_file}", flush=True)

    logging.info("all test is finished.")


def help(subparsers):
    import multiprocessing

    parser = subparsers.add_parser("test", help="test model and update reference")
    parser.add_argument("case", nargs="*", help="run case with id(could be multiple)")
    parser.add_argument(
        "-e", "--env", nargs="*", help="read enviroment variables from a file"
    )
    parser.add_argument(
        "-j",
        "--job_count",
        default=multiprocessing.cpu_count(),
        type=int,
        help="job count for how many test cases to run concurrently",
    )
    parser.set_defaults(func=main)
