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
import re
import pathlib
import random
import subprocess
import sys
import threading
import time
import traceback
import shutil
import glob
import numpy as np


from . import recipe, shell, parse_result
from . import convert, update_vaip_config, utility
from .imp import vaip_model_recipe as _impl
from .imp.workspace import _get_model_zoo_name
from . import combine_json_for_vart
from . import fast_accuracy_log_parse

scp_lock = threading.Lock()
golden_lock = threading.Lock()


def time_logged(func):
    func_name = func.__name__

    def wrap(s):
        start = time.time()
        func(s)
        end = time.time()
        elapse = end - start
        key = "step_elapse"
        if key not in s._json:
            s._json[key] = {}
        if func_name not in s._json[key]:
            s._json[key][func_name] = elapse
        else:
            s._json[key][func_name] += elapse
        print(f"{func_name} elapse: {elapse}", flush=True)

    return wrap


class VaipModelRecipe(recipe.Recipe):
    def __init__(self, workspace, json):
        self._json = {
            k: json.get(k, "")
            for k in [
                "id",
                "hostname",
                "onnx_model",
                "float_onnx",
                "onnx_data",
                "input",
                "input_hello_world",
                "perf_input",
                "accuracy_input",
                "accuracy_golden",
                "cos_target",
                "snr_target",
                "psnr_target",
                "l2norm_target",
                "lpips_target",
                "key_output_id",
                "hello_world_key_output_id",
                "hello_world_key_input_id_order",
                "hello_world_key_output_id_order",
                "md5_compare",
                "vaip_env",
                "injected_tiling_param",
                "inject_order_json",
                "xcompilerAttrs",
                "acc_type",
            ]
        }

        self._golden = json["golden"] if "golden" in json else ""
        self._output = ""
        self._print_env = False
        self._clean = False
        self._test_tool = ""
        self._input_prefix = ""
        self._local_input_path = pathlib.Path("/tmp")
        self._local_golden_path = pathlib.Path("/tmp")
        self._cache_dir = pathlib.Path("/tmp")
        self._seq = str(os.environ.get("SET_SEQ", "0"))
        self._golden_np = []
        self._output_np = []
        self._onnx_ep_output_np = []
        self._cpu_ep_output_np = []
        self._cpu_runner_output_np = []
        self._hello_world_output_np = []
        self._golden_files = []
        self._input = []
        self._input_hello_world = []
        self._cos_dict = {}
        self._snr_dict = {}
        self._snr_hw_dict = {}
        self._psnr_dict = {}
        self._psnr_hw_dict = {}
        self._l2norm_dict = {}
        self._l2norm_hw_dict = {}
        self._model_env = {}
        self._first_performance = True

        super().__init__(workspace)

    def name(self):
        return self._json["id"]

    def md5(self):
        return self._json["md5sum"]

    def onnx_model(self):
        """onnx model file in XCD, full path"""
        if self._json.get("ep_context_onnx", ""):
            return pathlib.Path(self._json["ep_context_onnx"])
        else:
            return pathlib.Path(self._json["onnx_model"])

    def quantized_onnx_model(self):
        return pathlib.Path(self._json["onnx_model"])

    def injected_tiling_param(self):
        """onnx.data model file in XCD, full path"""
        injected_tiling_param = (
            None
            if not self._json["injected_tiling_param"]
            else pathlib.Path(self._json["injected_tiling_param"])
        )
        if injected_tiling_param:
            return injected_tiling_param
        return (
            None
            if not self._model_env.get("INJECTED_TILING_PARAM", "")
            else pathlib.Path(self._model_env.get("INJECTED_TILING_PARAM"))
        )

    def inject_order_json(self):
        """onnx.data model file in XCD, full path"""
        inject_order_json = (
            None
            if not self._json["inject_order_json"]
            else pathlib.Path(self._json["inject_order_json"])
        )
        if inject_order_json:
            return inject_order_json
        return (
            None
            if not self._model_env.get("INJECTED_ORDER_JSON", "")
            else pathlib.Path(self._model_env.get("INJECTED_ORDER_JSON"))
        )

    def local_input_path(self):
        return self._local_input_path

    def local_golden_path(self):
        return self._local_golden_path

    def get_golden_files(self):
        """onnx golden file in XCD, full path"""
        return self._golden_files

    def get_onnx_input(self):
        """onnx model file in XCD, full path"""
        if self._input:
            return self._input
        else:
            print(f"[tmp]self._seq is{self._seq}")
            return (
                None
                if not self._json["accuracy_input"]
                else pathlib.Path(self._json["accuracy_input"][self._seq])
            )

        # if "input" in self._json and isinstance(self._json["input"], list):
        #     self._input = [str(path) for path in self._json["input"]]
        # return self._input

    def get_onnx_input_hello_world(self):
        """onnx model file in XCD, full path"""
        if self._input_hello_world:
            return self._input_hello_world

        if "input_hello_world" in self._json and isinstance(
            self._json["input_hello_world"], list
        ):
            self._input_hello_world = [
                str(path) for path in self._json["input_hello_world"]
            ]
        return self._input_hello_world

    def set_onnx_input(self, new_input):
        self._input = new_input

    def set_onnx_hello_world_input(self, new_input):
        if (
            len(new_input) > 1
            and self.hello_world_key_input_id_order()
            and self._model_env.get("TEST_HELLO_WORLD", "true") == "true"
        ):
            inputArr = []
            for item in self.hello_world_key_input_id_order():
                inputArr.append(new_input[int(item)])
            self._input = inputArr
        else:
            self._input = new_input

    def set_golden_files(self, outputs):
        self._golden_files = outputs

    def onnx_golden_list(self):
        golden_files = []
        for fullname in self._golden_files:
            output = os.path.basename(fullname)
            sub_dir = os.path.basename(os.path.dirname(fullname))
            golden_files.append(self.local_golden_path() / sub_dir / output)
        return golden_files

    def onnx_input_list(self, onnx_input, cmd):
        if isinstance(onnx_input, list):
            for input_path in onnx_input:
                input = os.path.basename(input_path)
                sub_dir = os.path.basename(os.path.dirname(input_path))
                input_name = self.local_input_path() / sub_dir / input
                cmd.append(str(input_name))
        else:
            input = os.path.basename(onnx_input)
            sub_dir = os.path.basename(os.path.dirname(onnx_input))
            input_name = self.local_input_path() / sub_dir / input
            cmd.append(str(input_name))

    def float_onnx(self):
        """float onnx model file in XCD, full path"""
        return (
            None
            if not self._json["float_onnx"]
            else pathlib.Path(self._json["float_onnx"])
        )

    def onnx_data(self):
        """onnx.data model file in XCD, full path"""
        return (
            None
            if not self._json["onnx_data"]
            else pathlib.Path(self._json["onnx_data"])
        )

    def key_output_id(self):
        """get key_output_id"""
        return self._json.get("key_output_id", None)

    def hello_world_key_output_id(self):
        """get hello_world_key_output_id"""
        return (
            None
            if not self._json["hello_world_key_output_id"]
            else self._json["hello_world_key_output_id"]
        )

    def hello_world_key_input_id_order(self):
        """get hello_world_key_input_id_order"""
        return (
            None
            if not self._json["hello_world_key_input_id_order"]
            else self._json["hello_world_key_input_id_order"]
        )

    def hello_world_key_output_id_order(self):
        """get hello_world_key_output_id_order"""
        return (
            None
            if not self._json["hello_world_key_output_id_order"]
            else self._json["hello_world_key_output_id_order"]
        )

    def cos_target(self):
        """get cos_target"""
        return self._json.get("cos_target", None)

    def snr_target(self):
        """get snr_target"""
        return self._json.get("snr_target", None)

    def lpips_target(self):
        """get lpips_target"""
        return self._json.get("lpips_target", None)

    def psnr_target(self):
        """get psnr_target"""
        return self._json.get("psnr_target", None)

    def l2norm_target(self):
        """get l2norm_target"""
        return self._json.get("l2norm_target", None)

    def perf_input(self):
        """onnx.data model file in XCD, full path"""
        return (
            None
            if not self._json["perf_input"]
            else pathlib.Path(self._json["perf_input"])
        )

    def accuracy_input(self):
        """accuracy input json, full path"""
        return (
            None
            if not self._json["accuracy_input"]
            else pathlib.Path(self._json["accuracy_input"])
        )

    def accuracy_golden(self):
        """accuracy golden json, full path"""
        return (
            None
            if not self._json["accuracy_golden"]
            else pathlib.Path(self._json["accuracy_golden"])
        )

    def get_accuracy_input_list(self):
        """model file input list in XCD, full path"""
        return (
            None
            if not self._json["accuracy_input"]
            else pathlib.Path(self._json["accuracy_input"])
        )

    def vaip_env(self):
        """onnx.data model file in XCD, full path"""
        return None if not self._json["vaip_env"] else self._json["vaip_env"]

    def xcompiler_attrs(self):
        """onnx.data model file in XCD, full path"""
        return (
            None if not self._json["xcompilerAttrs"] else self._json["xcompilerAttrs"]
        )

    def hostname(self):
        """default server name for the model zoo"""
        return self._json["hostname"]

    def build_dir(self):
        """default build directory for regression test"""
        return (
            self._workspace.build_dir()
            / self._model_env.get("VAIP_REGRESSION", "vaip_regression")
            / self.name()
        )

    def update_env(self, dict):
        for key, val in dict.items():
            self._model_env.update({key: val})
            self.run(["echo", "%s=%s" % (key, val)])

    @time_logged
    def _step_download_onnx(self):
        with shell.Cwd(self, self.build_dir()):
            try:
                if not os.path.isdir(self.build_dir()):
                    os.makedirs(self.build_dir(), exist_ok=True)
                scp_lock.acquire()  # this is due to the fact that remote would close connection if too many requests at same time
                local_onnx_model = os.path.split(str(self.onnx_model().as_posix()))[-1]
                if not os.path.exists(self.build_dir() / local_onnx_model):
                    self.run(
                        [
                            "scp",
                            "-o",
                            "StrictHostKeyChecking=no",
                            "-o",
                            "UserKnownHostsFile=/dev/null",
                            self.hostname() + ":" + str(self.onnx_model().as_posix()),
                            ".",
                        ]
                    )
                    self.set_md5(self.build_dir() / local_onnx_model)
                test_mode = self._model_env.get("TEST_MODE", "performance")
                default_output_checking = {"performance": "cpu_runner,onnx_ep"}
                output_checking_env = self._model_env.get("OUTPUT_CHECKING")
                if output_checking_env == "true":
                    output_checking = default_output_checking.get(
                        test_mode, "cpu_ep,onnx_ep"
                    )
                else:
                    output_checking = (
                        output_checking_env
                        if output_checking_env
                        else default_output_checking.get(test_mode, "cpu_ep,onnx_ep")
                    )
                # output_checking = self._model_env.get("OUTPUT_CHECKING", "cpu_runner,onnx_ep")
                if (
                    output_checking.find("cpu_ep") != -1
                    and self.float_onnx()
                    and not os.path.exists(
                        self.build_dir()
                        / os.path.split(str(self.float_onnx().as_posix()))[-1]
                    )
                ):
                    self.run(
                        [
                            "scp",
                            "-o",
                            "StrictHostKeyChecking=no",
                            "-o",
                            "UserKnownHostsFile=/dev/null",
                            self.hostname() + ":" + str(self.float_onnx().as_posix()),
                            ".",
                        ]
                    )
                if self.onnx_data():
                    self.run(
                        [
                            "scp",
                            "-o",
                            "StrictHostKeyChecking=no",
                            "-o",
                            "UserKnownHostsFile=/dev/null",
                            self.hostname() + ":" + str(self.onnx_data().as_posix()),
                            ".",
                        ]
                    )
                if self.get_onnx_input():
                    for item in self.get_onnx_input():
                        input = os.path.basename(item)
                        sub_dir = self.local_input_path() / os.path.basename(
                            os.path.dirname(item)
                        )
                        if not os.path.isdir(sub_dir):
                            os.makedirs(sub_dir)
                        if os.path.exists(sub_dir / input):
                            continue
                        self.run(
                            [
                                "scp",
                                "-o",
                                "StrictHostKeyChecking=no",
                                "-o",
                                "UserKnownHostsFile=/dev/null",
                                "-r",
                                self.hostname() + ":" + str(item),
                                str(sub_dir),
                            ]
                        )
                if self.get_golden_files():
                    for item in self.get_golden_files():
                        golden = os.path.basename(item)
                        sub_dir = self.local_golden_path() / os.path.basename(
                            os.path.dirname(item)
                        )
                        if not os.path.isdir(sub_dir):
                            os.makedirs(sub_dir)
                        if os.path.exists(sub_dir / golden):
                            continue
                        self.run(
                            [
                                "scp",
                                "-o",
                                "StrictHostKeyChecking=no",
                                "-o",
                                "UserKnownHostsFile=/dev/null",
                                "-r",
                                self.hostname() + ":" + str(item),
                                str(sub_dir),
                            ]
                        )
                inject_flag = (
                    self._model_env.get("ENABLE_PARAM_INJECTION", "") == "true"
                )
                if inject_flag and self.injected_tiling_param():
                    local_csv = os.path.split(
                        str(self.injected_tiling_param().as_posix())
                    )[-1]
                    if not os.path.exists(self.build_dir() / local_csv):
                        self.run(
                            [
                                "scp",
                                "-o",
                                "StrictHostKeyChecking=no",
                                "-o",
                                "UserKnownHostsFile=/dev/null",
                                self.hostname()
                                + ":"
                                + str(self.injected_tiling_param().as_posix()),
                                ".",
                            ]
                        )
                inject_flag = (
                    self._model_env.get("ENABLE_INJECT_ORDER", "false") == "true"
                )
                if inject_flag and self.inject_order_json():
                    local_json = os.path.split(
                        str(self.inject_order_json().as_posix())
                    )[-1]
                    if not os.path.exists(self.build_dir() / local_json):
                        self.run(
                            [
                                "scp",
                                "-o",
                                "StrictHostKeyChecking=no",
                                "-o",
                                "UserKnownHostsFile=/dev/null",
                                self.hostname()
                                + ":"
                                + str(self.inject_order_json().as_posix()),
                                ".",
                            ]
                        )
                scp_lock.release()
                self._json["result_download_onnx"] = "OK"
                # write xrt.ini in build workspace
                if os.environ.get("AIE_TRACE", "") == "true":
                    utility.write_xrt_ini(True)
                    aie_trace_config = str(
                        self._workspace.install_prefix()
                        / f"../ci/tools/config/aie_control_config.json"
                    )
                    if os.path.exists(aie_trace_config.replace("/", "\\")):
                        utility.cp_file(
                            aie_trace_config.replace("/", "\\"),
                            "aie_control_config.json",
                        )
                else:
                    utility.write_xrt_ini(False)

            except subprocess.CalledProcessError as e:
                self._json["result_download_onnx"] = "FAILED"
                self._json["result"] = "FAILED@download_onnx"
                scp_lock.release()

    def set_md5(self, name):
        self._json["md5sum"] = hashlib.md5(open(name, "rb").read()).hexdigest()
        self.run(["echo", "model md5sum is:", self._json["md5sum"]])

    def set_cache_dir(self, id=None):
        cache_dir = pathlib.Path("/tmp") / self._workspace.user()
        if self._workspace.is_windows():
            cache_dir = pathlib.Path("C:\\") / "temp" / self._workspace.user()

        if self._model_env.get("XLNX_CACHE_DIR"):
            self._cache_dir = pathlib.Path(self._model_env.get("XLNX_CACHE_DIR"))
            if id:
                self._cache_dir /= id
            dict = {"XLNX_CACHE_DIR": str(self._cache_dir)}
            self.update_env(dict)
        else:
            self._cache_dir = cache_dir / "vaip" / ".cache" / id

        if not os.path.isdir(self._cache_dir):
            os.makedirs(self._cache_dir)

    def cache_dir(self):
        return self._cache_dir

    def _get_xmodel(self, prefix):
        cache_dir = self.cache_dir()
        cache_sub_dirs = [
            x
            for x in os.listdir(cache_dir)
            if os.path.isdir(os.path.join(cache_dir, x))
        ]
        if len(cache_sub_dirs) == 1:
            cache_dir = os.path.join(cache_dir, cache_sub_dirs[0])
        files = [x for x in os.listdir(cache_dir) if x.startswith(prefix)]
        return None if len(files) == 0 else os.path.join(cache_dir, files[0])

    def _copy_xmodel(self, xmodel):
        try:
            if not xmodel:
                return
            base_name = os.path.basename(xmodel)
            new_name = "%s_%s" % (self._json["id"], base_name)
            new_xmodel = pathlib.Path(os.path.dirname(self.onnx_model())) / new_name
            self.run(
                [
                    "scp",
                    "-o",
                    "StrictHostKeyChecking=no",
                    "-o",
                    "UserKnownHostsFile=/dev/null",
                    xmodel,
                    self.hostname() + ":" + str(new_xmodel.as_posix()),
                ]
            )
        except Exception as e:
            logging.info("Copy xmodel failed: %s" % e)

    def missmatch_postprocess(self, onnx_name):
        try:
            result_status = parse_result.parse_result(self.log_file())
            logging.info("mismatch result status: %s" % result_status)
            pass_clean = self._model_env.get("PASS_CLEAN_CACHE", "")
            if self._model_env.get("IPUTRACE", "mismatch") == "all":
                self._clean = False
                self.trace_test_xmodel()
            elif pass_clean == "all":
                self._clean = True
            elif (
                result_status == 0
                and os.path.exists(self.cache_dir())
                and pass_clean == "pass"
            ):
                self._clean = True
            elif result_status == 1:
                self.test_xmodel_diff()
        except Exception as e:
            logging.info("Clean cache failed: %s" % e)

    def test_xmodel_diff(self):
        try:
            self.run(["echo", "###test_xmodel_diff###"])
            dict = {"USE_CPU_RUNNER": "0"}
            self.update_env(dict)
            compiled_xmodel = self._get_xmodel("compiled")
            if not compiled_xmodel:
                compiled_xmodel = self.onnx_model().name
            xmodel = str(compiled_xmodel)
            if not xmodel.endswith("xmodel"):
                logging.info(
                    "Warning: ignore test_xmodel_diff because of no xmodel found!"
                )
                return
            else:
                tool_name = "test_xmodel_diff"
                if self._workspace.is_windows():
                    tool_name = "test_xmodel_diff.exe"
                test_tool = str(self._workspace.install_prefix() / "bin" / tool_name)
                layer_dir = str(self.build_dir() / "layer_result")
                if self._model_env.get("CI_CACHE_OUTPUT", "false") == "true":
                    layer_dir = str(self.cache_dir() / "layer_result")
                firmware = self._model_env.get("XLNX_VART_FIRMWARE", "")
                self.run(
                    [test_tool, "-i", xmodel, "-f", firmware, "-o", layer_dir],
                    env=self._model_env,
                )

        except Exception as e:
            logging.info("failed: %s" % e)

    def find_files_with_specific_num(self, directory, num, mode, input_base_name=None):
        # print("num mode ------->", num, mode, flush=True)
        matched_files = []
        if mode == "vitisai":
            target_filename = f"{input_base_name}_output_{num[0]}.bin"
        else:
            target_filename = f"output_aie_float_{num[0]}.bin"
        # print("target_filename --->", target_filename, flush=True)
        for filename in os.listdir(directory):
            if filename == target_filename:
                matched_files.append(filename)
        print("matched_files ------->", matched_files, flush=True)
        return matched_files

    def test_hello_world(self, hello_world_golden):
        try:
            self.set_onnx_hello_world_input(self._input)
            l2norm_hw_str = ""
            snr_hw_str = ""
            psnr_hw_str = ""
            l2norm_hw_dict = {}
            compiled_xmodel = self._get_xmodel("compiled")
            if not compiled_xmodel:
                compiled_xmodel = self.onnx_model().name
            xmodel = str(compiled_xmodel)
            if not xmodel.endswith("xmodel"):
                logging.info(
                    "Warning: ignore test_hello_world because of no xmodel found!"
                )
                return

            self._hello_world_output = ""
            tool_name = (
                "hello_world.exe" if self._workspace.is_windows() else "hello_world"
            )
            self._test_tool = str(self._workspace.install_prefix() / "bin" / tool_name)
            cmd = [self._test_tool, xmodel]
            self.onnx_input_list(self.get_onnx_input(), cmd)
            self.run(cmd, timeout=int(self._model_env.get("TIMEOUT", "1200")))
            current_directory = self.build_dir()
            vitisai_directory = os.path.join(current_directory, "vitisai_ep")
            cpu_ep_directory = os.path.join(current_directory, "cpu_ep_float")
            target_directory = os.path.join(current_directory, "hello_world")
            self._prepare_directory(target_directory)
            input_base_name = os.path.basename(self.get_onnx_input()[0])
            print("input_base_name --->", input_base_name, flush=True)
            file_pattern = os.path.join(current_directory, "output_aie_float*")
            matching_files = glob.glob(file_pattern)
            print("matching_files ---> matching_files", matching_files, flush=True)
            # output_checking = self._model_env.get("OUTPUT_CHECKING", "cpu_runner,onnx_ep")
            test_mode = self._model_env.get("TEST_MODE", "performance")
            default_output_checking = {"performance": "cpu_runner,onnx_ep"}
            output_checking_env = self._model_env.get("OUTPUT_CHECKING")
            if output_checking_env == "true":
                output_checking = default_output_checking.get(
                    test_mode, "cpu_ep,onnx_ep"
                )
            else:
                output_checking = (
                    output_checking_env
                    if output_checking_env
                    else default_output_checking.get(test_mode, "cpu_ep,onnx_ep")
                )
            for file_path in matching_files:
                if os.path.isfile(file_path):
                    shutil.copy(file_path, target_directory)
                    if (
                        self._json["id"] in ("L_v_1_0_XINT8", "L_v_1_0_A8W8")
                        and output_checking != "performance"
                    ):
                        L_file_pattern = os.path.join(
                            target_directory, "output_aie_float*"
                        )
                        L_matching_files = glob.glob(L_file_pattern)
                        new_L_file_name = input_base_name + "_output_0.bin"
                        new_L_file_path = os.path.join(
                            target_directory, new_L_file_name
                        )
                        os.rename(L_matching_files[0], new_L_file_path)
            files = []
            i = 0
            for filename in os.listdir(target_directory):
                file_path = os.path.join(target_directory, f"output_aie_float_{i}.bin")
                files.append(file_path)
                i += 1
            if self.hello_world_key_output_id_order():
                files = []
                for item in self.hello_world_key_output_id_order():
                    file_path = os.path.join(
                        target_directory, f"output_aie_float_{int(item)}.bin"
                    )
                    files.append(file_path)
            print("files ----> ", files, flush=True)

            self._get_model_result_md5(files)
            self._hello_world_output_np = self._output_np

            if len(hello_world_golden) != len(self._hello_world_output_np):
                result_str = f"vitisai_ep output number {len(hello_world_golden)}, hello_world output number {len(self._hello_world_output_np)}"
                shell.write_to_log(self.log_file(), f"l2normDesc: {result_str}")
        except Exception as e:
            logging.info("test_hello_world failed: %s" % e)

    def _prepare_directory(self, target_directory):
        if not os.path.exists(target_directory):
            os.makedirs(target_directory)

    def _get_result_str(self, hw_dict):
        values_list = [val for val in hw_dict.values()]
        return "/".join(str(sublist[0]) for sublist in values_list)

    def dump_inst_data(self):
        try:
            tool_name = "dump_inst_data"
            if self._workspace.is_windows():
                tool_name = "dump_inst_data.exe"
            test_tool = str(self._workspace.install_prefix() / "bin" / tool_name)
            compiled_xmodel = self._get_xmodel("compiled")
            if compiled_xmodel and pathlib.Path.is_file(
                self._workspace.install_prefix() / "bin" / tool_name
            ):
                self.run(["echo", "###dump_inst_data###"])
                xmodel = str(compiled_xmodel)
                self.run(
                    [test_tool, xmodel],
                    env=self._model_env,
                )
            else:
                logging.info(
                    "Warning: ignore dump_inst_data because of no compiled xmodel generated or dump_inst_data!"
                )
                return
        except Exception as e:
            logging.info("failed: %s" % e)

    def trace_test_xmodel(self):
        try:
            if self._workspace.is_windows():
                self.run(["echo", "###trace_test_xmodel###"])

                dict = {"USE_CPU_RUNNER": "0"}
                self.update_env(dict)
                compiled_xmodel = self._get_xmodel("compiled")
                if not compiled_xmodel:
                    logging.info(
                        "Warning: ignore test_xmodel_diff because of no compiled xmodel generated!"
                    )
                    return
                else:
                    self.copy_trace_pdb()
                    xmodel = str(compiled_xmodel)
                    iputrace_tool = str(
                        self._workspace.install_prefix() / "bin" / "fpstrace.cmd"
                    )
                    fps_bat = str(
                        self._workspace.install_prefix() / "bin" / "fps_bat.bat"
                    )
                    tool_name = "test_xmodel_diff"
                    if self._workspace.is_windows():
                        tool_name = "test_xmodel_diff.exe"
                    test_tool = str(
                        self._workspace.install_prefix() / "bin" / tool_name
                    )

                    self.run(
                        [fps_bat, xmodel, iputrace_tool, test_tool], env=self._model_env
                    )
                    # self.run([ iputrace_tool, "start" ]) if self._model_env.get("IPUTRACE", "false") == "true" else None
                    # self.run([ test_tool, xmodel])
                    # self.run([ iputrace_tool, "stop" ]) if self._model_env.get("IPUTRACE", "false") == "true" else None
                    # self.run([ iputrace_tool, "view" ]) if self._model_env.get("IPUTRACE", "false") == "true" else None
            else:
                return
        except Exception as e:
            logging.info("failed: %s" % e)

    def run_command(self, cmd, run_env=None):
        try:
            time_out = self._model_env.get("TIMEOUT", "36000")
            if not run_env:
                run_env = self._model_env.copy()
            self.run(
                cmd,
                env=run_env,
                timeout=int(time_out),
            )
            self._json["returncode"] = ""
        except Exception as e:
            self._json["returncode"] = e.returncode
            # print("failed! print self._json", self._json)
            logging.info(f"failed: {e}, timeout limit is {time_out}")

    def copy_trace_pdb(self):
        try:
            kipudrv_path = os.path.join(
                self._workspace.install_prefix() / "bin", "kipudrv.pdb"
            )
            kipudrv_win10_path = os.path.join(
                self._workspace.install_prefix() / "bin", "kipudrv_win10.pdb"
            )
            get_fps_path = os.path.join(
                self._workspace.install_prefix() / "bin", "get_fps.py"
            )
            ipustack_path = os.path.join(
                self._workspace.install_prefix() / "bin", "ipustack.pdb"
            )
            # print(f"cp =============>{kipudrv_path}  self.build_dir()", flush=True)
            shutil.copy(get_fps_path, self.build_dir())
            shutil.copy(ipustack_path, self.build_dir())
            shutil.copy(kipudrv_path, self.build_dir())
            # print(
            #    f"cp =============>{kipudrv_win10_path}  self.build_dir()", flush=True
            # )
            # shutil.copy(kipudrv_win10_path, self.build_dir())
        except Exception as e:
            logging.info("copy files failed: %s" % e)

    def copy_dump_files(self, build_dir, dest_dir):
        tmp_files = [
            x for x in os.listdir(build_dir) if x.startswith("onnx-dpu_interface")
        ]
        for tmp_file in tmp_files:
            org_file = os.path.join(build_dir, tmp_file)
            if os.path.isfile(org_file):
                det_file = os.path.join(dest_dir, tmp_file)
                shutil.copyfile(org_file, det_file)

    def copy_files(self, onnx_name, cache_dir):
        try:
            target_dir = self.build_dir()
            # copy build.log
            cache_sub_dirs = [
                x
                for x in os.listdir(cache_dir)
                if os.path.isdir(os.path.join(cache_dir, x))
            ]
            if len(cache_sub_dirs) == 1:
                cache_dir = os.path.join(cache_dir, cache_sub_dirs[0])
            org_vaip_config = os.path.join(target_dir, "build.log")
            det_vaip_config = os.path.join(cache_dir, "build.log")
            if os.path.exists(org_vaip_config):
                # print(
                #    f"cp =============>{org_vaip_config} {det_vaip_config}", flush=True
                # )
                shutil.copyfile(org_vaip_config, det_vaip_config)
            # vitisai_ep_report
            org_vaip_config = os.path.join(cache_dir, "vitisai_ep_report.json")
            det_vaip_config = os.path.join(target_dir, "vitisai_ep_report.json")
            if os.path.exists(org_vaip_config):
                # print(
                #    f"cp =============>{org_vaip_config} {det_vaip_config}", flush=True
                # )
                shutil.copyfile(org_vaip_config, det_vaip_config)
            # vitisai_ep_dpu_report
            org_dpu_config = os.path.join(cache_dir, "vitisai_ep_dpu_report.json")
            det_dpu_config = os.path.join(target_dir, "vitisai_ep_dpu_report.json")
            if os.path.exists(org_dpu_config):
                # print(f"cp =============>{org_dpu_config} {det_dpu_config}", flush=True)
                shutil.copyfile(org_dpu_config, det_dpu_config)
            # context
            org_context = os.path.join(cache_dir, "context.json")
            det_context = os.path.join(target_dir, "context.json")
            if os.path.exists(org_context):
                # print(f"cp =============>{org_context} {det_context}", flush=True)
                shutil.copyfile(org_context, det_context)
        except Exception as e:
            logging.info("copy files failed: %s" % e)

    def scp_perf_input(self):
        try:
            print("perf_input --->", self.perf_input(), flush=True)
            current_directory = self.build_dir()
            test_data_set_path = os.path.join(current_directory, "test_data_set_0")
            if not os.path.exists(test_data_set_path):
                perf_input_dir = str(self.perf_input())
                host_name = "xcdl190074"
                print(f"scp {perf_input_dir} to board ...", flush=True)
                cmd_list = [
                    "scp",
                    "-r",
                    "-o",
                    "StrictHostKeyChecking=no",
                    "-o",
                    "UserKnownHostsFile=/dev/null",
                    host_name + ":" + perf_input_dir,
                    current_directory,
                ]
                print(f"{(' ').join([str(each) for each in cmd_list])}", flush=True)
                subprocess.check_call(cmd_list)
            else:
                print("The folder test_data_set already exists", flush=True)
        except Exception as e:
            logging.info("scp perf input failed: %s" % e)

    def run_onnxruntime_perf_test(
        self,
        test_tool,
        ep_type,
        vaip_config,
        onnx_name,
        trace_state,
        run_env,
        test_time,
        thread="1",
    ):
        set_config = f"config_file|{vaip_config}"
        if self._model_env.get("ENABLE_CACHE_FILE_IO_IN_MEM", "0") == "1":
            set_config = f"config_file|{vaip_config} enable_cache_file_io_in_mem|1"
        else:
            set_config = f"config_file|{vaip_config} enable_cache_file_io_in_mem|0"
        args = [
            test_tool,
        ]
        if trace_state or not self.perf_input():
            args.append("-I")
        else:
            self.scp_perf_input()
        if self._model_env.get("GEN_PROFILER_FILE", "false") == "true":
            args.append("-p onnxruntime_profile")
        custom_op_dll = run_env.get("CUSTOM_OP_DLL_PATH", "")
        if custom_op_dll != "":
            args.extend(["-R", custom_op_dll])
        args += [
            "-e",
            ep_type,
            "-i",
            set_config,
            "-t",
            test_time,
            "-c",
            thread,
            f".\\{onnx_name}",
        ]
        self.run_command(
            args,
            run_env=run_env,
        )

    def get_profiling_env(self, benchmark_env):
        profiling_env = benchmark_env.copy()
        profiling_env["NUM_OF_DPU_RUNNERS"] = "1"
        if os.environ.get("AIE_TRACE", "") == "true":
            profiling_env["AIE_TRACE_METRICSET"] = "all:input_output_ports_stalls"
        # profiling_env["Debug.ml_timeline"] = "true"

        return profiling_env

    def update_benchmark_env(self, benchmark_env, perf_env):
        for key, val in perf_env.items():
            shell.write_to_log(self.log_file(), f"{key}={val}")
        benchmark_env.update(perf_env)
        # self.run(
        #    [sys.executable, pathlib.Path(__file__).parent.parent / "printenv.py"],
        #    env=benchmark_env,
        # )

    def vaitrace_profiling(
        self, test_tool, vaip_config, onnx_name, test_time, thread, run_env
    ):
        set_config = f"config_file|{vaip_config}"
        if self._model_env.get("ENABLE_CACHE_FILE_IO_IN_MEM", "0") == "1":
            set_config = f"config_file|{vaip_config} enable_cache_file_io_in_mem|1"
        else:
            set_config = f"config_file|{vaip_config} enable_cache_file_io_in_mem|0"
        python_path = "python"
        if self._workspace.is_windows():
            python_path = str(
                self._workspace.install_prefix() / "python" / "python.exe"
            )
        vaitrace_path = str(
            self._workspace.install_prefix() / "bin" / "vaitrace" / "vaitrace.py"
        )
        vaitrace_exe = (
            self._workspace.install_prefix() / "python" / "Scripts" / "vaitrace.exe"
        )
        args = []
        if os.path.exists(vaitrace_exe):
            args += [
                vaitrace_exe,
            ]
        else:
            args += [python_path, vaitrace_path]
        args += [
            "-t",
            "500000",
            "-d",
            "--csv",
            test_tool,
            "-I",
            "-e",
            "vitisai",
            "-i",
            set_config,
            "-t",
            test_time,
            "-c",
            thread,
            f".\\{onnx_name}",
        ]
        if self._model_env.get("GEN_PROFILER_FILE", "false") == "true":
            args.insert(6, "-p onnxruntime_profile")
        custom_op_dll = run_env.get("CUSTOM_OP_DLL_PATH", "")
        if custom_op_dll != "":
            args.insert(6, "-R")
            args.insert(7, custom_op_dll)
        self.run_command(args, run_env)
        # backup vaitrace record_timer_ts.json
        if os.path.exists(self.build_dir() / "record_timer_ts.json"):
            shutil.copy(
                self.build_dir() / "record_timer_ts.json",
                self.build_dir() / "vaitrace_record_timer_ts.json",
            )

    def write_new_vaip_config(self):
        vaip_config = self._model_env.get("VITISAI_EP_JSON_CONFIG")
        vaip_config_name = "vaip_config.json"
        try:
            print(f"Note: read origin {vaip_config}", flush=True)
            json_file = open(vaip_config, "r")
            json_dict = json.load(json_file)
            json_file.close()
            update_vaip_config.update_from_env(
                self._model_env,
                self.xcompiler_attrs(),
                self.injected_tiling_param(),
                self.inject_order_json(),
                json_dict,
            )

            new_config = str(self.build_dir() / vaip_config_name)
            print(f"Note: write {new_config}", flush=True)
            json_file = open(new_config, "w")
            json.dump(json_dict, json_file, indent=4)
            json_file.close()
            dict = {"VITISAI_EP_JSON_CONFIG": new_config}
            self.update_env(dict)
        except Exception as e:
            logging.warning(f"!!! warning : write {vaip_config_name} failed! {e}.)")
            tb = traceback.format_exc()
            if tb is not None:
                print(tb, flush=True)

    def vaitrace_vart_perf(self, test_cmd, run_env):
        python_path = "python"
        if self._workspace.is_windows():
            python_path = str(
                self._workspace.install_prefix() / "python" / "python.exe"
            )
        vaitrace_path = str(
            self._workspace.install_prefix() / "bin" / "vaitrace" / "vaitrace.py"
        )
        vaitrace_exe = (
            self._workspace.install_prefix() / "python" / "Scripts" / "vaitrace.exe"
        )
        args = []
        if os.path.exists(vaitrace_exe):
            args += [
                vaitrace_exe,
            ]
        else:
            args += [python_path, vaitrace_path]
        args += [
            "-t",
            "500000",
            "-d",
            "--csv",
        ]
        args += test_cmd
        self.run_command(args, run_env)

    def run_perf_xmodel(
        self, benchmark_env, perf_env, compiled_xmodel, enable_vaitrace
    ):
        try:
            tool_name = "vart_perf"
            if self._workspace.is_windows():
                tool_name = "vart_perf.exe"
            # test_tool = os.path.join("C:\jenkins\workspace", tool_name)
            test_tool = str(self._workspace.install_prefix() / "bin" / tool_name)
            if compiled_xmodel:
                run_times = self._model_env.get("VART_PERF_CYCLES", "3000")
                func_run_times = self._model_env.get("VART_PERF_FUNC_CYCLES", "10")
                func_threads = self._model_env.get("VART_PERF_FUNC_THREADS", "2")
                vart_perf_env = benchmark_env.copy()
                vart_perf_env["NUM_OF_DPU_RUNNERS"] = "1"
                self.update_benchmark_env(benchmark_env, perf_env)
                self.run(["echo", "###vitisai_xmodel###"])
                vart_cmd = [
                    test_tool,
                    "-x",
                    compiled_xmodel,
                    "-r",
                    run_times,
                ]
                trace_name = "fpstrace.cmd"
                trace_tool = str(self._workspace.install_prefix() / "bin" / trace_name)
                trace_state = self._model_env.get("IPUTRACE", "") != ""
                if trace_state and (
                    self._model_env.get("MODEL_TYPE", "onnx") == "xmodel"
                    or self._model_env.get("TEST_MODE", "performance") == "vart_perf"
                ):
                    self.run(["echo", "###fpstrace###"])
                    self.copy_trace_pdb()
                    fps_vart_bat = str(
                        self._workspace.install_prefix() / "bin" / "fps_vart_bat.bat"
                    )
                    self.run(
                        [
                            fps_vart_bat,
                            trace_tool,
                            test_tool,
                            compiled_xmodel,
                            run_times,
                        ],
                    )
                if enable_vaitrace:
                    self.vaitrace_vart_perf(
                        vart_cmd, self.get_profiling_env(benchmark_env)
                    )
                else:
                    self.run_command(vart_cmd, run_env=benchmark_env)

                if self._model_env.get("LAYER_COMPARE", "false") == "true":
                    vart_cmd = [
                        test_tool,
                        "-x",
                        compiled_xmodel,
                    ]
                    if self._model_env.get("VART_PERF_MULTI_FUNC", "") == "true":
                        vart_cmd.extend(
                            ["-r", func_run_times, "-t", func_threads, "--mcompare"]
                        )
                    else:
                        vart_cmd.append("--xt")
                    self.run_command(vart_cmd, run_env=benchmark_env)
            else:
                logging.warning(
                    f"!!! warning : no {compiled_xmodel} for perf_xmodel test failed!"
                )
        except Exception as e:
            logging.warning(f"!!! warning : run perf_xmodel failed! {e}.)")

    def create_config_json(self, compiled_xmodel, n=1):
        json_path = self.build_dir() / "branch_model.json"
        json_file = open(json_path, "w")
        json_str = {}
        for i in range(n):
            json_str[i + 1] = compiled_xmodel
        json.dump(json_str, json_file)
        json_file.close()
        return json_path

    def run_branch_xmodel(self, compiled_xmodel):
        try:
            tool_name = "branch_model"
            if self._workspace.is_windows():
                tool_name = "branch_model.exe"
            test_tool = str(self._workspace.install_prefix() / "bin" / tool_name)
            if compiled_xmodel:
                run_times = self._model_env.get("BRANCH_MODEL_DURATION", "10")
                firmware = self._model_env.get("XLNX_VART_FIRMWARE", "")
                if not firmware:
                    cache_dir = self.cache_dir()
                    if len(os.listdir(cache_dir)) == 1:
                        cache_md5_dir = os.path.join(
                            cache_dir, os.listdir(cache_dir)[0]
                        )
                        xmodels = [
                            x for x in os.listdir(cache_md5_dir) if x.endswith("xclbin")
                        ]
                        if len(xmodels) == 1:
                            firmware = os.path.join(cache_md5_dir, xmodels[0])
                        print("use saved xclbin %s" % firmware)

                branch_env = {
                    "DEBUG_DIFF": "1",
                    "DURATION": run_times,
                    "XLNX_VART_FIRMWARE": firmware,
                }
                self.run(["echo", f"###branch_model###"])
                for i in range(4):
                    json_path = self.create_config_json(compiled_xmodel, i + 1)
                    self.run(["echo", f"####branch_model threads {i+1}####"])
                    branch_cmd = [
                        test_tool,
                        json_path,
                    ]
                    self.run_command(branch_cmd, run_env=branch_env)
            else:
                logging.warning(
                    f"!!! warning : no {compiled_xmodel} for run_branch_xmodel test failed!"
                )
        except Exception as e:
            logging.warning(f"!!! warning : run run_branch_xmodel failed! {e}.)")

    @time_logged
    def _run_print_env(self):
        self._print_env = True
        self.run(
            [sys.executable, pathlib.Path(__file__).parent.parent / "printenv.py"],
            env=self._model_env,
        )

        if self._workspace.is_windows():
            xrt_tool = str(
                pathlib.Path("C:\\") / "Windows" / "System32" / "AMD" / "xbutil.exe"
            )
            xbutil_download = (
                pathlib.Path("C:\\")
                / "ipu_stack_rel_silicon"
                / "kipudrv"
                / "xbutil.exe"
            )
            if os.path.exists(xbutil_download):
                xrt_tool = str(xbutil_download)
            xrt_smi = str(
                pathlib.Path("C:\\") / "Windows" / "System32" / "AMD" / "xrt-smi.exe"
            )
            if os.path.exists(xrt_smi):
                xrt_tool = xrt_smi
            try:
                self.run(["systeminfo"])
                self.run([xrt_tool, "examine", "-d", "00c5:00:01.1", "-r", "all"])
            except Exception as e:
                print(e, flush=True)
                # tb = traceback.format_exc()
                # if tb is not None:
                #    print(tb, flush=True)

    @time_logged
    def _step_gen_ep_context(self):
        # print("_step_gen_ep_context: ", self._first_performance, flush=True)
        if self._json.get("result_gen_ep_onnx", "") == "PASS":
            return
        with shell.Cwd(self, self.build_dir()):
            onnx_name = self.onnx_model().name
            tool_name = "test_onnx_runner"
            if self._workspace.is_windows():
                tool_name = "test_onnx_runner.exe"
            self._test_tool = str(self._workspace.install_prefix() / "bin" / tool_name)

            gen_ep_context_env = self._model_env.copy()
            gen_ep_context_env["XLNX_ENABLE_CACHE_CONTEXT"] = "1"
            custom_op_dll = gen_ep_context_env.get("CUSTOM_OP_DLL_PATH", "")
            cmd = [
                self._test_tool,
            ]
            if custom_op_dll != "":
                cmd.extend(["-c", custom_op_dll])
            cmd.extend(
                [
                    onnx_name,
                ]
            )

            self.run(["echo", "###generate_ep_context###"])
            self.run_command(cmd, run_env=gen_ep_context_env)

            ep_context_onnx = os.path.splitext(onnx_name)[0] + ".onnx_ctx.onnx"
            log = os.path.join(self.build_dir(), "build.log")
            result_lines = utility.read_lines(log)
            dpu_subgraph_0 = utility.pattern_match(r".*dpu subgraph: 0", result_lines)
            if os.path.exists(ep_context_onnx):
                self._json["ep_context_onnx"] = ep_context_onnx
                self._json["result_gen_ep_onnx"] = "PASS"
            elif dpu_subgraph_0 is not None:
                self._json["result_gen_ep_onnx"] = "PASS"
            else:
                self._json["result_gen_ep_onnx"] = "FAILED"
                self._json["result"] = "FAILED@generate_ep_context"
                self.run(["echo", "FAILED@generate_ep_context"])
        model_cache_dir = self.cache_dir()
        if os.path.exists(model_cache_dir):
            print("rm existed model cache dir: %s" % model_cache_dir)
            shutil.rmtree(model_cache_dir)

    @time_logged
    def _step_compile_onnx(self):
        # print("_step_compile_onnx: ", self._first_performance, flush=True)
        if self._json.get("result_gen_ep_onnx", "") == "FAILED":
            return
        with shell.Cwd(self, self.build_dir()):
            self._json["result_run_onnx"] = "FAILED"
            onnx_name = self.onnx_model().name
            try:
                print(os.getcwd(), flush=True)
                # self.set_md5(onnx_name)
                # self.set_cache_dir(self.md5(), self._json["id"])
                self._json["result_compile_onnx"] = "FAILED"
                test_mode = self._model_env.get("TEST_MODE", "performance")
                # output_checking = self._model_env.get("OUTPUT_CHECKING", "cpu_runner,onnx_ep")
                default_output_checking = {"performance": "cpu_runner,onnx_ep"}
                output_checking_env = self._model_env.get("OUTPUT_CHECKING")
                if output_checking_env == "true":
                    output_checking = default_output_checking.get(
                        test_mode, "cpu_ep,onnx_ep"
                    )
                else:
                    output_checking = (
                        output_checking_env
                        if output_checking_env
                        else default_output_checking.get(test_mode, "cpu_ep,onnx_ep")
                    )
                self.acc_test_models = [x.strip() for x in output_checking.split(",")]
                tool_name = "test_onnx_runner"
                if self._workspace.is_windows():
                    tool_name = "test_onnx_runner.exe"
                self._test_tool = str(
                    self._workspace.install_prefix() / "bin" / tool_name
                )
                test_time = self._model_env.get("TEST_TIME", "30")
                test_seed = self._model_env.get("TEST_TIME2", "5")
                test_seed = self._model_env.get("TEST_SEED", "5")
                benchmark_env = self._model_env.copy()
                perf_env = {
                    "DEBUG_RUNNER": "0",
                    "XLNX_ENABLE_DUMP": "0",
                    "DEBUG_GRAPH_RUNNER": "0",
                    "DUMP_SUBGRAPH_OPS": "false",
                }
                if (
                    self._model_env.get("MODEL_TYPE", "onnx") == "xmodel"
                    or self._model_env.get("TEST_MODE", "performance") == "vart_perf"
                ):
                    enable_vaitrace = (
                        self._model_env.get("VAITRACE_PROFILING", "") == "true"
                    )
                    self.run_perf_xmodel(
                        benchmark_env, perf_env, onnx_name, enable_vaitrace
                    )
                    json_file = os.path.join(self.build_dir(), "vart_perf_result.json")
                    func_result = ""
                    if os.path.exists(json_file):
                        vart_result = json.load(open(json_file))
                        func_result = vart_result[0].get("result")
                    # if self._model_env.get("LAYER_COMPARE", "false") == "true":
                    self.test_xmodel_diff()
                if test_mode == "mismatch":
                    dict = {
                        "ENABLE_CACHE_FILE_IO_IN_MEM": os.environ.get(
                            "ENABLE_CACHE_FILE_IO_IN_MEM", "0"
                        )
                    }
                    self.update_env(dict)
                    tool_name = "classification"
                    if self._workspace.is_windows():
                        tool_name = "classification.exe"
                    self._test_tool = str(
                        self._workspace.install_prefix() / "bin" / tool_name
                    )
                    test_jpg = str(self._workspace.install_prefix() / "bin" / "dog.jpg")
                    dict = {"DEEPHI_PROFILING": "1"}
                    self.update_env(dict)

                    self.run(["echo", "###test_dpu_ep###"])
                    dict = {"USE_CPU_RUNNER": "0"}
                    self.update_env(dict)
                    self.run_command([self._test_tool, onnx_name, test_jpg])

                    self.run(["echo", "###test_cpu_ep###"])
                    quantized_onnx_model = self.quantized_onnx_model().name
                    self.run_command(
                        [self._test_tool, "-n", quantized_onnx_model, test_jpg]
                    )

                    self.run(["echo", "###test_cpu_runner###"])
                    dict = {"USE_CPU_RUNNER": "1"}
                    self.update_env(dict)
                    self.run_command([self._test_tool, onnx_name, test_jpg])
                    dict = {"USE_CPU_RUNNER": "0"}
                    self.update_env(dict)

                    self.missmatch_postprocess(onnx_name)

                if (
                    "performance" in test_mode
                    and self._model_env.get("MODEL_TYPE", "onnx") == "onnx"
                    and self._first_performance
                ):
                    if os.path.exists(self.build_dir() / "log"):
                        shutil.rmtree(self.build_dir() / "log")

                    tool_name = "onnxruntime_perf_test"
                    if self._workspace.is_windows():
                        tool_name = "onnxruntime_perf_test.exe"
                    self._test_tool = str(
                        self._workspace.install_prefix() / "bin" / tool_name
                    )
                    vaip_config = self._model_env.get("VITISAI_EP_JSON_CONFIG")
                    target_ep = self._model_env.get("TARGET_EP", "vitisai")
                    trace_name = "fpstrace.cmd"
                    trace_tool = str(
                        self._workspace.install_prefix() / "bin" / trace_name
                    )
                    trace_state = self._model_env.get("IPUTRACE", "") != ""
                    self.update_benchmark_env(benchmark_env, perf_env)

                    if self._model_env.get("SKIP_CPU_EP", "true") != "true":
                        self.run_command(["echo", "###cpu_ep_latency###"])
                        self.run_onnxruntime_perf_test(
                            self._test_tool,
                            "cpu",
                            vaip_config,
                            onnx_name,
                            trace_state,
                            benchmark_env,
                            test_time,
                            thread="1",
                        )
                    # latency should set NUM_OF_DPU_RUNNERS 1
                    self.update_benchmark_env(
                        benchmark_env, {"NUM_OF_DPU_RUNNERS": "1"}
                    )
                    if trace_state:
                        self.run(["echo", "###fpstrace###"])
                        self.copy_trace_pdb()
                        fps_perf_bat = str(
                            self._workspace.install_prefix() / "bin" / "fps_per_bat.bat"
                        )
                        self.run(
                            [
                                fps_perf_bat,
                                trace_tool,
                                self._test_tool,
                                vaip_config,
                                onnx_name,
                                test_time,
                            ],
                            env=benchmark_env,
                        )

                    self.run(["echo", "###vitisai_ep_latency###"])
                    if self._model_env.get("VAITRACE_PROFILING", "") == "true":
                        self.vaitrace_profiling(
                            self._test_tool,
                            vaip_config,
                            onnx_name,
                            test_time,
                            "1",
                            self.get_profiling_env(benchmark_env),
                        )
                    else:
                        self.run_onnxruntime_perf_test(
                            self._test_tool,
                            target_ep,
                            vaip_config,
                            onnx_name,
                            trace_state,
                            benchmark_env,
                            test_time,
                            thread="1",
                        )

                    self.update_benchmark_env(
                        benchmark_env, {"Debug.ml_timeline": "false"}
                    )
                    if self._model_env.get("SKIP_CPU_EP", "true") != "true":
                        self.run(["echo", "###cpu_ep_throughput###"])
                        self.run_onnxruntime_perf_test(
                            self._test_tool,
                            "cpu",
                            vaip_config,
                            onnx_name,
                            trace_state,
                            benchmark_env,
                            test_time,
                            thread="16",
                        )

                    thread = self._model_env.get("THREAD", "1")
                    if (
                        thread != "1"
                        or self._model_env.get("AGM_CAPTURE", "") == "true"
                    ):
                        self.run(["echo", "###vitisai_ep_throughput###"])
                        dpu_runners = self._model_env.get("NUM_OF_DPU_RUNNERS", "1")
                        self.update_benchmark_env(
                            benchmark_env, {"NUM_OF_DPU_RUNNERS": dpu_runners}
                        )
                        if self._model_env.get("AGM_CAPTURE", "") == "true":
                            from .agm import agm_stats as stats

                            with stats.AGMCollector(
                                "tmp", self.build_dir() / "output.csv"
                            ):
                                self.run_onnxruntime_perf_test(
                                    self._test_tool,
                                    target_ep,
                                    vaip_config,
                                    onnx_name,
                                    trace_state,
                                    benchmark_env,
                                    test_time,
                                    thread=thread,
                                )
                            if self._model_env.get("AGM_VISUALIZER", "") == "true":
                                target_type = self._model_env.get(
                                    "TARGET_TYPE", "STRIX"
                                )
                                python_path = str(
                                    self._workspace.install_prefix()
                                    / "python"
                                    / "python.exe"
                                )
                                self.run(
                                    [
                                        python_path,
                                        str(
                                            self._workspace.install_prefix()
                                            / f"../ci/tools/agm/agm_visualizer.py"
                                        ),
                                        self.build_dir() / "output.csv",
                                        "--target_type",
                                        target_type,
                                    ]
                                )
                        else:
                            self.run_onnxruntime_perf_test(
                                self._test_tool,
                                target_ep,
                                vaip_config,
                                onnx_name,
                                trace_state,
                                benchmark_env,
                                test_time,
                                thread=thread,
                            )

                    if self._model_env.get("PERF_XMODEL", "") == "true":
                        self.run_perf_xmodel(
                            benchmark_env, perf_env, self._get_xmodel("compiled"), False
                        )
                    if self._model_env.get("BRANCH_XMODEL", "") == "true":
                        self.run_branch_xmodel(self._get_xmodel("compiled"))

                    if output_checking_env != "false":
                        self._output = ""
                        tool_name = "test_onnx_runner"
                        if self._workspace.is_windows():
                            tool_name = "test_onnx_runner.exe"
                        self._test_tool = str(
                            self._workspace.install_prefix() / "bin" / tool_name
                        )
                        graph_runner_env = self._model_env.copy()
                        graph_runner_env["DEBUG_RUNNER"] = "1"
                        graph_runner_env["XLNX_ENABLE_DUMP"] = "1"
                        graph_runner_env["DEBUG_GRAPH_RUNNER"] = "1"
                        graph_runner_env["DUMP_SUBGRAPH_OPS"] = "true"

                        # can not use with -l here because the xmodel not clean
                        cmd = [self._test_tool]
                        custom_op_dll = graph_runner_env.get("CUSTOM_OP_DLL_PATH", "")
                        if custom_op_dll != "":
                            cmd.extend(["-c", custom_op_dll])
                        if self.get_onnx_input():
                            cmd.append(onnx_name)
                            self.onnx_input_list(self.get_onnx_input(), cmd)
                        else:
                            cmd.extend(["-s", test_seed, onnx_name])

                        self.run(["echo", "###vitisai_ep_tensor###"])
                        graph_runner_env["USE_CPU_RUNNER"] = "0"
                        self.run_command(
                            cmd,
                            graph_runner_env,
                        )
                        self._get_model_result_md5(self._get_model_result("vitisai_ep"))
                        self.run(["echo", "onnx_ep output tensor: %s" % self._output])
                        vitisai_output = self._output
                        hello_world_golden = self._output_np
                        self._onnx_ep_output_np = self._output_np
                        print("vitisai_ep---->", hello_world_golden, flush=True)

                        self.run(["echo", "###cpu_runner_tensor###"])
                        graph_runner_env["USE_CPU_RUNNER"] = "1"
                        self.run_command(
                            cmd,
                            graph_runner_env,
                        )
                        self._get_model_result_md5(self._get_model_result("cpu_runner"))
                        self.run(
                            ["echo", "cpu_runner output tensor: %s" % self._output]
                        )
                        self._golden = self._output
                        cpu_runner_output = self._output
                        self._cpu_runner_output_np = self._output_np
                        self._output = vitisai_output
                        if self._golden == self._output:
                            self.run(["echo", "vitisai_ep md5 same with cpu_runner."])
                        else:
                            self.run(
                                ["echo", "vitisai_ep md5 mismatch with cpu_runner."]
                            )

                        if (
                            self._golden != self._output
                            or self._model_env.get("LAYER_COMPARE", "false") == "true"
                        ):
                            self.test_xmodel_diff()

                        # self.set_cache_dir(self._json["id"])
                        if (
                            self._model_env.get("TEST_HELLO_WORLD", "true") == "true"
                            and self.accuracy_input()
                        ):
                            self.run(["echo", "###test_hello_world###"])
                            self.test_hello_world(hello_world_golden)

                if test_mode == "fast_accuracy":
                    self.step_fast_accuracy_onnx()

                elif "accuracy" in test_mode or test_mode == "functionality":
                    tool_name = "test_onnx_runner"
                    if self._workspace.is_windows():
                        tool_name = "test_onnx_runner.exe"
                    self._test_tool = str(
                        self._workspace.install_prefix() / "bin" / tool_name
                    )
                    if "cpu_ep" in self.acc_test_models:
                        self._run_cpu_ep()
                        self._cpu_ep_output_np = self._output_np
                    if "cpu_runner" in self.acc_test_models:
                        self._run_cpu_runner()
                        self._cpu_runner_output_np = self._output_np
                    if "onnx_ep" in self.acc_test_models:
                        self._run_onnx_ep()
                        self._onnx_ep_output_np = self._output_np
                        if (
                            self._model_env.get("TEST_HELLO_WORLD", "true") == "true"
                            and self.accuracy_input()
                        ):
                            self.run(["echo", "###test_hello_world###"])
                            self.test_hello_world(self._golden_np)

                self._post_process_result()

            except subprocess.CalledProcessError as e:
                logging.info(
                    "model " + self._json["id"] + " compile fail:CalledProcessError"
                )
                tb = traceback.format_exc()
                if tb is not None:
                    print(tb, flush=True)
                self._json["result"] = "FAILED@compile_compile_onnx"
            except FileNotFoundError as e:
                logging.info(
                    "model " + self._json["id"] + " compile fail:FileNotFoundError"
                )
                tb = traceback.format_exc()
                if tb is not None:
                    print(tb, flush=True)
                self._json["result"] = "FAILED@compile_compile_onnx"
            except subprocess.TimeoutExpired as e:
                logging.info(
                    "model " + self._json["id"] + " compile subprocess TimeoutExpired"
                )
                tb = traceback.format_exc()
                if tb is not None:
                    print(tb, flush=True)
                self._json["result"] = "FAILED@Timeout"
            except Exception as e:
                tb = traceback.format_exc()
                if tb is not None:
                    print(tb, flush=True)

    def get_test_cmd(self, cpu_ep_flag=False, use_float_model_flag=False):
        onnx_name = self.onnx_model().name
        quantized_onnx_model = self.quantized_onnx_model().name
        set_level0 = self._model_env.get("CI_ONNXRUNTIME_LEVEL0", "true")
        cmd = [self._test_tool]
        if cpu_ep_flag:
            cmd.append("-n")
        if (
            set_level0 == "true"
            or self._model_env.get("CI_ONNXRUNTIME_LEVEL", "") != ""
        ):
            level = self._model_env.get("CI_ONNXRUNTIME_LEVEL", "0")
            cmd.extend(["-l", level])
        custom_op_dll = self._model_env.get("CUSTOM_OP_DLL_PATH", "")
        if custom_op_dll != "":
            cmd.extend(["-c", custom_op_dll])
        # cpu_ep_float = self._model_env.get("CPU_EP_FLOAT", "true")
        if cpu_ep_flag and use_float_model_flag and self.float_onnx():
            float_onnx = self.float_onnx().name
            cmd.append(float_onnx)
        elif cpu_ep_flag and not use_float_model_flag:
            cmd.append(quantized_onnx_model)
        else:
            cmd.append(onnx_name)

        if self.get_onnx_input():
            self.onnx_input_list(self.get_onnx_input(), cmd)

        return cmd

    @time_logged
    def _run_cpu_ep(self):
        onnx_name = self.onnx_model().name
        output_checking_env = self._model_env.get("OUTPUT_CHECKING")
        if output_checking_env == "true":
            output_checking = "cpu_ep,onnx_ep"
        else:
            output_checking = output_checking_env
        self._clean_io()
        dict = {"USE_CPU_RUNNER": "0"}
        self.update_env(dict)

        def cpu_ep_executor(as_gloden=False, use_float_model_flag=False):
            if use_float_model_flag:
                self.run(["echo", "###cpu_ep_tensor with float model###"])
            else:
                self.run(["echo", "###cpu_ep_tensor with quantized model###"])
            cmd = self.get_test_cmd(
                cpu_ep_flag=True, use_float_model_flag=use_float_model_flag
            )
            self.run_command(cmd)
            self._get_model_result_md5(self._get_model_result("cpu_ep_float"))
            self._cpu_ep_output_np = self._output_np
            self.run(["echo", "cpu_ep output tensor: %s" % self._output])
            if as_gloden:
                self._golden = self._output
                self._golden_np = self._output_np

        use_cpuep_golden = os.environ.get("USE_CPUEP_GOLDEN", "true")
        use_float_model_flag = self._model_env.get("CPU_EP_FLOAT", "true") == "true"

        if output_checking == "cpu_ep,cpu_ep":
            cpu_ep_executor(as_gloden=True, use_float_model_flag=True)
            cpu_ep_executor(as_gloden=False, use_float_model_flag=False)
        else:
            if self.get_golden_files() and use_cpuep_golden == "true":
                self.run(["echo", "###cpu_ep_tensor from golden file###"])
                self._get_model_result_md5(self.onnx_golden_list())
                self.run(["echo", "cpu_ep output tensor: %s" % self._output])
                self._golden = self._output
                self._golden_np = self._output_np
                # if len(self.acc_test_models) == 1:
                cpu_ep_executor(
                    as_gloden=False, use_float_model_flag=use_float_model_flag
                )
            else:
                cpu_ep_executor(
                    as_gloden=True, use_float_model_flag=use_float_model_flag
                )

    @time_logged
    def _run_cpu_runner(self):
        onnx_name = self.onnx_model().name
        self.run(["echo", "###cpu_runner_tensor###"])
        self._clean_io()
        dict = {"USE_CPU_RUNNER": "1"}
        self.update_env(dict)
        cmd = self.get_test_cmd()
        self.run_command(cmd)
        self._get_model_result_md5(self._get_model_result("cpu_runner"))
        self._cpu_runner_output_np = self._output_np
        self.run(["echo", "cpu_runner output tensor: %s" % self._output])
        output_checking_env = self._model_env.get("OUTPUT_CHECKING")
        if output_checking_env == "true":
            output_checking = "cpu_ep,onnx_ep"
        else:
            output_checking = self._model_env.get("OUTPUT_CHECKING", "cpu_ep,onnx_ep")
        if output_checking.find("cpu_ep") == -1:
            self._golden = self._output
            self._golden_np = self._output_np

    @time_logged
    def _run_onnx_ep(self):
        onnx_name = self.onnx_model().name
        self._clean_io()
        self.run(["echo", "###vitisai_ep_tensor###"])
        dict = {"USE_CPU_RUNNER": "0"}
        self.update_env(dict)
        cmd = self.get_test_cmd()
        self.run_command(cmd)
        self._get_model_result_md5(self._get_model_result("vitisai_ep"))
        self._onnx_ep_output_np = self._output_np
        self.run(["echo", "onnx_ep output tensor: %s" % self._output])

    @time_logged
    def step_fast_accuracy_onnx(self):
        onnx_name = self.onnx_model().name
        run_times = self._model_env.get("VART_PERF_ACCURACY_CYCLES", "1")

        tool_name = "vart_perf.exe"
        test_tool = str(self._workspace.install_prefix() / "bin" / tool_name)
        current_directory = self.build_dir()
        accuracy_combined_json = (
            f"{current_directory}/{self.name()}_accuracy_group.json"
        )
        vart_cmd = [
            test_tool,
            "-o",
            onnx_name,
            "-r",
            run_times,
            "--group_run",
            accuracy_combined_json,
        ]
        print(f"[CI][Fast_accuracy]Fast_accuracy vart_cmd is {vart_cmd}", flush=True)
        self.run(["echo", "###Fast_accuracy start###"])
        self.run_command(vart_cmd)
        self.run(["echo", "###Fast_accuracy finish###"])

        input_log_path = os.path.join(os.getcwd(), "build.log")
        output_log_path = os.path.join(os.getcwd(), "group_run_log.txt")
        start_marker = "###Fast_accuracy start###"
        end_marker = "###Fast_accuracy finish###"

        fast_accuracy_log_parse.extract_log_segment(
            input_log_path, output_log_path, start_marker, end_marker
        )

        print(
            f"[CI][Fast_accuracy]Extracted log segment has been saved to {output_log_path}"
        )
        log_file = "group_run_log.txt"
        result = fast_accuracy_log_parse.parse_log(log_file)
        with open("fast_accuracy_summary.json", "w") as outfile:
            json.dump(result, outfile, indent=4)

    @time_logged
    def _post_process_result(self):
        onnx_name = self.onnx_model().name
        test_mode = self._model_env.get("TEST_MODE", "performance")
        default_output_checking = {"performance": "cpu_runner,onnx_ep"}
        output_checking_env = self._model_env.get("OUTPUT_CHECKING")
        if output_checking_env == "true":
            output_checking = default_output_checking.get(test_mode, "cpu_ep,onnx_ep")
        else:
            output_checking = (
                output_checking_env
                if output_checking_env
                else default_output_checking.get(test_mode, "cpu_ep,onnx_ep")
            )
        # output_checking = self._model_env.get("OUTPUT_CHECKING", "cpu_runner,onnx_ep")
        cache_dir = self.cache_dir()
        self.copy_files(onnx_name, cache_dir)
        if os.path.exists(cache_dir / "context.json"):
            self._json["context"] = self._workspace.load_json(
                cache_dir / "context.json"
            )
        else:
            self._json["context"] = {}
        self._json["result_run_onnx"] = "OK"
        if "stacks" in self._json["context"]:
            print(
                "FAILED@stacks in context.json, ignore post process result!", flush=True
            )
            self._json["result_compile_onnx"] = "FAILED"
            for s in self._json["context"]["stacks"]:
                if not s.startswith("google") and not s.startswith("xir"):
                    self._json["result"] = "FAILED@xcompiler_" + s

                    break
        else:
            self._json["result_compile_onnx"] = "OK"
            # cpu runner only,
            is_dump = self._model_env.get("XLNX_ENABLE_DUMP", "0")
            if is_dump != "0" and (
                "onnx_ep" in output_checking
                or "cpu_runner" in output_checking
                or (output_checking == "cpu_ep,cpu_ep")
                or ("golden_file" in output_checking)
                or (self._model_env.get("TEST_HELLO_WORLD", "true") == "true")
            ):
                result_compile_text = parse_result.parse_compile(
                    self.log_file(), output_checking
                )
                if (
                    output_checking == "cpu_ep,cpu_ep"
                    or output_checking == "golden_file,cpu_ep"
                    or output_checking == "cpu_ep,golden_file"
                ):
                    self._json["result_compile_onnx"] = "NA"
                    self._json["result_onnx_ep"] = "NA"
                    self._json["result"] = "OK"
                elif result_compile_text == "ERROR":
                    self._json["result_compile_onnx"] = "OK"
                    self._json["result_onnx_ep"] = "FAILED@Unexcepted exception"
                    self._json["result"] = "FAILED@onnx_ep"
                elif result_compile_text != "PASS":
                    self._json["result_compile_onnx"] = result_compile_text
                    self._json["result"] = result_compile_text
                    self._json["result_onnx_ep"] = "OK"
                else:
                    self._json["result_compile_onnx"] = "OK"
                    self._json["result_onnx_ep"] = "OK"
                    self._json["result"] = "OK"

                result_other_text = parse_result.parse_other(
                    self.log_file(), output_checking
                )
                if result_other_text != "PASS":
                    self._json["result_onnx_ep"] = result_other_text
                    self._json["result"] = result_other_text
                # self._get_model_result_md5()  # get dumped binary output
                if self._json["result"] == "OK":
                    self._update_ref_using_golden()  # if dumped binary mismatched, update result to fail
                else:
                    print(
                        "%s, ignore post process result!" % self._json["result"],
                        flush=True,
                    )
                if output_checking == "cpu_runner":
                    self._update_golden()  # if no golden exists, update the golden to model_zoo.json
        logging.info("model " + self._json["id"] + " compile pass")

    def _step_clear_cache(self):
        cache_dir = self.cache_dir()
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)

    def _clean_dump_files(self):
        files_to_clean = [
            x
            for x in os.listdir(self.build_dir())
            if x.endswith("bin") or x.endswith("json") or x.endswith("pdb")
        ]
        try:
            for f in files_to_clean:
                os.remove(os.path.join(self.build_dir(), f))
        except Exception as e:
            print(e)

    @time_logged
    def _step_show_opset(self):
        with shell.Cwd(self, self.build_dir()):
            onnx_filename = self.build_dir() / self.onnx_model().name
            try:
                import onnx  # you may need to pip install onnx

                model = onnx.load(onnx_filename)
                print(
                    f"""{onnx_filename} ir={model.ir_version} opset=[{",".join([str(i.version) for i in model.opset_import])}]  by {model.producer_name} ver {model.producer_version}"""
                )
                self._json["opset"] = [str(i.version) for i in model.opset_import]
                self._json["ir"] = str(model.ir_version)
                self._json["producer_name"] = str(model.producer_name)
                self._json["producer_version"] = str(model.producer_version)
            except:
                self._json["comment"] = str(traceback.format_exc())
                self._json["enable"] = False
                self._json["result"] = "FAILED@show_opset"

    @time_logged
    def _step_compare_with_ref(self):
        _impl.compare_with_ref(self._json, self._workspace._ref_json)

    @time_logged
    def _step_run_on_board(self):
        logging.info("TODO")

    @time_logged
    def _clean_io(self):
        result_json = self.build_dir() / "io.json"
        if os.path.exists(result_json):
            files = json.load(open(result_json))
            for f in files["output"]:
                os.remove(self.build_dir() / f)
                print("Note: clean %s" % str(self.build_dir() / f), flush=True)
            os.remove(result_json)
            print("Note: clean %s" % str(result_json), flush=True)

    def _get_model_result(self, ep):
        files = []
        src_dir = self.build_dir()
        dest_dir = self.build_dir() / ep
        if self._model_env.get("CI_CACHE_OUTPUT", "false") == "true":
            dest_dir = self.cache_dir() / ep
        result_json = src_dir / "io.json"
        if self.get_onnx_input() and isinstance(self.get_onnx_input(), list):
            base_name = os.path.basename(self.get_onnx_input()[0])
            self._input_prefix = base_name.split(":")[0]
        if os.path.exists(result_json):
            if not os.path.isdir(dest_dir):
                os.makedirs(dest_dir)
            self.copy_dump_files(src_dir, dest_dir)
            print(
                "Note: copy %s to %s" % (str(result_json), str(dest_dir / "io.json")),
                flush=True,
            )
            shutil.copy(result_json, dest_dir / "io.json")
            file_c = json.load(open(result_json))
            i = 0
            for f in file_c["output"]:
                new_output = dest_dir / f
                if self._input_prefix != "":
                    new_output = dest_dir / f"{self._input_prefix}_output_{i}.bin"
                print(
                    "Note: copy %s to %s" % (str(src_dir / f), str(new_output)),
                    flush=True,
                )
                shutil.copy(src_dir / f, new_output)
                files.append(new_output)
                i += 1

        else:
            print(f"Warning: {result_json} not exist!", flush=True)

        return files

    def _get_model_result_md5(self, outfiles):
        self._output = ""
        self._output_np = []
        all_output_md5sum = []
        output_bin_type = os.environ.get("CI_OUTPUT_BIN_TYPE", "float32")
        logging.info(
            f"Using {output_bin_type} as CI_OUTPUT_BIN_TYPE to read output bin"
        )
        for f in outfiles:
            md5sum = hashlib.md5(open(f, "rb").read()).hexdigest()
            print("Note: %s md5sum %s" % (str(f), md5sum), flush=True)
            np_data = np.fromfile(f, dtype=np.dtype(output_bin_type)).astype(np.float32)
            self._output_np.append(np_data)
            print(np_data.shape, flush=True)
            all_output_md5sum.append(md5sum)
        self._output = "_".join(all_output_md5sum)

    @time_logged
    def _update_golden(self):
        if not hasattr(self, "_golden"):
            logging.info(
                "model "
                + self._json["id"]
                + " ran first time with cpu-runner and result md5sum "
                + self._output
            )
            # save it
            json_file_path = pathlib.Path(__file__).parent.parent / (
                _get_model_zoo_name(self._workspace) + ".json"
            )
            golden_lock.acquire()
            json_file = open(json_file_path, "r+")
            json_array = json.load(json_file)
            json_file.close()
            for j in json_array:
                if j["id"] != self._json["id"]:
                    continue
                j["golden"] = self._output
            json_file.seek(0)
            json.dump(json_array, json_file, indent=4)
            json_file.truncate()
            golden_lock.release()

    @time_logged
    def _update_ref_using_golden(self):
        result = self._result_compare()
        golden_np_jude = self._golden_np
        output_np_jude = self._onnx_ep_output_np
        if (
            "performance" in self._model_env.get("TEST_MODE", "performance")
            and self._model_env.get("TEST_HELLO_WORLD", "true") == "true"
        ):
            golden_np_jude = self._onnx_ep_output_np
            output_np_jude = self._hello_world_output_np

        if len(golden_np_jude) != len(output_np_jude):
            logging.info(
                "golden num %s not match with output num %s"
                % (len(self._golden_np), len(self._output_np))
            )
        elif len(self._golden_np) == 0 and len(self._output_np) > 0:
            self._json["result_run_onnx"] = "OK"
            self._json["result"] = "OK"
        print("result---------------->", self._json["result"])
        logging.info(f"golden md5: {self._golden}")
        logging.info(f"result md5: {self._output}")
        if self._output == "":
            self._json["md5_compare"] = "NO_OUTPUT"
            self._json["result_run_onnx"] = "FAILED"
            self._json["result"] = "FAILED"
            return
        if self._golden == "":
            self._json["md5_compare"] = "NO_GOLDEN"
            return
        if self._golden == "" and not hasattr(self, "_golden"):
            logging.info(
                "ignore compare output md5 of %s, because there is no output golden in json"
                % self._json["id"]
            )
            self._json["md5_compare"] = "NO_OUTPUT"
            return
        if self._golden != "" and self._golden == self._output:
            if self._json["md5_compare"] == "N/A":
                self._json["md5_compare"] = "OK"
            logging.info("%s output md5 compare pass" % self._json["id"])
        else:
            logging.info("%s output md5 compare fail" % self._json["id"])
            self._json["output_md5"] = self._output
            self._json["result_run_onnx"] = "FAILED"
            self._json["md5_compare"] = "MISMATCH"

    def calculate_l2_distance(self, arr1, arr2):
        l2norm = np.linalg.norm(arr1 - arr2)
        return l2norm

    def calculate_cos(self, arr1, arr2):
        try:
            dot_product = np.dot(arr1, arr2)
            norm_arr1 = np.linalg.norm(arr1)
            norm_arr2 = np.linalg.norm(arr2)
            cos_sim = dot_product / (norm_arr1 * norm_arr2)
            return cos_sim
        except ValueError as e:
            print(f"Error: {e}")
            return "NA"

    def calculate_snr(self, arr1, arr2):
        try:
            # print("arr1 --->", arr1, flush=True)
            # print("arr2 --->", arr2, flush=True)
            # mse = np.mean(np.square(arr1 - arr2))
            denominator = np.sum(np.square(arr1 - arr2))
            if denominator == 0:
                return float("inf")
            image_SNR = 10.0 * np.log10(np.sum(np.square(arr1)) / denominator)
            return image_SNR
        except ValueError as e:
            print(f"Error: {e}")
            return "NA"

    def calculate_psnr(self, arr1, arr2):
        try:
            mse = np.mean((arr1 - arr2) ** 2)
            if mse == 0:
                mse = mse + 1e-10  # avoid INF for psnr
            max_pixel_value = np.max(arr1)
            if max_pixel_value <= 0:
                max_pixel_value = 1e-10
            psnr = 20 * np.log10(max_pixel_value) - 10 * np.log10(mse)
            return psnr
        except ValueError as e:
            print(f"Error: {e}")
            return "NA"

    @time_logged
    def _result_compare(self):
        print("run _result_compare", flush=True)
        test_hello_world_env = self._model_env.get("TEST_HELLO_WORLD", "true")
        is_ok = True
        try:
            if (
                "cpu_ep" in self.acc_test_models
                and "golden_file" in self.acc_test_models
            ):
                output_num = len(self._cpu_ep_output_np)
                compare_out_np = self._cpu_ep_output_np
            elif "onnx_ep" in self.acc_test_models:
                output_num = len(self._onnx_ep_output_np)
                compare_out_np = self._onnx_ep_output_np
            else:
                output_num = len(self._output_np)
                compare_out_np = self._output_np
            print("run output_num", output_num, flush=True)
            for i in range(output_num):
                if self._json["id"] in ("F1", "F2", "F1_v0.3", "F2_v0.3"):
                    if "performance" in self._model_env.get("TEST_MODE", "performance"):
                        if (
                            "onnx_ep" in self.acc_test_models
                            and test_hello_world_env == "true"
                        ):
                            l2norm_hw = self.calculate_l2_distance(
                                self._onnx_ep_output_np[i],
                                self._hello_world_output_np[i],
                            )
                            shell.write_to_log(
                                self.log_file(), f"l2norm_hw: {l2norm_hw}"
                            )
                            if f"{i}" not in self._l2norm_hw_dict:
                                self._l2norm_hw_dict[f"{i}"] = [l2norm_hw]
                            else:
                                self._l2norm_hw_dict[f"{i}"].append(l2norm_hw)
                    if self._model_env.get("TEST_MODE", "performance") != "performance":
                        snr = self.calculate_snr(self._golden_np[i], compare_out_np[i])
                        shell.write_to_log(self.log_file(), f"snr: {snr}")
                        if f"{i}" not in self._snr_dict:
                            self._snr_dict[f"{i}"] = [snr]
                        else:
                            self._snr_dict[f"{i}"].append(snr)

                        if (
                            "onnx_ep" in self.acc_test_models
                            and test_hello_world_env == "true"
                        ):
                            snr_hw = self.calculate_snr(
                                self._golden_np[i], self._hello_world_output_np[i]
                            )
                            shell.write_to_log(self.log_file(), f"snr_hw: {snr_hw}")
                            if f"{i}" not in self._snr_hw_dict:
                                self._snr_hw_dict[f"{i}"] = [snr_hw]
                            else:
                                self._snr_hw_dict[f"{i}"].append(snr_hw)
                    continue

                if "performance" in self._model_env.get("TEST_MODE", "performance"):
                    if (
                        "onnx_ep" in self.acc_test_models
                        and test_hello_world_env == "true"
                    ):
                        l2norm_hw = self.calculate_l2_distance(
                            self._onnx_ep_output_np[i], self._hello_world_output_np[i]
                        )
                        shell.write_to_log(self.log_file(), f"l2norm_hw: {l2norm_hw}")
                        if f"{i}" not in self._l2norm_hw_dict:
                            self._l2norm_hw_dict[f"{i}"] = [l2norm_hw]
                        else:
                            self._l2norm_hw_dict[f"{i}"].append(l2norm_hw)

                if self._model_env.get("TEST_MODE", "performance") != "performance":
                    cos = self.calculate_cos(self._golden_np[i], compare_out_np[i])
                    shell.write_to_log(self.log_file(), f"cos: {cos}")
                    if f"{i}" not in self._cos_dict:
                        self._cos_dict[f"{i}"] = [cos]
                    else:
                        self._cos_dict[f"{i}"].append(cos)

                    psnr = self.calculate_psnr(self._golden_np[i], compare_out_np[i])
                    shell.write_to_log(self.log_file(), f"psnr: {psnr}")
                    if f"{i}" not in self._psnr_dict:
                        self._psnr_dict[f"{i}"] = [psnr]
                    else:
                        self._psnr_dict[f"{i}"].append(psnr)

                    l2norm = self.calculate_l2_distance(
                        self._golden_np[i], compare_out_np[i]
                    )
                    shell.write_to_log(self.log_file(), f"l2norm: {l2norm}")
                    if f"{i}" not in self._l2norm_dict:
                        self._l2norm_dict[f"{i}"] = [l2norm]
                    else:
                        self._l2norm_dict[f"{i}"].append(l2norm)

                    if (
                        "onnx_ep" in self.acc_test_models
                        and test_hello_world_env == "true"
                    ):
                        l2norm_hw = self.calculate_l2_distance(
                            self._golden_np[i], self._hello_world_output_np[i]
                        )
                        shell.write_to_log(self.log_file(), f"l2norm_hw: {l2norm_hw}")
                        if f"{i}" not in self._l2norm_hw_dict:
                            self._l2norm_hw_dict[f"{i}"] = [l2norm_hw]
                        else:
                            self._l2norm_hw_dict[f"{i}"].append(l2norm_hw)

                        psnr_hw = self.calculate_psnr(
                            self._golden_np[i], self._hello_world_output_np[i]
                        )
                        shell.write_to_log(self.log_file(), f"psnr_hw: {psnr_hw}")
                        if f"{i}" not in self._psnr_hw_dict:
                            self._psnr_hw_dict[f"{i}"] = [psnr_hw]
                        else:
                            self._psnr_hw_dict[f"{i}"].append(psnr_hw)

        except Exception as e:
            is_ok = False
            tb = traceback.format_exc()
            if tb is not None:
                print(tb, flush=True)
        return is_ok

    def gdb(self):
        self._log_file = None
        self._step_download_onnx()
        with shell.Cwd(self, self.build_dir()):
            onnx_name = self.onnx_model().name
            if self._workspace.is_windows():
                self.run(
                    [
                        sys.executable,
                        pathlib.Path(__file__).parent.parent / "printenv.py",
                    ]
                )
                if shell.is_file(
                    "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Professional\\Common7\\IDE\\devenv.exe"
                ):
                    self.run(
                        [
                            "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Professional\\Common7\\IDE\\devenv.exe",
                            "/DebugExe",
                            self._workspace.install_prefix()
                            / "bin"
                            / "test_onnx_runner.exe",
                            onnx_name,
                        ]
                    )
                elif shell.is_file(
                    "C:\\Program Files (x86)\\Windows Kits\\10\\Debuggers\\x64\\windbg.exe"
                ):
                    self.run(
                        [
                            "C:\\Program Files (x86)\\Windows Kits\\10\\Debuggers\\x64\\windbg.exe",
                            "test_onnx_runner",
                            onnx_name,
                        ]
                    )
                else:
                    logging.error("cannot find debugger")
                    sys.exit(1)
            else:
                self.run(["gdb", "--args", "test_onnx_runner", onnx_name])

    def cache_cleanup(self, name):
        print("cache_cleanup step")
        try:
            onnx_name = self.build_dir() / self.onnx_model().name
            cache_dir = self.cache_dir()
            new_cache_dir = cache_dir
            if self._model_env.get("COPY_XMODEL", "false") == "true":
                self._copy_xmodel(self._get_xmodel("compiled"))
                self._copy_xmodel(self._get_xmodel("xir"))
            if self._model_env.get("CACHE_RENAME", "true") == "true":
                new_cache_dir = os.path.join(os.path.dirname(cache_dir), name)
                if os.path.isdir(cache_dir):
                    if os.path.split(cache_dir)[-1] != name:
                        os.rename(cache_dir, new_cache_dir)
                    else:
                        # if cache dir is model name, move files from md5 dir
                        cache_sub_dirs = [
                            x
                            for x in os.listdir(cache_dir)
                            if os.path.isdir(os.path.join(cache_dir, x))
                        ]
                        assert len(cache_sub_dirs) == 1
                        for _file in os.listdir(
                            os.path.join(cache_dir, cache_sub_dirs[0])
                        ):
                            shutil.move(
                                os.path.join(cache_dir, cache_sub_dirs[0], _file),
                                os.path.join(cache_dir, _file),
                            )

            if self._clean:
                logging.info("clean all cache %s" % str(new_cache_dir))
                shutil.rmtree(new_cache_dir)
                logging.info("pass clean onnx model %s" % str(onnx_name))
                os.remove(onnx_name)
        except Exception as e:
            print("ERROR: cache_cleanup error: %s" % e)

    def get_snr_mean(self, text="snr"):
        _dict_map = {"snr": self._snr_dict, "snr_hw": self._snr_hw_dict}
        print('self._json["id"]', self._json["id"], flush=True)
        if self._model_env.get(
            "TEST_MODE", "performance"
        ) != "performance" and self._json["id"] not in (
            "F1",
            "F2",
            "F1_v0.3",
            "F2_v0.3",
        ):
            return
        try:
            snr_target = self.snr_target()
            snr_str = ""
            is_ok = self._json["result"]
            snr_g = 25.0
            snr_list = snr_target.split("/") if snr_target else []
            for key, val in _dict_map.get(text).items():
                if self.key_output_id() and key not in self.key_output_id():
                    continue
                i = int(key)
                snr_mean = abs(float(np.mean(val)))
                if snr_target:
                    if i < len(snr_list):
                        snr_g = float(snr_list[i])
                else:
                    self._json["snr_target"] = snr_g
                if is_ok == "OK":
                    self._json["result"] = (
                        "OK" if abs(snr_mean) > abs(snr_g) else "FAILED"
                    )
                snr = f"{snr_mean:.4f}"
                shell.write_to_log(
                    self.log_file(), f"{text} mean: {snr}, snr target: {snr_g}"
                )
                snr_str = f"{snr}" if snr_str == "" else f"{snr_str}/{snr}"
            self._json[text] = snr_str
        except Exception as e:
            print("ERROR: get snr mean error: %s" % e)
            self._json[text] = "NA"

    def get_psnr_mean(self, text="psnr"):
        _dict_map = {"psnr": self._psnr_dict, "psnr_hw": self._psnr_hw_dict}
        if self._json["id"] in ("F1", "F2", "F1_v0.3", "F2_v0.3"):
            return
        try:
            psnr_target = self.psnr_target()
            psnr_str = ""
            is_ok = self._json["result"]
            psnr_g = 20.0
            # print(self._psnr_dict, flush=True)
            psnr_list = psnr_target.split("/") if psnr_target else []
            for key, val in _dict_map.get(text).items():
                i = int(key)
                psnr_mean = abs(float(np.mean(val)))
                if psnr_target:
                    if i < len(psnr_list):
                        psnr_g = float(psnr_list[i])
                else:
                    self._json["psnr_target"] = psnr_g
                if self._json["result"] == "OK":
                    self._json["result"] = (
                        "OK" if abs(psnr_mean) > abs(psnr_g) else "FAILED"
                    )
                psnr = f"{psnr_mean:.4f}"
                shell.write_to_log(
                    self.log_file(), f"{text} mean: {psnr}, psnr target: {psnr_g}"
                )
                psnr_str = f"{psnr}" if psnr_str == "" else f"{psnr_str}/{psnr}"
            self._json[text] = psnr_str
        except Exception as e:
            print("ERROR: get psnr mean error: %s" % e)
            self._json[text] = "NA"

    def get_l2norm_mean(self, text="l2norm"):
        _dict_map = {"l2norm": self._l2norm_dict, "l2norm_hw": self._l2norm_hw_dict}
        if self._model_env.get(
            "TEST_MODE", "performance"
        ) != "performance" and self._json["id"] in ("F1", "F2", "F1_v0.3", "F2_v0.3"):
            return
        try:
            l2norm_target = self.l2norm_target()
            l2norm_str = ""
            l2norm_g = 0.1
            l2norm_list = l2norm_target.split("/") if l2norm_target else []
            l2_str_dict = {}
            for key, val in _dict_map.get(text).items():
                i = int(key)
                if text == "l2norm":
                    val_str_list = [f"{j:.4f}" for j in val]
                    l2_str_dict[key] = val_str_list
                l2norm_mean = float(np.mean(val))
                if l2norm_target:
                    if i < len(l2norm_list):
                        l2norm_g = float(l2norm_list[i])
                else:
                    self._json["l2norm_target"] = l2norm_g
                if self._json["result"] == "OK":
                    self._json["result"] = "OK" if l2norm_mean < l2norm_g else "FAILED"
                l2norm = f"{l2norm_mean:.4f}"
                shell.write_to_log(
                    self.log_file(), f"{text} mean: {l2norm}, l2norm target: {l2norm_g}"
                )
                l2norm_str = (
                    f"{l2norm}" if l2norm_str == "" else f"{l2norm_str}/{l2norm}"
                )
            self._json[text] = l2norm_str
            print("l2norm_str", l2norm_str, flush=True)
            shell.write_to_log(self.log_file(), f"{text}_log: {l2norm_str}")
            if text == "l2norm":
                self._json["l2norm_detail"] = l2_str_dict
        except Exception as e:
            print("ERROR: get l2norm mean error: %s" % e)
            self._json[text] = "NA"
            if text == "l2norm":
                self._json["l2norm_detail"] = "NA"

    def convert_log2bat(self):
        try:
            build_log = os.path.join(self.build_dir(), "build.log")
            convert.main(build_log)
        except Exception as e:
            print(e, flush=True)
            tb = traceback.format_exc()
            if tb is not None:
                print(tb, flush=True)

    def steps(self):
        try:
            cwd = os.getcwd()
            os.chdir(self.build_dir())
            self._model_env = self._workspace.environ().copy()
            if self._workspace.is_windows():
                self._local_input_path = pathlib.Path("C:\\") / "win24_models_input"
                self._local_golden_path = pathlib.Path("C:\\") / "win24_models_golden"

            if self.vaip_env():
                self.update_env(self.vaip_env())

            custom_op_dll = self._model_env.get("CUSTOM_OP_DLL_PATH", "")
            custom_op_base = os.path.basename(custom_op_dll)
            if custom_op_dll != "":
                shutil.copy(pathlib.Path(custom_op_dll), self.build_dir())
                self._model_env["CUSTOM_OP_DLL_PATH"] = f".\\{custom_op_base}"

            self.write_new_vaip_config()
            if not self._print_env:
                self._run_print_env()

            self._json["result"] = "OK"
            self._json["result_download_onnx"] = "N/A"
            self._json["result_compile_onnx"] = "N/A"
            self._json["md5_compare"] = "N/A"
            accuracy_input_dict = {}
            org_workspace = self._model_env.get("WORKSPACE", "")
            self.set_cache_dir(self._json["id"])
            if self.accuracy_input():
                accuracy_input_json = open(
                    pathlib.Path(org_workspace) / "ci" / self.accuracy_input()
                )
                accuracy_input_dict = json.load(accuracy_input_json)
                accuracy_input_json.close()
            is_accuracy = (
                self._model_env.get("TEST_MODE", "performance") != "functionality"
                and self._model_env.get("TEST_MODE", "performance") != "fast_accuracy"
                and (
                    self._model_env.get("ACCURACY_TEST", "false") == "true"
                    or "accuracy" in self._model_env.get("TEST_MODE", "performance")
                )
            )
            fast_accuracy = (
                self._model_env.get("TEST_MODE", "performance") == "fast_accuracy"
            )
            if self.accuracy_input() and fast_accuracy:
                accuracy_golden_json_path = (
                    pathlib.Path(org_workspace) / "ci" / self.accuracy_golden()
                )
                print(
                    f"[CI][Combine_json_for_vart]golden json : {self.accuracy_golden()}"
                )
                accuracy_input_json_path = (
                    pathlib.Path(org_workspace) / "ci" / self.accuracy_input()
                )
                print(
                    f"[CI][Combine_json_for_vart]input json : {self.accuracy_input()}"
                )
                current_directory = self.build_dir()
                accuracy_combined_json = (
                    f"{current_directory}/{self.name()}_accuracy_group.json"
                )
                print(f"[CI][Combine_json_for_vart]combined json : {self.name()}")
                combine_json_for_vart.combine_json_for_vart(
                    accuracy_input_json_path,
                    accuracy_golden_json_path,
                    accuracy_combined_json,
                )

                # to avoid set_onnx_input return none value
                first_key, first_val = next(iter(accuracy_input_dict.items()))
                self._seq = first_key
                self.set_onnx_input(first_val)
                self.run_steps()

            elif self.accuracy_input() and is_accuracy:
                accuracy_golden_dict = {}
                if self.accuracy_golden():
                    accuracy_golden_json = open(
                        pathlib.Path(org_workspace) / "ci" / self.accuracy_golden()
                    )
                    accuracy_golden_dict = json.load(accuracy_golden_json)
                    accuracy_golden_json.close()
                for key, val in accuracy_input_dict.items():
                    self._seq = key
                    self.set_onnx_input(val)
                    if key in accuracy_golden_dict.keys():
                        self.set_golden_files(accuracy_golden_dict[key])
                    self.run_steps()
                    self._first_performance = False
            else:
                if accuracy_input_dict:
                    self.set_onnx_input(accuracy_input_dict[self._seq])
                self.run_steps()
            self._json["accuracy_input"] = accuracy_input_dict
            self.dump_inst_data()

            self.cache_cleanup(self._json["id"])

            self.get_snr_mean("snr")
            self.get_snr_mean("snr_hw")
            self.get_psnr_mean("psnr")
            self.get_psnr_mean("psnr_hw")
            self.get_l2norm_mean("l2norm")
            self.get_l2norm_mean("l2norm_hw")

            self.convert_log2bat()
            os.chdir(cwd)
        except Exception as e:
            tb = traceback.format_exc()
            if tb is not None:
                print(tb, flush=True)

    def run_steps(self):
        if int(self._model_env.get("CLEAR_CACHE", "0")):
            step_list = [
                "_step_download_onnx",
                "_step_compile_onnx",
                "_step_clear_cache",
            ]
        elif self._model_env.get("USER_RUN_MODE"):
            step_list = [
                f"_step_{each}_onnx"
                for each in self._model_env.get("USER_RUN_MODE").split(",")
            ]
        elif (
            self._model_env.get("USE_EP_CONTEXT", "true") == "true"
            and self._model_env.get("MODEL_TYPE", "onnx") == "onnx"
        ):
            step_list = [
                "_step_download_onnx",
                "_step_gen_ep_context",
                "_step_compile_onnx",
            ]
        else:
            step_list = ["_step_download_onnx", "_step_compile_onnx"]
        for step in step_list:
            getattr(self, step)()


def create_with_ids(workspace, jsonArray, ids):
    recipe = []
    idSet = set(ids)
    print(idSet, flush=True)
    ret = [
        VaipModelRecipe(workspace, item)
        for item in jsonArray
        if item["id"] in idSet
        and item.get("onnx_model", "") != ""
        and check_run(item.get("id"))
    ]
    for r in ret:
        idSet.remove(r.name())
    if not len(idSet) == 0:
        logging.fatal(f"cannot find case {idSet}")
    return ret


def check_run(model_name):
    incremental_mode = os.environ.get("INCREMENTAL_TEST", "false")
    run_list, skip_list = read_control_file()

    workspace_home = (
        pathlib.Path.home()
        if not os.environ.get("JOB_BASE_NAME", "")
        else pathlib.Path(r"C:\\IPU_workspace") / os.environ["JOB_BASE_NAME"]
    )

    log_file = workspace_home / "vaip_regression" / model_name / "build.log"
    run_flag = (
        incremental_mode == "true" and not pathlib.Path.is_file(log_file)
    ) or incremental_mode == "false"

    if len(run_list) > 0 and model_name not in run_list:
        run_flag = False
    if model_name in skip_list:
        run_flag = False
    return run_flag


def read_control_file():
    run_list = []
    skip_list = []
    control_file = os.environ.get("USER_CONTROL_FILE", "")
    if control_file != "":
        with open(control_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                tmp_list = line.strip().split(",")
                if tmp_list[0] == "SKIP":
                    skip_list.append(tmp_list[1])
                else:
                    run_list.append(tmp_list[1])
        f.close()
    return run_list, skip_list


def create_all_recipes(workspace, jsonArray):
    forbid_list = (
        workspace._environ.get("FORBID_CASE").split(" ")
        if workspace._environ.get("FORBID_CASE")
        else []
    )
    return [
        VaipModelRecipe(workspace, item)
        for item in jsonArray
        if item.get("onnx_model", "") != ""
        and item.get("enable", True)
        and item.get("id") not in forbid_list
        and check_run(item.get("id"))
    ]


def create_receipes_from_args(workspace, arg):
    jsonFilePath = pathlib.Path(__file__).parent.parent / (
        _get_model_zoo_name(workspace) + ".json"
    )
    jsonFile = open(jsonFilePath)
    jsonArray = json.load(jsonFile)
    jsonFile.close()
    # single case
    if hasattr(arg, "case") and arg.case:
        return create_with_ids(workspace, jsonArray, arg.case)
    # all cases
    else:
        return create_all_recipes(workspace, jsonArray)
