##
##  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc.
##
##  Licensed under the Apache License, Version 2.0 (the "License");
##  you may not use this file except in compliance with the License.
##  You may obtain a copy of the License at
##
##  http://www.apache.org/licenses/LICENSE-2.0
##
##  Unless required by applicable law or agreed to in writing, software
##  distributed under the License is distributed on an "AS IS" BASIS,
##  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
##  See the License for the specific language governing permissions and
##  limitations under the License.
##
import sys
import os
import subprocess


def cmd_execute(args, run_env=None, capture_output=False):
    try:
        print("Run cmd: %s" % " ".join(args))
        output = subprocess.run(
            args, shell=False, check=True, env=run_env, capture_output=capture_output
        )
        return output
    except Exception as e:
        print("ERROR: %s" % e)


def scp_add_onnx(vaip_regression_dir):
    if not os.path.exists(vaip_regression_dir):
        os.mkdir(vaip_regression_dir)
    scp_args = [
        "scp",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "xcdl190074.xilinx.com"
        + ":"
        + "/group/dphi_software/software/workspace/yanjunz/ipu_benchmark/add_only/add_only_quantized.onnx",
        vaip_regression_dir,
    ]
    cmd_execute(scp_args)


def get_sw_rt(voe_path, vaip_config, vaip_regression_dir):
    set_config = f"config_file|{vaip_config}"
    if os.environ.get("ENABLE_CACHE_FILE_IO_IN_MEM", "0") == "1":
        set_config = f"config_file|{vaip_config} enable_cache_file_io_in_mem|1"
    else:
        set_config = f"config_file|{vaip_config} enable_cache_file_io_in_mem|0"
    add_only_onnx = os.path.join(vaip_regression_dir, "add_only_quantized.onnx")
    if not os.path.exists(add_only_onnx):
        scp_add_onnx(vaip_regression_dir)
    if not os.path.exists(add_only_onnx):
        print("ERROR: add_only not exists!")
        return

    vaitrace_env = os.environ.copy()
    if not vaitrace_env.get("XLNX_VART_FIRMWARE", ""):
        print("ERROR: No XLNX_VART_FIRMWARE set!")
        return
    if not vaitrace_env.get("XLNX_TARGET_NAME", ""):
        print("ERROR: No XLNX_TARGET_NAME set!")
        return
    vaitrace_env["PATH"] = (
        f"{voe_path}\\bin;{voe_path}\\python;{voe_path};" + vaitrace_env["PATH"]
    )
    vaitrace_env["OPT_LEVEL"] = "3"
    vaitrace_env["NUM_OF_DPU_RUNNERS"] = "1"
    vaitrace_env["XLNX_MINIMUM_NUM_OF_CONV"] = "0"

    args = [
        os.path.join(voe_path, "python", "python.exe"),
        os.path.join(voe_path, "bin", "vaitrace", "vaitrace.py"),
        "-t",
        "500000",
        "-d",
        "--txt",
        "--fine_grained",
        os.path.join(voe_path, "bin", "onnxruntime_perf_test.exe"),
        "-I",
        "-D",
        "-e",
        "vitisai",
        "-i",
        set_config,
        "-t",
        "20",
        "-c",
        "1",
        f"{add_only_onnx}",
    ]
    try:
        output = cmd_execute(args, vaitrace_env, capture_output=True).stdout.decode(
            "utf-8"
        )
        if "y_QuantizeLinear_Input" not in output:
            output = cmd_execute(args, vaitrace_env, capture_output=True).stdout.decode(
                "utf-8"
            )

        lines = output.split("\n")
        for line in lines:
            if "y_QuantizeLinear_Input" not in line:
                continue

            datas = [x.strip() for x in line.split("|")]
            assert len(datas) == 6
            sw_rt = datas[2]
            print("Software Runtime is %s" % sw_rt)
            return sw_rt
    except Exception as e:
        print("ERROR: %s" % e)


if __name__ == "__main__":
    voe_path = r"C:\yanjunz\voe-vaitrace-1122"
    vaip_config = os.path.join(voe_path, "vaip_config.json")
    vaip_regression_dir = r"C:\Users\deephi\build\vaip_regression"
    get_sw_rt(voe_path, vaip_config, vaip_regression_dir)
