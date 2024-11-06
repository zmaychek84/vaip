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
import argparse
import re
import hashlib
import traceback
import logging
from pathlib import Path
from collections import OrderedDict

from . import utility


def is_match(results):
    cpu_top1 = results.get("cpu", [])[0]
    cpu_runner_top1 = results.get("cpu_runner", [])[0]
    ipu_top1 = results.get("ipu", [])[0]

    # step 1: check if ONNX ep is all the same with cpu_runer
    onnx_diff_cpurunner_flag = False
    if (
        cpu_runner_top1.get("lable", "") != ipu_top1.get("lable", "")
        or cpu_runner_top1.get("text", "") != ipu_top1.get("text", "")
        or cpu_runner_top1.get("score", "") != ipu_top1.get("score", "")
    ):
        # print('ERROR: ONNX ep not same with cpu_runer')
        onnx_diff_cpurunner_flag = True
    # step 2: check if ONNX ep with CPU EP
    if cpu_top1.get("lable", "") != ipu_top1.get("lable", ""):
        # print('ERROR Type2: ONNX EP lable not match  CPU EP')
        onnx_diff_cpu_flag = True
    elif cpu_top1.get("score") != cpu_runner_top1.get("score"):
        onnx_diff_cpu_flag = True
    else:
        onnx_diff_cpu_flag = False

    return onnx_diff_cpurunner_flag, onnx_diff_cpu_flag


def get_result(type, results_dict=None, output_lines=None):
    if results_dict is None:
        results_dict = {}
    pattern = (
        r".*?score\[(\d+)\].*?(\d\.\d+|1|0|-nan\(ind\)|nan|\d+\.\d+e-\d+)\s+text:\s(.*)"
    )
    for line in output_lines:
        m = re.match(pattern, line)
        if not m:
            continue
        results_dict[type] = (
            [] if not results_dict.get(type, []) else results_dict[type]
        )
        lable, score, text = m.group(1), m.group(2), m.group(3).strip(", ")
        results_dict[type].append(
            {"score": score, "lable": lable, "text": text, "line": line}
        )


def parse_compile(result_file, output_checking):
    return_txt = "PASS"
    try:
        # logging.info(f"output_checking: {output_checking}")
        if not os.path.join(result_file):
            print("ERROR:no result file, %s" % result_file)
            return_txt = "ERROR:no result file, %s" % result_file
        result_lines = utility.read_lines(result_file)
        done_state = False
        begin_state = False
        error_state = False
        not_support = False
        x_pattern = r".*\sDPU\ssubgraph\snumber\s(\d+)"
        v_pattern = r".*\[XLNX_ONNX_EP_VERBOSE\]\sdpu\ssubgraph:\s(\d+)"
        xcompiler_dpu_num = 0
        on_ipu_dpu_num = 0
        for line in result_lines:
            if "Begin to compile" in line:
                begin_state = True
            elif "Compile done" in line:
                done_state = True
            elif "[VITIS AI EP] This model is not a supported CNN model" in line:
                not_support = True
            elif line.lower().find("exception") != -1:
                error_state = True
            elif line.lower().find("Failed") != -1:
                error_state = True
            m = re.match(x_pattern, line)
            if m:
                xcompiler_dpu_num = int(m.group(1))
            m = re.match(v_pattern, line)
            if m:
                on_ipu_dpu_num = int(m.group(1))

        if output_checking != "cpu_ep" and not_support:
            return_txt = "number of conv <=1"
        elif output_checking != "cpu_ep" and not begin_state:
            return_txt = "FAILED@generate_xir_xmodel"
        elif output_checking != "cpu_ep" and (not done_state or xcompiler_dpu_num == 0):
            return_txt = "FAILED@xcompiler_compile_xmodel"
        elif error_state or on_ipu_dpu_num == 0:
            return_txt = "ERROR"
    except Exception as e:
        print(e, flush=True)
    return return_txt


def parse_other(result_file, output_checking):
    # print(f"output_checking======>{output_checking}")
    return_txt = "PASS"
    if not os.path.join(result_file):
        print("ERROR:no result file, %s" % result_file)
        return_txt = "ERROR:no result file, %s" % result_file
    result_lines = utility.read_lines(result_file)
    dpu_timeout = False
    for line in result_lines:
        if re.match(r".*DPU timeout:\s.*", line):
            dpu_timeout = True
    if dpu_timeout:
        return_txt = "FAILED@DPU Timeout"
    return return_txt


def parse_result(result_file):
    if not os.path.join(result_file):
        print("ERROR:no result file, %s" % result_file)
        return 1
    result_lines = utility.read_lines(result_file)

    # split all result lines with key word
    ipu_start = (
        ipu_end
    ) = (
        cpu_ep_start
    ) = (
        cpu_ep_end
    ) = cpu_runner_start = perf_test_start = cpu_runner_end = perf_test_end = 0
    for index, line in enumerate(result_lines):
        if "test_dpu_ep" in line:
            ipu_start = index
        if "test_cpu_ep" in line:
            ipu_end = index - 1
            cpu_ep_start = index
        if "test_cpu_runner" in line:
            cpu_ep_end = index - 1
            cpu_runner_start = index
        if "onnxruntime_perf_test" in line:
            cpu_runner_end = index - 1
            perf_test_start = index
    if not cpu_runner_end:
        assert not perf_test_start
        cpu_runner_end = -3
    # get every type result
    results = {}
    if ipu_start and ipu_end:
        get_result("ipu", results, result_lines[ipu_start:ipu_end])
    if cpu_ep_start and cpu_ep_end:
        get_result("cpu", results, result_lines[cpu_ep_start:cpu_ep_end])
    if cpu_runner_start and cpu_runner_end:
        get_result("cpu_runner", results, result_lines[cpu_runner_start:cpu_runner_end])

    # check if all typy have correct result
    if not results.get("cpu", []):
        print("FAILED: result parse fail(CPU_EP Error)")
        return 1
    if results.get("cpu_runner", []) and (not results.get("ipu", [])):
        print("FAILED: result parse fail(IPU Timeout)")
        return 1
    if not results.get("cpu_runner", []):
        if not result_lines[ipu_start:ipu_end]:
            print("FAILED: result parse fail(NO IPU Result)")
            return 1
        for line in result_lines[ipu_start:ipu_end]:
            if "Compile done" in line:
                print("'FAILED: result parse fail(VitisAI EP Error)")
                return 1
        else:
            print("FAILED: result parse fail(xcompiler failed)")
            return 1

    # check if result is match
    onnx_diff_cpurunner_flag, onnx_diff_cpu_flag = is_match(results)
    if onnx_diff_cpurunner_flag:
        print("FAILED: mismatch")
        return 1
    elif (not onnx_diff_cpurunner_flag) and onnx_diff_cpu_flag:
        print("PASS: rounding mode")
        return 0
    elif (not onnx_diff_cpurunner_flag) and (not onnx_diff_cpu_flag):
        # CPU ALL
        for line in result_lines[ipu_start:ipu_end]:
            if "Catch fatal exception" in line:
                print("PASS: CPU ALL:xcompiler check fatal")
                return 1
        else:
            print("PASS: Fall Back CPU:dpu subgraph>2")
            return 2
    else:
        print("ERROR: unknown match flag")
        return 1


if __name__ == "__main__":
    log_file = sys.argv[1]
    return_code = parse_result(log_file)
    sys.exit(return_code)
