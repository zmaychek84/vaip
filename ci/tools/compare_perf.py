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
import json
import logging
from pathlib import Path
from collections import OrderedDict

from . import utility
from . import constant
from . import parse_ops
from . import compare_with_baseline


key_words = {
    "cpu_ep_latency": "cpu_ep_latency",
    "vitisai_ep_latency": "vitisai_ep_latency",
    "cpu_ep_throughput": "cpu_ep_throughput",
    "vitisai_ep_throughput": "vitisai_ep_throughput",
    "fpstrace": "fpstrace",
    "vaitrace_profiling": "vaitrace_profiling",
    "vitisai_xmodel": "vitisai_xmodel",
}


def get_result_summary(data):
    if os.environ.get("TEST_MODE", "performance").find("perf") == -1:
        return ""
    html = ""
    html += '<h3>Summary:</h3> <table border="1">'
    dic = {}
    for name, item in data.items():
        key = item.get("Summary", {}).get("Functionality", "")

        if key in dic:
            dic[key] += 1
        else:
            dic[key] = 1
    # head
    html += "<tr>"
    html += "<th>Total</th>"
    for val in dic.keys():
        html += f"<th>{val}</th>"
    html += "</tr>"

    # content
    html += "<tr>"
    html += f"<td>{len(data)}</td>"
    for val in dic.keys():
        html += f"<td>{dic[val]}</td>"
    html += "</tr>"
    html += "</table><p></p>"
    return html


def parse_result(result_lines):
    print("start to parse results")
    results = {}
    fps_pattern = r".*Number\sof\sinferences per\ssecond:\s(\d+\.?\d+)"
    latency_pattern = r".*Average\sinference\stime\scost:\s(\d+\.?\d+)\sms"
    xmodel_perf_pattern = (
        r".*Average\sinference\stime\scost\s\(latency\):\s(\d+\.?\d+)\sus"
    )
    ipu_tracing_pattern = r".*Avg\sCU\s\d+\sinference\stime\s\(ms\):\s(.*)\sms.*?"
    result_type = ""

    for line in result_lines:
        for type, key in key_words.items():
            if key in line:
                result_type = type
        m = re.match(fps_pattern, line)
        if m and result_type:
            results[result_type] = (
                {} if not results.get(result_type, {}) else results[result_type]
            )
            fps = m.group(1)
            results[result_type]["fps"] = fps

        a = re.match(latency_pattern, line)
        if a and result_type:
            results[result_type] = (
                {} if not results.get(result_type, {}) else results[result_type]
            )
            latency = a.group(1)
            results[result_type]["latency"] = latency
        x = re.match(xmodel_perf_pattern, line)
        if x and result_type and (not results.get(result_type, None)):
            results[result_type] = (
                {} if not results.get(result_type, {}) else results[result_type]
            )
            latency = x.group(1)
            results[result_type]["latency"] = round(float(latency) / 1000, 4)
        i = re.match(ipu_tracing_pattern, line)
        if i:
            latency = i.group(1)
            results["fpstrace"] = (
                {} if not results.get("fpstrace", {}) else results["fpstrace"]
            )
            results["fpstrace"]["latency"] = round(float(latency), 4)
    return results


def make_empty_model_tr(model_name, col_num, model_results_dict):
    model_results_dict["TEST_STATUS"] = "SKIP"
    model_results_dict["EXIT_CODE"] = 0
    model_tr = "<tr>"
    model_tr += "<td>%s</td>\n" % model_name
    for i in range(1, col_num):
        if i == 1:
            model_tr += '<td align="center">SKIP</td>\n'
        else:
            model_tr += '<td align="center">NA</td>\n'
    model_tr += "</tr>"
    return model_tr


def make_model_tr(
    model_name, log_path, result_file, model_results_dict, env_output_checking
):
    # make model tr as string
    model_tr = "<tr>"
    # get result in build.log
    print("build log file: %s" % result_file)
    result_lines = utility.read_lines(result_file)
    results = parse_result(result_lines)
    # print("results --------", results)
    model_results_dict["Summary"] = {}
    elapse = utility.get_test_time(model_results_dict, result_lines)

    skip_cpu_ep_flag = os.environ.get("SKIP_CPU_EP", "true") == "true"
    iputrace_flag = os.environ.get("IPUTRACE", "") == "true"
    vart_perf_flag = (
        os.environ.get("PERF_XMODEL", "") == "true"
        or os.environ.get("MODEL_TYPE", "onnx") == "xmodel"
        or os.environ.get("TEST_MODE", "performance") == "vart_perf"
    )
    hello_world_flag = (os.environ.get("TEST_HELLO_WORLD", "true") == "true") and (
        "onnx_ep" in env_output_checking
    )
    output_checking_flag = (
        ("cpu_runner" in env_output_checking and "onnx_ep" in env_output_checking)
        or os.environ.get("MODEL_TYPE", "onnx") == "xmodel"
        or os.environ.get("TEST_MODE", "performance") == "vart_perf"
    )
    vaitrace_flag = os.environ.get("VAITRACE_PROFILING", "") == "true"
    branch_flag = os.environ.get("BRANCH_XMODEL", "") == "true"

    model_tr += "<td>%s</td>\n" % model_name

    subg_vai_PDI_match = utility.pattern_match(
        r".*\stotal_pdi_swaps:\s(\d+)", result_lines
    )

    # make the output tensor check
    description = get_description_str(
        model_results_dict, result_lines, log_path, model_name
    )
    vart_perf_mismatch = ""
    functionality = ""
    if output_checking_flag:
        # if model_results_dict.get("TEST_STATUS", "") == "TIMEOUT":
        #     functionality = "TIMEOUT"
        #     model_tr += (
        #         '<td align="center"><b><font color="red">Timeout</font></b></td>'
        #     )
        # elif model_results_dict.get("TEST_STATUS", "") == "Mismatch":
        #     model_tr += (
        #         '<td align="center"><b><font color="red">Mismatch</font></b></td>'
        #     )
        #     functionality = "Mismatch"
        # else:
        o = utility.pattern_match(r".*onnx_ep\soutput\stensor:\s(.*)\'", result_lines)
        onnx_ep_tensor = "" if not o else o.group(1).strip()
        c = utility.pattern_match(
            r".*cpu_runner\soutput\stensor:\s(.*)\'", result_lines
        )
        cpu_runner_tensor = "" if not c else c.group(1).strip()
        print("vitisai_ep output tensor: %s" % onnx_ep_tensor)
        print("cpu_runner output tensor: %s" % cpu_runner_tensor)
        if os.environ.get("MODEL_TYPE", "onnx") != "xmodel":
            timeout_layer_match = utility.pattern_match(
                r".*DPU\s+timeout:.*", result_lines
            )
            if timeout_layer_match:
                model_results_dict["TEST_STATUS"] = "TIMEOUT"
                model_results_dict["EXIT_CODE"] = 1
                functionality = "TIMEOUT"
                model_tr += (
                    f'<td align="center"><b><font color="red">TIMEOUT</font></b></td>'
                )
            elif (
                not (onnx_ep_tensor and cpu_runner_tensor)
                and model_results_dict.get("TEST_STATUS", "") != "SKIP"
            ):
                model_results_dict["TEST_STATUS"] = "NO_OUTPUT"
                model_results_dict["EXIT_CODE"] = 1
                functionality = "NO_OUTPUT"
                model_tr += f'<td align="center"><b><font color="red">{functionality}</font></b></td>'
            else:
                if model_results_dict.get("TEST_STATUS", "") not in [
                    "Mismatch",
                    "NO_OUTPUT",
                    "Xmodel_Diff_Crash",
                    "TIMEOUT",
                    "FAIL",
                ]:
                    functionality = "PASS"
                    model_results_dict["EXIT_CODE"] = 0
                    check_str = f'<td align="center"><b><font>PASS</font></b></td>'
                    subg_num_on_ipu_match = utility.pattern_match(
                        r".*\[XLNX_ONNX_EP_VERBOSE\]\sdpu\ssubgraph:\s(\d+)",
                        result_lines,
                    )
                    if subg_num_on_ipu_match:
                        real_dpu = subg_num_on_ipu_match.group(1).strip()
                        real_dpu_num = int(real_dpu)
                        if real_dpu_num <= 0:
                            check_str = f'<td align="center"><b><font color="red">PASS <br> CPU ALL</font></b></td>'
                    model_tr += check_str
                else:
                    model_tr += f'<td align="center"><b><font color="red">{model_results_dict["TEST_STATUS"]}</font></b></td>'
                    functionality = model_results_dict["TEST_STATUS"]
                    model_results_dict["EXIT_CODE"] = 1
        else:
            functionality = model_results_dict["TEST_STATUS"]
            if model_results_dict["TEST_STATUS"] not in [
                "Mismatch",
                "NO_OUTPUT",
                "Xmodel_Diff_Crash",
                "TIMEOUT",
                "FAIL",
            ]:
                functionality = "PASS"
                model_results_dict["EXIT_CODE"] = 0
                model_tr += f'<td align="center"><b><font>{model_results_dict["TEST_STATUS"]}</font></b></td>'
            else:
                model_tr += f'<td align="center"><b><font color="red">{model_results_dict["TEST_STATUS"]}</font></b></td>'
                model_results_dict["EXIT_CODE"] = 1
    model_results_dict["Summary"]["Functionality"] = functionality

    latency_dict = {}
    model_results_dict["Summary"]["Latency(ms)"] = latency_dict

    # make latency table
    comparison_latency = 0.0
    if not skip_cpu_ep_flag:
        comparison_latency = results.get("cpu_ep_latency", {}).get("latency", 0.0)

        comparison_latency_td = (
            '<b><font color="red">NA</font></b>'
            if not comparison_latency
            else f"<b><font>{comparison_latency}</font></b>"
        )
        model_tr += (
            '<td align ="center"><b><font>%s</font></b></td>\n' % comparison_latency_td
        )

    # get silicon time
    if vaitrace_flag:
        json_file = os.path.join(log_path, model_name, "total_runtime.json")
        if os.path.exists(json_file):
            try:
                with open(json_file, "r") as f:
                    runtime = json.load(f)
                total_hw_runtime = runtime.get("total_hw_runtime", 0)
                if not total_hw_runtime:
                    total_hw_runtime = runtime.get("total_superlayer_hw_runtime", 0)
                if not total_hw_runtime:
                    total_hw_runtime = (
                        runtime.get("profiling_data", {})
                        .get("aie", {})
                        .get("total_super_layer_runtime", 0)
                    )
                if type(total_hw_runtime) == dict:
                    min_hw_runtime = total_hw_runtime.get("total_min_hw_runtime", 0)
                    if not min_hw_runtime:
                        min_hw_runtime = total_hw_runtime.get("min_t", 0)
                else:
                    min_hw_runtime = total_hw_runtime

                total_pdi_runtime = runtime.get("total_pdi_runtime", 0)
                if not total_pdi_runtime:
                    total_pdi_runtime = runtime.get("total_pdi_hw_runtime", 0)
                if not total_pdi_runtime:
                    total_pdi_runtime = (
                        runtime.get("profiling_data", {})
                        .get("aie", {})
                        .get("total_pdi_sg_runtime", 0)
                    )
                min_pdi_runtime = (
                    total_pdi_runtime
                    if type(total_pdi_runtime) != dict
                    else total_pdi_runtime.get("min_pdi_hw_runtime", 0)
                )

                silicon_time = (
                    round(min_hw_runtime / 1000, 4)
                    if min_hw_runtime
                    else round(min_pdi_runtime / 1000, 4)
                )
                # check if aie profile buffer full
                if (
                    runtime.get("aie_profile_buffer_full", False)
                    or runtime.get("aie_profile_buf_full", False)
                    or runtime.get("metadata", {}).get("aie_profile_buf_full", False)
                ):
                    silicon_time = None
                model_results_dict["vaitrace_total_runtime"] = runtime
                if silicon_time:
                    latency_dict["silicon_time"] = silicon_time
                xmodel_latency_td = (
                    '<b><font color="red">NA</font></b>'
                    if not silicon_time
                    else f"<b><font>{silicon_time}</font></b>"
                )
                model_tr += (
                    '<td align ="center"><b><font>%s</font></b></td>\n'
                    % xmodel_latency_td
                )
            except Exception as e:
                print(e, flush=True)
                model_tr += '<td align ="center"><font></font></td>\n'
        else:
            model_tr += '<td align ="center"><font></font></td>\n'

    # get vart latency
    if vaitrace_flag:
        json_file = os.path.join(log_path, model_name, "total_runtime.json")
        if os.path.exists(json_file):
            try:
                with open(json_file, "r") as f:
                    runtime = json.load(f)
                total_sw_runtime = runtime.get("graph_engine::dpu_kernel_run", {})
                if not total_sw_runtime:
                    total_sw_runtime = (
                        runtime.get("profiling_data", {})
                        .get("graph_engine", {})
                        .get("graph_engine::dpu_kernel_run", {})
                    )
                vart_latency = round(total_sw_runtime.get("ave_t", 0) / 1000, 4)

                total_dpu_sg_num = (
                    runtime.get("profiling_data", {})
                    .get("graph_engine", {})
                    .get("total_dpu_sg_num", 0)
                )

                total_xrt_runtime = (
                    runtime.get("profiling_data", {})
                    .get("graph_engine", {})
                    .get("total_xrt_runtime", 0)
                )
                xrt_latency = round(total_xrt_runtime / 1000, 4)
                if total_xrt_runtime:
                    latency_dict["xrt"] = xrt_latency
                xrt_latency_td = (
                    '<b><font color="red">NA</font></b>'
                    if not xrt_latency
                    else f"<b><font>{xrt_latency}</font></b>"
                )
                model_tr += (
                    '<td align ="center"><b><font>%s</font></b></td>\n' % xrt_latency_td
                )

                if total_dpu_sg_num > 1:
                    vart_latency = xrt_latency
                if vart_latency:
                    latency_dict["vart"] = vart_latency

                xmodel_latency_td = (
                    '<b><font color="red">NA</font></b>'
                    if not vart_latency
                    else f"<b><font>{vart_latency}</font></b>"
                )
                model_tr += (
                    '<td align ="center"><b><font>%s</font></b></td>\n'
                    % xmodel_latency_td
                )
            except Exception as e:
                print(e, flush=True)
                model_tr += '<td align ="center"><font></font></td>\n'
                model_tr += '<td align ="center"><font></font></td>\n'
        else:
            model_tr += '<td align ="center"><font></font></td>\n'
            # one more for xrt
            model_tr += '<td align ="center"><font></font></td>\n'
    else:
        if iputrace_flag:
            if subg_vai_PDI_match and (int(subg_vai_PDI_match.group(1).strip()) == 0):
                fpstrace_latency = results.get("fpstrace", {}).get("latency", "")
                fpstrace_latency = (
                    round(float(fpstrace_latency), 4)
                    if fpstrace_latency
                    else fpstrace_latency
                )
                if fpstrace_latency:
                    latency_dict["fpstrace"] = fpstrace_latency
                fpstrace_latency_td = (
                    '<b><font color="red">NA</font></b>'
                    if not fpstrace_latency
                    else f"<b><font>{fpstrace_latency}</font></b>"
                )
                model_tr += (
                    '<td align ="center"><b><font>%s</font></b></td>\n'
                    % fpstrace_latency_td
                )
            else:
                model_tr += '<td align ="center"><font></font></td>\n'
        if vart_perf_flag:
            xmodel_latency = results.get("vitisai_xmodel", {}).get("latency", "")
            xmodel_latency = (
                round(float(xmodel_latency), 4) if xmodel_latency else xmodel_latency
            )
            if xmodel_latency and not latency_dict.get("vart", ""):
                latency_dict["vart"] = xmodel_latency
            xmodel_latency_td = (
                '<b><font color="red">NA</font></b>'
                if not xmodel_latency
                else f"<b><font>{xmodel_latency}</font></b>"
            )
            model_tr += (
                '<td align ="center"><b><font>%s</font></b></td>\n' % xmodel_latency_td
            )

    vitisai_ep_latency = results.get("vitisai_ep_latency", {}).get("latency", "")
    vitisai_ep_latency = (
        round(float(vitisai_ep_latency), 4)
        if vitisai_ep_latency
        else vitisai_ep_latency
    )
    if vitisai_ep_latency:
        latency_dict["onnxep"] = float(vitisai_ep_latency)
    vitisai_ep_latency_td = (
        '<b><font color="red">NA</font></b>'
        if not vitisai_ep_latency
        else f"<b><font>{vitisai_ep_latency}</font></b>"
    )
    model_tr += (
        '<td align ="center"><b><font>%s</font></b></td>\n' % vitisai_ep_latency_td
    )
    if not skip_cpu_ep_flag:
        if comparison_latency and vitisai_ep_latency:
            latency_ratio = float(comparison_latency) / float(vitisai_ep_latency)
            latency_ratio_td = (
                '<b><font color="red">%s</font></b>' % round(latency_ratio, 2)
                if latency_ratio < 1
                else "<b><font>%s</font></b>" % round(latency_ratio, 2)
            )
        else:
            latency_ratio_td = '<b><font color="red">NA</font></b>'
        model_tr += '<td align ="center">%s</td>\n' % latency_ratio_td

    # make fps td
    comparison_fps = ""
    comparison_fps_td = ""
    throughput_dict = {}
    model_results_dict["Summary"]["Throughput(fps)"] = throughput_dict
    if not skip_cpu_ep_flag:
        comparison_fps = results.get("cpu_ep_throughput", {}).get("fps", "")
        throughput_dict["cpuep"] = float(comparison_fps)
        comparison_fps_td = (
            '<b><font color="red">NA</font></b>'
            if not comparison_fps
            else f"<b><font>{comparison_fps}</font></b>"
        )

    vitisai_ep_latency_fps = results.get("vitisai_ep_latency", {}).get("fps", "")
    vitisai_ep_latency_fps = (
        round(float(vitisai_ep_latency_fps), 4)
        if vitisai_ep_latency_fps
        else vitisai_ep_latency_fps
    )
    vitisai_ep_fps = results.get("vitisai_ep_throughput", {}).get("fps", "")
    vitisai_ep_fps = (
        round(float(vitisai_ep_fps), 4) if vitisai_ep_fps else vitisai_ep_fps
    )
    if not vitisai_ep_fps:
        vitisai_ep_fps = vitisai_ep_latency_fps
    throughput_dict["onnxep"] = vitisai_ep_fps
    vitisai_ep_fps_td = (
        '<b><font color="red">NA</font></b>'
        if not vitisai_ep_fps
        else f"<b><font>{vitisai_ep_fps}</font></b>"
    )
    model_tr += (
        ""
        if skip_cpu_ep_flag
        else '<td align ="center"><b><font>%s</font></b></td>\n' % comparison_fps_td
    )
    model_tr += '<td align ="center"><b><font>%s</font></b></td>\n' % vitisai_ep_fps_td

    if not skip_cpu_ep_flag:
        if comparison_fps and vitisai_ep_fps:
            fps_ratio = float(vitisai_ep_fps) / float(comparison_fps)
            fps_ratio_td = (
                '<b><font color="red">%s</font></b>' % round(fps_ratio, 2)
                if fps_ratio < 1
                else "<b><font>%s</font></b>" % round(fps_ratio, 2)
            )
        else:
            fps_ratio_td = '<b><font color="red">NA</font></b>'
        model_tr += '<td align ="center">%s</td>\n' % fps_ratio_td

    # real DPU subg
    subg_num_on_ipu_match = utility.pattern_match(
        r".*\[XLNX_ONNX_EP_VERBOSE\]\sdpu\ssubgraph:\s(\d+)", result_lines
    )
    if subg_num_on_ipu_match:
        real_dpu = subg_num_on_ipu_match.group(1).strip()
        real_dpu_num = int(real_dpu)
        model_results_dict["Summary"]["DPU_Subgraph"] = (
            {}
            if not model_results_dict.get("Summary", {}).get("DPU_Subgraph", {})
            else model_results_dict["Summary"]["DPU_Subgraph"]
        )
        model_results_dict["Summary"]["DPU_Subgraph"]["On_IPU"] = real_dpu_num
        if real_dpu_num <= 0:
            real_dpu_num = '<b><font color="red">0</font></b>'
            if model_results_dict["Summary"]["Functionality"] == "PASS":
                model_results_dict["Summary"]["Functionality"] == "PASS - ToCpu"
    else:
        real_dpu_num = '<b><font color="red">NA</font></b>'
    # print("real DPU  ------------->", real_dpu_num)
    subg_num_on_ipu_td = '<td align ="center"><b><font>%s</font></b></td>' % (
        real_dpu_num
    )
    model_tr += subg_num_on_ipu_td + "\n"

    #  total DPU subg
    subg_total_DPU_match = utility.pattern_match(
        r".*\sDPU\ssubgraph\snumber\s(\d+)", result_lines
    )
    if subg_total_DPU_match:
        # print(subg_total_DPU_match.group(1), "-----------------------------")
        total_dpu = subg_total_DPU_match.group(1).strip()
        total_dpu_num = int(total_dpu)
        model_results_dict["Summary"]["DPU_Subgraph"] = (
            {}
            if not model_results_dict.get("Summary", {}).get("DPU_Subgraph", {})
            else model_results_dict["Summary"]["DPU_Subgraph"]
        )
        model_results_dict["Summary"]["DPU_Subgraph"]["xcompiler"] = total_dpu_num
    else:
        total_dpu_num = '<b><font color="red">NA</font></b>'
    # print("xcompiler DPU  ------------->", total_dpu_num)
    subg_total_dpu_td = '<td align ="center"><b><font>%s</font></b></td>' % (
        total_dpu_num
    )
    model_tr += subg_total_dpu_td + "\n"

    #  real pdi
    if subg_vai_PDI_match:
        # print(subg_vai_PDI_match.group(1), "-----------------------------")
        vai_pdi = subg_vai_PDI_match.group(1).strip()
        vai_pdi_num = int(vai_pdi)
        model_results_dict["Summary"]["PDI_Number"] = (
            {}
            if not model_results_dict.get("Summary", {}).get("PDI_Number", {})
            else model_results_dict["Summary"]["PDI_Number"]
        )
        model_results_dict["Summary"]["PDI_Number"]["On_IPU"] = vai_pdi_num
    else:
        vai_pdi_num = "NA"
    # print("real pdi ------------->", vai_pdi_num)
    subg_vai_PDI_td = '<td align="center"><b><font>%s</font></b></td>' % (vai_pdi_num)
    model_tr += subg_vai_PDI_td + "\n"

    #   xcompiler pdi
    subg_num_on_PDI_match = utility.pattern_match(
        r".*\stotal\snumber\sof\sPDI\sswaps\s(\d+)", result_lines
    )
    if subg_num_on_PDI_match:
        # print(subg_num_on_PDI_match.group(1), "-----------------------------")
        xcompiler_pdi = subg_num_on_PDI_match.group(1).strip()
        xcompiler_pdi_num = int(xcompiler_pdi)
        model_results_dict["Summary"]["PDI_Number"] = (
            {}
            if not model_results_dict.get("Summary", {}).get("PDI_Number", {})
            else model_results_dict["Summary"]["PDI_Number"]
        )
    else:
        xcompiler_pdi_num = "NA"
    model_results_dict["Summary"]["PDI_Number"]["xcompiler"] = xcompiler_pdi_num
    # print("xcompiler pdi ------------->", xcompiler_pdi_num)
    subg_total_PDI_td = '<td align="center"><b><font>%s</font></b></td>' % (
        xcompiler_pdi_num
    )
    model_tr += subg_total_PDI_td + "\n"

    #  total CPU subg
    subg_total_CPU_match = utility.pattern_match(
        r".*\sCPU\ssubgraph\snumber\s(\d+)", result_lines
    )
    if subg_total_CPU_match:
        # print(subg_total_CPU_match.group(1), "-----------------------------")
        total_cpu = subg_total_CPU_match.group(1).strip()
        total_cpu_num = int(total_cpu)
        model_results_dict["Summary"]["CPU_Subgraph"] = total_cpu_num
    else:
        total_cpu_num = '<b><font color="red">NA</font></b>'
    # print("xcompiler CPU  ------------->", total_cpu_num)
    subg_num_on_cpu_td = '<td align ="center"><b><font>%s</font></b></td>' % (
        total_cpu_num
    )
    model_tr += subg_num_on_cpu_td + "\n"

    # OPS
    print("Start to parse OPS")
    ops = parse_ops.parse_ops(model_name, log_path)
    # if not ops:
    #     ops_match = utility.pattern_match(
    #         r".*No. of Operators : .+?(\d+\.\d+%)", result_lines
    #     )
    #     ops = "" if not ops_match else ops_match.group(1).strip()
    model_results_dict["Summary"]["OPS"] = ops
    model_tr += (
        '<td align ="center">%s%%</td>\n' % ops
        if ops
        else '<td align ="center"><b><font color="red">NA</font></b></td>\n'
    )
    # make I2norm table
    if hello_world_flag:
        # l2normDesc = utility.pattern_match(r"l2normDesc:.*", result_lines)
        l2normDesc = utility.pattern_match(r"l2normDesc:\s*(.*)", result_lines)
        l2norm_pat = utility.pattern_match(r"l2norm_hw_log:\s*(.*)", result_lines)
        if l2normDesc:
            l2norm_val = l2normDesc.group(1).strip()
            model_tr += f'<td align="center"><b><font>{l2norm_val}</font></b></td>'
            print("l2normDesc ---->", l2normDesc, flush=True)

        elif l2norm_pat:
            print("l2norm_pat ---->", l2norm_pat, flush=True)
            l2norm_val = l2norm_pat.group(1).strip()
            model_tr += f'<td align="center"><b><font>{l2norm_val}</font></b></td>'
            model_results_dict["l2norm"] = l2norm_val
        else:
            model_tr += f'<td align="center"><b><font color="red">NA</font></b></td>'

    if branch_flag:
        pattern_begin_thread1 = r".*####branch_model threads 1####.*"
        pattern_begin_thread2 = r".*####branch_model threads 2####.*"
        pattern_begin_thread3 = r".*####branch_model threads 3####.*"
        pattern_begin_thread4 = r".*####branch_model threads 4####.*"

        begin1, begin2, begin3, begin4 = None, None, None, None
        for i, line in enumerate(result_lines):
            if re.match(pattern_begin_thread1, line):
                begin1 = i

            elif re.match(pattern_begin_thread2, line):
                begin2 = i

            elif re.match(pattern_begin_thread3, line):
                begin3 = i

            elif re.match(pattern_begin_thread4, line):
                begin4 = i
        if begin1 and begin2 and begin3 and begin4:
            model_tr += search_branch_model(result_lines[begin1:begin2])
            model_tr += search_branch_model(result_lines[begin2:begin3])
            model_tr += search_branch_model(result_lines[begin3:begin4])
            model_tr += search_branch_model(result_lines[begin4:])
        else:
            model_tr += '<td align="center"><b><font color="red">N/A</font></b></td>'
            model_tr += '<td align="center"><b><font color="red">N/A</font></b></td>'
            model_tr += '<td align="center"><b><font color="red">N/A</font></b></td>'
            model_tr += '<td align="center"><b><font color="red">N/A</font></b></td>'

    # md5
    onnx_md5_match = utility.pattern_match(r".*model\smd5sum\sis:'\s(.*)", result_lines)
    onnx_md5 = "" if not onnx_md5_match else onnx_md5_match.group(1).strip()[:7]
    model_results_dict["Summary"]["onnx_model_md5"] = onnx_md5
    # print("onnx_md5 ------------->", onnx_md5)
    # print("elapse ------------->", elapse)
    model_tr += '<td align ="center">%s</td>\n' % onnx_md5
    model_tr += "<td><b><font>%s</font></b></td>" % elapse
    model_results_dict["Summary"]["Elapse(s)"] = elapse

    # Description

    model_tr += description
    model_tr += "</tr>"

    return model_tr


def search_branch_model(result_lines):
    pattern_result = r".*xmodel\s*:\s*.+context id\s*:\s*\d+,\s*diff rate\s*:\s*.+,\s*wrong\s*:\s*\d+,\s*right\s*:\s*\d+"
    pattern_match = r".*xmodel\s*:\s*.+context id\s*:\s*\d+,\s*diff rate\s*:\s*0,\s*wrong\s*:\s*0,\s*right\s*:\s*\d+"
    pattern_timeout = r".*DPU timeout:.*"
    pattern_error = r"F.*Error.*"
    for line in result_lines:
        print(line.replace("\n", ""))
    for line in result_lines:
        if re.match(pattern_result, line):
            if not re.match(pattern_match, line):
                return (
                    '<td align="center"><b><font color="red">MISMATCH</font></b></td>'
                )
        elif re.match(pattern_timeout, line):
            return '<td align="center"><b><font color="red">TIMEOUT</font></b></td>'
        elif re.match(pattern_error, line):
            return '<td align="center"><b><font color="red">ERROR</font></b></td>'
    return '<td align="center"><b><font>PASS</font></b></td>'


def serach_onnx_ep(result_lines):
    try:
        split_index = None
        for i, line in enumerate(result_lines):
            if "###test_xmodel_diff###" in line:
                split_index = i
                break
        if split_index is not None:
            first_half = result_lines[:split_index]
            second_half = result_lines[split_index:]
            timeout_layer_match = utility.pattern_match(
                r".*\DPU\s+timeout:.*", first_half
            )
        else:
            timeout_layer_match = utility.pattern_match(
                r".*\DPU\s+timeout:.*", result_lines
            )
        return not not timeout_layer_match
    except Exception as e:
        print(f"!!! warning : serach onnx_ep timeout failed! {e}.)")


def check_timeout(result_lines, tool_flag):
    timeout_layer_match = utility.pattern_match(r".*DPU\s+timeout:.*", result_lines)
    layer_name = ""
    if timeout_layer_match:
        # print(timeout_layer_match.group(), "---timeout---")
        pattern = r"\sDPU\s+timeout:.*"
        layer_name = (
            tool_flag
            + " -------- "
            + re.search(pattern, timeout_layer_match.group().strip()).group()
        )
    return layer_name


def check_failed(search_pattern, result_lines, tool_flag):
    fail_match = utility.pattern_match(search_pattern, result_lines)
    fail_mes = ""
    if fail_match:
        # print(fail_match.group(), "---fail---")
        pattern = f"\s\S+\]\s+{search_pattern}"
        if re.search(pattern, fail_match.group().strip()):
            fail_mes = (
                tool_flag
                + " -------- "
                + re.search(pattern, fail_match.group().strip()).group()
            )
        else:
            fail_mes = tool_flag + " -------- " + fail_match.group().strip()
    return fail_mes


def analyze_parsed_failed_cases(result_lines):
    try:
        error_message = []
        ipu_ep_start = (
            ipu_ep_end
        ) = (
            ipu_throughput_start
        ) = (
            ipu_throughput_end
        ) = (
            vitisai_xmodel_start
        ) = vitisai_xmodel_end = xmodel_diff_start = xmodel_diff_end = 0
        for index, line in enumerate(result_lines):
            if "###vitisai_ep_latency###" in line:
                ipu_ep_start = index
            if "###cpu_ep_throughput###" in line:
                ipu_ep_end = index - 1
            if "###vitisai_ep_throughput###" in line:
                ipu_throughput_start = index
                if not ipu_ep_end:
                    ipu_ep_end = index - 1
            if "###vitisai_xmodel###" in line:
                vitisai_xmodel_start = index
                if not ipu_ep_end:
                    ipu_ep_end = index - 1
                elif not ipu_throughput_end:
                    ipu_throughput_end = index - 1
            if "###vitisai_ep_tensor###" in line:
                if not ipu_ep_end:
                    ipu_ep_end = index - 1
                elif ipu_throughput_start and not ipu_throughput_end:
                    ipu_throughput_end = index - 1
                elif vitisai_xmodel_start and not vitisai_xmodel_end:
                    vitisai_xmodel_end = index - 1
            if "###test_xmodel_diff###" in line:
                xmodel_diff_start = index
                if not ipu_ep_end:
                    ipu_ep_end = index - 1
                elif ipu_throughput_start and not ipu_throughput_end:
                    ipu_throughput_end = index - 1
                elif vitisai_xmodel_start and not vitisai_xmodel_end:
                    vitisai_xmodel_end = index - 1
            if "###test_hello_world###" in line:
                xmodel_diff_end = index - 1
            if "###dump_inst_data###" in line:
                if not xmodel_diff_end:
                    xmodel_diff_end = index - 1

        if not ipu_ep_end:
            ipu_ep_end = -1
        if not ipu_throughput_end:
            ipu_throughput_end = -1
        if not vitisai_xmodel_end:
            vitisai_xmodel_end = -1
        if not xmodel_diff_end:
            xmodel_diff_end = -1

        ipu_ep_lines = []
        if ipu_ep_start and ipu_ep_end:
            ipu_ep_lines = result_lines[ipu_ep_start : ipu_ep_end + 1]
            if ipu_ep_end == -1:
                ipu_ep_lines = result_lines[ipu_ep_start:ipu_ep_end]
        ipu_throughput_lines = []
        if ipu_throughput_start and ipu_throughput_end:
            ipu_throughput_lines = result_lines[
                ipu_throughput_start : ipu_throughput_end + 1
            ]
            if ipu_throughput_end == -1:
                ipu_throughput_lines = result_lines[
                    ipu_throughput_start:ipu_throughput_end
                ]
        vitisai_xmodel_lines = []
        if vitisai_xmodel_start and vitisai_xmodel_end:
            vitisai_xmodel_lines = result_lines[
                vitisai_xmodel_start : vitisai_xmodel_end + 1
            ]
            if vitisai_xmodel_end == -1:
                vitisai_xmodel_lines = result_lines[
                    vitisai_xmodel_start:vitisai_xmodel_end
                ]
        xmodel_diff_lines = []
        if xmodel_diff_start and xmodel_diff_end:
            xmodel_diff_lines = result_lines[xmodel_diff_start : xmodel_diff_end + 1]
            if xmodel_diff_end == -1:
                xmodel_diff_lines = result_lines[xmodel_diff_start:xmodel_diff_end]

        if ipu_ep_lines:
            compile_start = compile_end = False
            tool_flag = "Onnxruntime perf test"
            for line in ipu_ep_lines:
                if "Begin to compile" in line:
                    compile_start = True
                if "Compile done" in line:
                    compile_end = True
            if compile_start and compile_end:
                timeout_layer = check_timeout(ipu_ep_lines, tool_flag)
                GE_error_pattern = r".*Error:.*"
                GE_error_match = check_failed(GE_error_pattern, ipu_ep_lines, tool_flag)
                if timeout_layer != "":
                    error_message.append(timeout_layer)
                if GE_error_match != "":
                    error_message.append(GE_error_match)
            else:
                check_fail_pattern = r".*Check\s+failed.*"
                fatal_error_pattern = r".*Fatal\s+error.*"
                exception_pattern = ".*\(Exception\s+type.*"
                check_fail_match = check_failed(
                    check_fail_pattern, ipu_ep_lines, tool_flag
                )
                fatal_error_match = check_failed(
                    fatal_error_pattern, ipu_ep_lines, tool_flag
                )
                exception_match = check_failed(
                    exception_pattern, ipu_ep_lines, tool_flag
                )
                if check_fail_match != "":
                    error_message.append(check_fail_match)
                if fatal_error_match != "":
                    error_message.append(fatal_error_match)
                if exception_match != "":
                    error_message.append(exception_match)

        if ipu_throughput_lines:
            tool_flag = "Onnxruntime perf test throughput"
            timeout_layer = check_timeout(ipu_throughput_lines, tool_flag)
            if timeout_layer != "":
                error_message.append(timeout_layer)

        if vitisai_xmodel_lines:
            tool_flag = "Vart perf"
            timeout_layer = check_timeout(vitisai_xmodel_lines, tool_flag)
            if timeout_layer != "":
                error_message.append(timeout_layer)

        if xmodel_diff_lines:
            tool_flag = "Test xmodel diff"
            timeout_layer = check_timeout(xmodel_diff_lines, tool_flag)
            exception_pattern = ".*\(Exception\s+type.*"
            exception_match = check_failed(
                exception_pattern, xmodel_diff_lines, tool_flag
            )
            if timeout_layer != "":
                error_message.append(timeout_layer)
            elif exception_match != "":
                error_message.append(exception_match)

        return error_message
    except Exception as e:
        logging.warning(
            f"!!! warning : analyze_parsed_failed_cases function failed! {e}.)"
        )


def get_description_str(model_results_dict, result_lines, log_path, model_name):
    o = utility.pattern_match(r".*onnx_ep\soutput\stensor:\s(.*)\'", result_lines)
    onnx_ep_tensor = "" if not o else o.group(1).strip()
    c = utility.pattern_match(r".*cpu_runner\soutput\stensor:\s(.*)\'", result_lines)
    cpu_runner_tensor = "" if not c else c.group(1).strip()
    is_mismatch = cpu_runner_tensor != onnx_ep_tensor
    layer_name = ""
    error_type = analyze_parsed_failed_cases(result_lines)
    if error_type:
        for i in error_type[:-1]:
            layer_name += i + "<br>"
        if "timeout" in error_type[-1]:
            model_results_dict["TEST_STATUS"] = "TIMEOUT"
            layer_name += error_type[-1]
        elif (
            "Exception type" in error_type[-1]
            or "Check failed" in error_type[-1]
            or "Fatal error" in error_type[-1]
            or "Error:" in error_type[-1]
        ):
            model_results_dict["TEST_STATUS"] = "FAIL"
            layer_name += error_type[-1]
    else:
        diff_status = False
        for index, line in enumerate(result_lines):
            if "###test_xmodel_diff###" in line:
                diff_status = True
                break
        get_check_mismatch(
            model_results_dict, log_path, model_name, diff_status, is_mismatch
        )
        layer_name += model_results_dict["desc"]

    py3_env_match = utility.pattern_match(r"XLNX_ENABLE_PY3_ROUND.*", result_lines)
    if py3_env_match:
        layer_name += "<br>" + py3_env_match.group(0)
    qdq_env_match = utility.pattern_match(r"XLNX_ENABLE_OLD_QDQ.*", result_lines)
    if qdq_env_match:
        layer_name += "<br>" + qdq_env_match.group(0)
    ep_ctx_failed_match = utility.pattern_match(
        r"FAILED@generate_ep_context.*", result_lines
    )
    if ep_ctx_failed_match:
        layer_name += (
            "<br>"
            + '<b><font color="red">%s</font></b>' % "FAILED@generate_ep_context "
        )
    model_results_dict["desc"] = layer_name

    # print("Description message:  ------------->", layer_name)
    description = '<td align ="center"><b><font>%s</font></b></td>' % (layer_name)
    return description


def get_check_mismatch(
    model_results_dict, log_path, model_name, diff_status, is_mismatch
):
    try:
        if not diff_status:
            model_results_dict["TEST_STATUS"] = "PASS"
            model_results_dict["EXIT_CODE"] = 0
            model_results_dict["desc"] = ""
            return

        diff_json = os.path.join(log_path, model_name, "layer_result", "result.json")
        if os.path.exists(diff_json):
            with open(diff_json, "r") as file:
                data = json.load(file)
            for detail in data.get("summary", {}).get("detail", []):
                if detail.get("status") == "mismatch":
                    layer_name = detail.get("mismatch_layer", [])
                    for subgraph in detail.get("subgraph_status", []):
                        if subgraph.get("subgraph_status") == "mismatch":
                            subgraph_name = subgraph.get("subgraph_name", "")
                    model_results_dict["TEST_STATUS"] = "Mismatch"
                    model_results_dict["EXIT_CODE"] = 1
                    model_results_dict[
                        "desc"
                    ] = f"Test xmodel diff -------- Mismatch! DPU subgraph:{subgraph_name}, layer:{','.join(layer_name)}"
                else:
                    model_results_dict["desc"] = ""
                    model_results_dict["EXIT_CODE"] = 0
                    is_all_zero_layer_list = []
                    for subgraph in detail.get("subgraph_status", []):
                        for subgraph_item in subgraph.get("superlayer_status", []):
                            if subgraph_item.get("is_all_zero") == 1:
                                is_all_zero_layer_list.append(subgraph_item.get("name"))
                    if len(is_all_zero_layer_list):
                        model_results_dict[
                            "TEST_STATUS"
                        ] = f"PASS <br> is_all_zero_layer = 1 layer: {','.join(is_all_zero_layer_list)}"
                    else:
                        model_results_dict["TEST_STATUS"] = "PASS"
        elif is_mismatch:
            model_results_dict["TEST_STATUS"] = "Mismatch"
            model_results_dict["desc"] = ""
            model_results_dict["EXIT_CODE"] = 1
        else:
            print("result.json file not found ", flush=True)
            model_results_dict["TEST_STATUS"] = "Xmodel_Diff_Crash"
            model_results_dict["EXIT_CODE"] = 1
            model_results_dict["desc"] = ""
        print("model_results_dict --->", model_results_dict, flush=True)
    except Exception as e:
        print(f"!!! warning : get_check_mismatch failed! {e}.)")


def cal_procyon_score(modelzoo_results):
    procyon_models = (
        "deeplabv3",
        "inceptionv4",
        "resnet50",
        "yolov3",
        "esrgan",
        "mobilenetv3",
    )
    procyon_latency_list = []
    for procyon_model in procyon_models:
        procyon_model_latency = (
            modelzoo_results.get(procyon_model, {})
            .get("Summary", {})
            .get("Latency(ms)", {})
            .get("onnxep", "")
        )
        if not procyon_model_latency:
            print("Warning: no %s latency" % procyon_model)
            return ""
        procyon_latency_list.append(float(procyon_model_latency))
    print("procyon_latency_list: %s " % str(procyon_latency_list))
    return round(5000 * (1 / utility.geomean(procyon_latency_list)), 2)


def make_thead(
    target_type,
    dpu_type,
    comparison_data_name,
    comparison_type,
    runners_num,
    threads_num,
    env_output_checking,
):
    col_num = 0
    thead_tr1 = f'<th rowspan="2" style="width:200px;">{target_type} {dpu_type}<br>Model Name</th>\n'
    col_num += 1
    # thead_tr1 += '<th rowspan="2" style="width:200px;">DPU Subgraph</th>\n'
    # if os.environ.get("OUTPUT_CHECKING", "true") == "true":
    if (
        ("cpu_runner" in env_output_checking and "onnx_ep" in env_output_checking)
        or os.environ.get("MODEL_TYPE", "onnx") == "xmodel"
        or os.environ.get("TEST_MODE", "performance") == "vart_perf"
    ):
        thead_tr1 += '<th rowspan="2" style="width:200px;">Functionality CPU_RUNNER VS VITISAI_EP</th>\n'
        col_num += 1
    skip_cpu_ep_flag = os.environ.get("SKIP_CPU_EP", "true") == "true"
    vaitrace_flag = os.environ.get("VAITRACE_PROFILING", "") == "true"
    iputrace_flag = os.environ.get("IPUTRACE", "") == "true"
    vart_perf_flag = (
        os.environ.get("PERF_XMODEL", "") == "true"
        or os.environ.get("MODEL_TYPE", "onnx") == "xmodel"
        or os.environ.get("TEST_MODE", "performance") == "vart_perf"
    )
    latency_rowspan = 1
    latency_rowspan = latency_rowspan + 2 if not skip_cpu_ep_flag else latency_rowspan
    latency_rowspan = latency_rowspan + 1 if vaitrace_flag else latency_rowspan
    latency_rowspan = (
        latency_rowspan + 1 if iputrace_flag or vaitrace_flag else latency_rowspan
    )
    latency_rowspan = (
        latency_rowspan + 1 if (vart_perf_flag or vaitrace_flag) else latency_rowspan
    )
    thead_tr1 += (
        f'<th colspan="{latency_rowspan}" style="width:200px;">Latency(ms)</th>\n'
    )
    col_num += latency_rowspan

    throughput_rowspan = 1
    throughput_rowspan = (
        throughput_rowspan + 2 if not skip_cpu_ep_flag else throughput_rowspan
    )
    thead_tr1 += f'<th colspan="{throughput_rowspan}" style="width:300px;">Throughput(fps)</th>\n'
    col_num += throughput_rowspan
    thead_tr1 += '<th colspan="2" style="width:200px;">DPU subgraph</th>\n'
    col_num += 2
    thead_tr1 += '<th colspan="2" style="width:200px;">PDI SWAP</th>\n'
    col_num += 2
    thead_tr1 += '<th style="width:200px;"> CPU subgraph</th>\n'
    col_num += 1
    thead_tr1 += '<th rowspan="2" style="width:200px;">OPS on IPU</th>\n'
    col_num += 1

    if (os.environ.get("TEST_HELLO_WORLD", "true") == "true") and (
        "onnx_ep" in env_output_checking
    ):
        # if os.environ.get("TEST_HELLO_WORLD", "true") == "true":
        thead_tr1 += '<th rowspan="2" style="width:200px;">Hello World l2norm(graph_engine vs vitisai_ep)</th>\n'
        col_num += 1
    if os.environ.get("BRANCH_XMODEL", "") == "true":
        thead_tr1 += (
            '<th rowspan="2" style="width:200px;">branch model<br>1 thread</th>\n'
        )
        col_num += 1

        thead_tr1 += (
            '<th rowspan="2" style="width:200px;">branch model<br>2 thread</th>\n'
        )
        col_num += 1

        thead_tr1 += (
            '<th rowspan="2" style="width:200px;">branch model<br>3 thread</th>\n'
        )
        col_num += 1

        thead_tr1 += (
            '<th rowspan="2" style="width:200px;">branch model<br>4 thread</th>\n'
        )
        col_num += 1

    thead_tr1 += '<th rowspan="2" style="width:200px;">onnx md5</th>\n'
    col_num += 1
    thead_tr1 += '<th rowspan="2" style="width:200px;">Elapse(s)</th>\n'
    col_num += 1
    thead_tr1 += '<th rowspan="2" style="width:500px;">Description</th>\n'
    col_num += 1

    thead_tr2 = (
        f'<th style="width:100px;">{comparison_data_name}</th>\n'
        if not skip_cpu_ep_flag
        else ""
    )
    thead_tr2 += '<th style="width:100px;">Silicon Time</th>\n' if vaitrace_flag else ""
    thead_tr2 += (
        '<th style="width:100px;">xrt</th>\n' if iputrace_flag or vaitrace_flag else ""
    )
    thead_tr2 += (
        '<th style="width:100px;">vart</th>'
        if (vart_perf_flag or vaitrace_flag)
        else ""
    )
    thead_tr2 += '<th style="width:150px;">onnx ep<br>(1 runner,1 thread)</th>'
    thead_tr2 += (
        f'<th style="width:100px;">{comparison_type}</th>'
        if not skip_cpu_ep_flag
        else ""
    )
    thead_tr2 += (
        f'<th style="width:100px;">{comparison_data_name}</th>\n'
        if not skip_cpu_ep_flag
        else ""
    )
    thead_tr2 += f'<th style="width:150px;">onnx ep<br>({runners_num} runners,{threads_num} threads)</th>'
    thead_tr2 += (
        f'<th style="width:100px;">{comparison_type}</th>'
        if not skip_cpu_ep_flag
        else ""
    )
    thead_tr2 += f'<th style="width:150px;">On IPU</th>'
    thead_tr2 += f'<th style="width:150px;">Xcompiler</th>'
    thead_tr2 += f'<th style="width:150px;">On IPU</th>'
    thead_tr2 += f'<th style="width:150px;">Xcompiler</th>'
    thead_tr2 += f'<th style="width:150px;">Xcompiler</th>'

    return thead_tr1, thead_tr2, col_num


def read_control_file():
    all_list = []
    skip_list = []
    control_file = os.environ.get("USER_CONTROL_FILE", "")
    print(f"control_file: {control_file}", flush=True)
    if control_file != "":
        with open(control_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                tmp_list = line.strip().split(",")
                # print(tmp_list, flush=True)
                if tmp_list[0] == "SKIP":
                    skip_list.append(tmp_list[1])
                all_list.append(tmp_list[1])
    return all_list, skip_list


def compare_perf(
    model_list,
    log_path,
    compare_html,
    modelzoo_list,
    suite_run_elapsed="",
    baseline_benchmark_data=None,
    run_date="",
):
    try:
        print("Step into compare_perf function ...", flush=True)
        modelzoo_results = {}
        if not model_list:
            print("ERROR:No valid model list", flush=True)
            return

        jenkins_url = (
            os.environ.get("BUILD_URL", "http://xcdl190259:8080/view/software/job/dw/")
            + "artifact/"
            + compare_html
        )
        jenkins_url = '<a href="%s">%s</a>' % (jenkins_url, jenkins_url)
        job_name = os.environ.get("JOB_BASE_NAME", "dw")
        build_id = os.environ.get("BUILD_ID", "999")
        target_type = os.environ.get("TARGET_TYPE", "STRIX")
        tbody = ""

        comparison_data_name = "cpu ep"
        comparison_type_name = "Speed Up<br>onnx VS cpu"

        env_output_checking = os.environ.get("OUTPUT_CHECKING", "cpu_runner,onnx_ep")
        env_model_type = os.environ.get("MODEL_TYPE", "onnx")
        env_test_mode = os.environ.get("TEST_MODE", "performance")
        if env_output_checking == "true":
            if (
                env_model_type == "xmodel"
                or os.environ.get("TEST_MODE", "performance") == "vart_perf"
            ):
                env_output_checking = "cpu_runner,ipu"
            elif "performance" in env_test_mode:
                env_output_checking = "cpu_runner,onnx_ep"
            else:
                env_output_checking = "cpu_ep,onnx_ep"

        dpu_type = os.environ.get("XLNX_VART_FIRMWARE", "").strip().split("\\")[-1]
        thead_tr1, thead_tr2, col_num = make_thead(
            target_type,
            dpu_type,
            comparison_data_name,
            comparison_type_name,
            os.environ.get("NUM_OF_DPU_RUNNERS", "1"),
            os.environ.get("THREAD", "1"),
            env_output_checking,
        )

        system_version, verbose, xcompiler_optimizations = utility.get_verbose()
        # print("xcompiler optimization---------------->", xcompiler_optimizations)
        print(f"Compare perf table header column num: {col_num}", flush=True)
        all_list, skip_list = read_control_file()
        print(all_list, flush=True)
        for model_name in all_list if len(all_list) else model_list:
            print("making model %s" % model_name, flush=True)
            modelzoo_results[model_name] = {}
            result_file = os.path.join(log_path, model_name, "build.log")
            if not os.path.exists(result_file):
                print("Warning: %s no build.log found!" % model_name)
                tr_str = make_empty_model_tr(
                    model_name, col_num, modelzoo_results[model_name]
                )
            else:
                tr_str = make_model_tr(
                    model_name,
                    log_path,
                    result_file,
                    modelzoo_results[model_name],
                    env_output_checking,
                )
            tbody += tr_str

        procyon_score = cal_procyon_score(modelzoo_results)

        procyon_score_font = (
            ""
            if not procyon_score
            else f"<b><font>Procyon Score: {procyon_score}</font></b><br><b>"
        )

        # write html report
        target_info = utility.make_target_info(modelzoo_list)
        opt_level = os.environ.get("OPT_LEVEL", "0")
        modelzoo = os.environ.get("MODEL_ZOO", "")
        run_type = "%s %s OPT%s" % (modelzoo.upper(), dpu_type, opt_level.upper())
        run_date_str = "" if not run_date else "Test Date: %s\n" % run_date
        elapsed_str = (
            "" if not suite_run_elapsed else "Suite Run Elapsed: %s" % suite_run_elapsed
        )

        result_summary = get_result_summary(modelzoo_results)
        modelzoo_results["VERSION_INFO"] = utility.get_version_info()
        modelzoo_results["XCOMPILER_ATTRS"] = utility.html_to_json(verbose)

        modelzoo_results["TEST_MODE"] = os.environ.get("TEST_MODE", "")
        modelzoo_results["REGRESSION_TYPE"] = os.environ.get("REGRESSION_TYPE", "")
        modelzoo_results["MODEL_GROUP"] = os.environ.get("MODEL_GROUP", "")
        modelzoo_results["TARGET_TYPE"] = target_type.capitalize()
        if dpu_type:
            modelzoo_results["TARGET_NAME"] = os.path.splitext(dpu_type)[0]
        modelzoo_results["RUN_DATE"] = run_date
        modelzoo_results["MODEL_ZOO"] = os.environ.get("MODEL_ZOO", "")
        modelzoo_results["BUILD_ID"] = build_id
        modelzoo_results["JOB_BASE_NAME"] = job_name
        modelzoo_results["BUILD_USER"] = os.environ.get("BUILD_USER", "")
        modelzoo_results["BUILD_URL"] = os.environ.get("BUILD_URL", "")
        if procyon_score:
            modelzoo_results["Procyon_Score"] = procyon_score
        print("modelzoo_results ----------------------", modelzoo_results)

        json_to_save_data = os.environ.get(
            "BENCHMARK_RESULT_JSON",
            f"benchmark_result_{job_name}_{build_id}_{run_date}.json",
        )
        print("benchmark result json ----------------------", json_to_save_data)
        if os.path.exists(json_to_save_data):
            for model in model_list:
                update_modelzoo_result(modelzoo_results, model, json_to_save_data)
                print("modelzoo_results updated", flush=True)
        utility.dump_dict_to_json(modelzoo_results, json_to_save_data)

        test_mode = os.environ.get("TEST_MODE", "performance")
        compare_table = ""
        if (
            baseline_benchmark_data
            and os.path.exists(baseline_benchmark_data)
            and os.path.exists(json_to_save_data)
        ):
            if "performance" in test_mode:
                (
                    compare_table,
                    compare_data,
                ) = compare_with_baseline.compare_with_baseline(
                    baseline_benchmark_data,
                    json_to_save_data,
                    "performance",
                    target_type,
                    dpu_type,
                )
                opera_json(json_to_save_data, compare_data)

        paras = {
            "thead_tr1": thead_tr1,
            "thead_tr2": thead_tr2,
            "title": "performance report ",
            "tbody_content": tbody,
            "target_info": target_info,
            "jenkins_url": jenkins_url,
            "run_type": run_type,
            "run_date": run_date_str,
            "elapsed": elapsed_str,
            "system_version": system_version,
            "procyon_score": procyon_score_font,
            "verbose_list": verbose,
            "result_summary": result_summary,
            "xcompiler_optimizations": xcompiler_optimizations,
            "compare_table": compare_table,
        }
        utility.write_file(
            compare_html, constant.PERFORMANCE_HTML_TEMPLATE.format(**paras)
        )

        json_to_txt = os.environ.get(
            "BENCHMARK_RESULT_TXT",
            f"user_control_test_{job_name}_{build_id}_{run_date}.txt",
        )
        utility.dump_dict_to_txt(modelzoo_results, json_to_txt)

        return modelzoo_results
    except Exception:
        tb = traceback.format_exc()
        if not tb is None:
            print(tb)


def update_modelzoo_result(modelzoo_results, model, json_to_save_data):
    try:
        with open(json_to_save_data, "r") as file:
            list_json_data = json.load(file)
        for data in list_json_data:
            if data == model:
                print("update result for model: ", model, flush=True)
                for key, val in list_json_data.get(model).get("Summary").items():
                    if key not in modelzoo_results.get(model).get("Summary"):
                        modelzoo_results[model]["Summary"][key] = val
                        print(f"add {key}={val}", flush=True)
        print("updated modelzoo_results ----------------------", modelzoo_results)
    except Exception as e:
        print("update modelzoo result failed!", e)


def opera_json(json_file, compare_data):
    with open(json_file, "r") as total_file:
        total_json = json.load(total_file)

    for key, value in compare_data["COMPARE_DATA"].items():
        all_pass = True
        for sub_key, sub_value in value.items():
            if sub_value.get("status") != "PASS":
                all_pass = False
                break

        if key in total_json:
            if all_pass:
                total_json[key]["Summary"]["BenchmarkStatus"] = "PASS"
            else:
                total_json[key]["Summary"]["BenchmarkStatus"] = "FAIL"

    with open(json_file, "w") as file:
        json.dump(total_json, file, indent=4)


def set_env(env):
    try:
        with open(env, "r") as f:
            lines = f.readlines()
            for each in lines:
                each_env = each.strip().split("=")
                if len(each_env) != 2:
                    print(f"environment '{each}' not valid!")
                else:
                    os.environ[each_env[0].strip()] = each_env[1].strip()
    except Exception as e:
        print(f"set_env {env} error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", type=str, default="", help="log path.")
    parser.add_argument(
        "--compare_html", type=str, default="", help="match compare report."
    )
    parser.add_argument("--model_list", type=str, default="", help="model list.")
    parser.add_argument("--modelzoo", type=str, default="", help="modelzoo json path.")
    parser.add_argument("--env", type=str, default="", help="env file path.")
    args = parser.parse_args()
    set_env(args.env)
    model_list = None if not args.model_list else args.model_list.split(" ")
    modelzoo_json = args.modelzoo
    with open(modelzoo_json, "r") as mz:
        modelzoo_list = json.load(mz)
    compare_perf(model_list, args.log_path, args.compare_html, modelzoo_list)
