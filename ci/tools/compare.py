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
import time
import hashlib
import logging
import traceback
import json
from pathlib import Path
from collections import OrderedDict
from . import utility
from . import constant

# import utility
# from . import onnx_ops_parse


def get_compute_time(result_lines):
    try:
        compute_time = {}
        key_words = {
            "cpu": "test_cpu_ep",
            "cpu_runner": "test_cpu_runner",
            "ipu": "test_dpu_ep",
        }
        pattern = r"dpu subgraph: (\d+)"
        pattern_time = r"COMPUTE : (\d+)"
        result_type = ""
        computes = []
        for line in result_lines:
            for type, key in key_words.items():
                if key in line:
                    result_type = type
            m = re.search(pattern, line)
            m_time = re.search(pattern_time, line)
            if m and result_type:
                if not compute_time.get(result_type, {}):
                    compute_time[result_type] = {}
                    computes = []
                dpu_subgraph = m.group(1)
                compute_time[result_type]["dpu_subgraph"] = dpu_subgraph
            if m_time and result_type:
                compute_time[result_type] = (
                    {}
                    if not compute_time.get(result_type, {})
                    else compute_time[result_type]
                )
                computes.append({"compute": m_time.group(1)})
                compute_time[result_type]["computes"] = computes
        return compute_time
    except Exception as e:
        logging.warning(f"!!! warning : get compute time failed! {e}.)")


def parse_result(result_lines):
    results = {}
    key_words = {
        "cpu": "test_cpu_ep",
        "cpu_runner": "test_cpu_runner",
        "ipu": "test_dpu_ep",
    }
    # score[258]  =  0.771313     text: Samoyed, Samoyede,
    pattern = (
        r".*?score\[(\d+)\].*?(\d\.\d+|1|0|-nan\(ind\)|nan|\d+\.\d+e-\d+)\s+text:\s(.*)"
    )
    result_type = ""
    for line in result_lines:
        for type, key in key_words.items():
            if key in line:
                result_type = type
        m = re.match(pattern, line)
        if m and result_type:
            results[result_type] = (
                [] if not results.get(result_type, []) else results[result_type]
            )
            lable, score, text = m.group(1), m.group(2), m.group(3).strip(", ")
            results[result_type].append(
                {"score": score, "lable": lable, "text": text, "line": line}
            )

    if results:
        return OrderedDict(sorted(results.items(), key=lambda t: t[0], reverse=True))
    else:
        return results


def analyze_parsed_failed_cases(results, result_lines):
    try:
        ipu_start = (
            ipu_end
        ) = (
            cpu_ep_start
        ) = cpu_ep_end = cpu_runner_start = cpu_runner_end = xmodel_diff_start = 0
        for index, line in enumerate(result_lines):
            if "test_dpu_ep" in line:
                ipu_start = index
            if "test_cpu_ep" in line:
                ipu_end = index - 1
                cpu_ep_start = index
            if "test_cpu_runner" in line:
                cpu_ep_end = index - 1
                cpu_runner_start = index
            if "test_xmodel_diff" in line:
                cpu_runner_end = index - 1
                xmodel_diff_start = index
        xmodel_diff_end = -3
        ipu_lines = []
        if ipu_start and ipu_end:
            ipu_lines = result_lines[ipu_start:ipu_end]
        if cpu_ep_start and cpu_ep_end:
            cpu_lines = result_lines[cpu_ep_start:cpu_ep_end]
        cpu_runner_lines = []
        if cpu_runner_start and cpu_runner_end:
            cpu_runner_lines = result_lines[cpu_runner_start:cpu_runner_end]
        xmodel_diff_lines = []
        if xmodel_diff_start and xmodel_diff_end:
            xmodel_diff_lines = result_lines[xmodel_diff_start:xmodel_diff_end]

        if not results.get("cpu", []):
            return "CPU_EP Error"  # no cpu ep result
        if results.get("cpu_runner", []) and (not results.get("ipu", [])):
            if xmodel_diff_lines:
                tool_name = "Test xmodel diff"
                timeout_layer = check_timeout(xmodel_diff_lines, tool_name)
                exception_pattern = ".*\(Exception\s+type.*"
                exception_match = check_failed(
                    exception_pattern, xmodel_diff_lines, tool_name
                )
                if timeout_layer != "":
                    return timeout_layer
                elif exception_match != "":
                    return exception_match
            return check_graph_engine(ipu_lines)
        if not results.get("cpu_runner", []):
            compile_start = False
            if not cpu_runner_lines:
                return "NO CPU_RUNNER Result"
            for line in ipu_lines:
                if "Begin to compile" in line:
                    compile_start = True
                if "Compile done" in line:
                    return "VitisAI EP Error"  # compiling done, no cpu_runner result
            else:
                if not compile_start:
                    return "VitisAI EP Error"  # compile not start
                return "xcompiler failed"  # compiling failed

        return "Unknown Error Type"
    except Exception as e:
        logging.warning(
            f"!!! warning : analyze_parsed_failed_cases function failed! {e}.)"
        )


def check_graph_engine(ipu_lines):
    try:
        tool_name = "VitisAI EP"
        error_message = ""
        compile_start = compile_end = False
        for line in ipu_lines:
            if "Begin to compile" in line:
                compile_start = True
            if "Compile done" in line:
                compile_end = True
        if compile_start and compile_end:
            timeout_layer = check_timeout(ipu_lines, tool_name)
            GE_error_pattern = r".*Error:.*"
            GE_error_match = check_failed(GE_error_pattern, ipu_lines, tool_name)
            if timeout_layer != "":
                error_message += timeout_layer
            if GE_error_match != "":
                error_message += GE_error_match
        else:
            check_fail_pattern = r".*Check\s+failed.*"
            fatal_error_pattern = r".*Fatal\s+error.*"
            exception_pattern = ".*\(Exception\s+type.*"
            check_fail_match = check_failed(check_fail_pattern, ipu_lines, tool_name)
            fatal_error_match = check_failed(fatal_error_pattern, ipu_lines, tool_name)
            exception_match = check_failed(exception_pattern, ipu_lines, tool_name)
            if check_fail_match != "":
                error_message += check_fail_match
            if fatal_error_match != "":
                error_message += fatal_error_match
            if exception_match != "":
                error_message += exception_match
        return error_message if error_message != "" else "VitisAI EP Error"

    except Exception as e:
        logging.warning(f"!!! warning : check_graph_engine function failed! {e}.)")


def check_timeout(result_lines, tool_name):
    timeout_layer_match = utility.pattern_match(r".*DPU\s+timeout:.*", result_lines)
    layer_name = ""
    if timeout_layer_match:
        pattern = r"\sDPU\s+timeout:.*"
        layer_name = (
            tool_name
            + " -------- "
            + re.search(pattern, timeout_layer_match.group().strip()).group()
        )
    return layer_name


def check_failed(search_pattern, result_lines, tool_name):
    fail_match = utility.pattern_match(search_pattern, result_lines)
    fail_mes = ""
    if fail_match:
        pattern = f"\s\S+\]\s+{search_pattern}"
        if re.search(pattern, fail_match.group().strip()):
            fail_mes = (
                tool_name
                + " -------- "
                + re.search(pattern, fail_match.group().strip()).group()
            )
        else:
            fail_mes = tool_name + " -------- " + fail_match.group().strip()
    return fail_mes


def is_match(results):
    assert len(results.get("ipu", [])) > 2 and len(results.get("cpu", [])) > 2
    ipu_results = results["ipu"]
    cpu_results = results["cpu"]
    cpu_runner_results = results["cpu_runner"]

    # step 1 check if ONNX ep is all the same with cpu_runer
    for index, ipu_result in enumerate(ipu_results):
        if (
            ipu_result.get("lable", "") != cpu_runner_results[index].get("lable", "")
            or ipu_result.get("text", "") != cpu_runner_results[index].get("text", "")
            or ipu_result.get("score", "") != cpu_runner_results[index].get("score", "")
        ):
            print("ERROR: ONNX ep not same with cpu_runer")
            onnxep_match_cpurunner_flag = False
            break
    else:
        onnxep_match_cpurunner_flag = True

    # step 2: check if ONNX ep with CPU EP
    for index, ipu_result in enumerate(ipu_results):
        if (
            ipu_result.get("lable", "") != cpu_results[index].get("lable", "")
            or ipu_result.get("score", "") != cpu_results[index].get("score", "")
            or ipu_result.get("text", "") != cpu_results[index].get("text", "")
        ):
            onnxep_diff_cpuep_flag = True
            break
    else:
        onnxep_diff_cpuep_flag = False

    return onnxep_match_cpurunner_flag, onnxep_diff_cpuep_flag


def get_run_time(model_result_dict, result_lines):
    try:
        # get run time COMPUTE
        ipu_compute_times = model_result_dict.get("COMPUTE_TIMES", {}).get("ipu", {})
        # print("compute_times %s" % model_result_dict["COMPUTE_TIMES"])
        # print("dpu_subgraph %s" % ipu_compute_times["dpu_subgraph"])
        if (
            not bool(ipu_compute_times.get("dpu_subgraph"))
            or ipu_compute_times.get("dpu_subgraph") == "0"
        ):
            print("dpu_subgraph == 0.")
            return "NA"
        elif not int(ipu_compute_times.get("dpu_subgraph")) == len(
            ipu_compute_times.get("computes", [])
        ):
            print("dpu_subgraph and COMPUTE are not equal.")
            return "NA"
        else:
            total_microseconds = 0
            for item in ipu_compute_times.get("computes", []):
                compute_value = int(item.get("compute", "0"))
                total_microseconds += compute_value
            total_seconds = total_microseconds / 1000000
            return total_seconds
    except Exception as e:
        logging.warning(f"!!! warning : get run time failed! {e}.)")


def get_test_time(model_result_dict, result_lines):
    # get time consumption
    if re.match("TEST_START_TIME=(.*)", result_lines[-3]):
        model_result_dict["TEST_START_TIME"] = (
            re.match("TEST_START_TIME=(.*)", result_lines[-3]).group(1).strip()
        )
    if re.match("TEST_END_TIME=(.*)", result_lines[-2]):
        model_result_dict["TEST_END_TIME"] = (
            re.match("TEST_END_TIME=(.*)", result_lines[-2]).group(1).strip()
        )

    if re.match("Build elapse seconds:(.*)", result_lines[-1]):
        elapse = (
            re.match("Build elapse seconds:(.*)", result_lines[-1]).group(1).strip()
        )
        model_result_dict["TEST_ELAPSED"] = int(float(elapse.strip()))
    else:
        elapse = "NA"
    return elapse


def make_op_analyze_table(total_op_on_cpu_counts, total_op_on_ipu_counts):
    if not total_op_on_cpu_counts:
        return ""
    op_tbody = ""
    print(total_op_on_cpu_counts)
    print(total_op_on_ipu_counts)

    op_merged_counts = {}
    for op_name, op_count_on_ipu in total_op_on_ipu_counts.items():
        op_merged_counts[op_name] = {}
        op_total = op_count_on_ipu
        if op_name in total_op_on_cpu_counts.keys():
            op_count_on_cpu = total_op_on_cpu_counts[op_name]
            op_total += op_count_on_cpu
            ipu_ratio = round((op_count_on_ipu / op_total) * 100, 2)
            total_op_on_cpu_counts.pop(op_name)
        else:
            ipu_ratio = 100
            op_count_on_cpu = 0
        op_merged_counts[op_name]["ipu_ratio"] = ipu_ratio
        op_merged_counts[op_name]["ipu"] = op_count_on_ipu
        op_merged_counts[op_name]["cpu"] = op_count_on_cpu
        op_merged_counts[op_name]["total"] = op_total

    if total_op_on_cpu_counts:
        for op_name, op_count_on_cpu in total_op_on_cpu_counts.items():
            op_merged_counts[op_name] = {}
            op_merged_counts[op_name]["ipu_ratio"] = 0
            op_merged_counts[op_name]["ipu"] = 0
            op_merged_counts[op_name]["cpu"] = op_count_on_cpu
            op_merged_counts[op_name]["total"] = op_count_on_cpu

    for op_name, op_counts_dict in sorted(
        op_merged_counts.items(),
        key=lambda kv: (kv[1].get("ipu_ratio", 0), kv[0]),
        reverse=True,
    ):
        op_tr = "<tr>"
        op_tr += "<td>%s</td>\n" % op_name
        op_tr += (
            '<td align="center">%s</td>\n' % op_counts_dict["ipu_ratio"]
        )  # ipu ratio
        op_tr += (
            '<td align="center">%s</td>\n' % op_counts_dict["ipu"]
            if op_counts_dict["ipu"]
            else "<td></td>\n"
        )  # ipu
        op_tr += (
            '<td align="center">%s</td>\n' % op_counts_dict["cpu"]
            if op_counts_dict["cpu"]
            else "<td></td>\n"
        )  # cpu
        op_tr += '<td align="center">%s</td>\n' % op_counts_dict["total"]  # total
        op_tr += "</tr>"
        op_tbody += op_tr

    return constant.OP_TABLE.format(op_tbody=op_tbody)


def add_dpu_ratio(log_path, model_name):
    try:
        vitisai_ep_report = os.path.join(log_path, model_name, "vitisai_ep_report.json")
        with open(vitisai_ep_report, "r") as report:
            json_file = json.load(report)
            dpu_nodeNum = "N/A"
            all_nodeNum = "N/A"
            for item in json_file["deviceStat"]:
                if item["name"] == "all":
                    all_nodeNum = item["nodeNum"] if item["nodeNum"] else "N/A"
                if item["name"] == "NPU":
                    dpu_nodeNum = item["nodeNum"] if item["nodeNum"] else "N/A"
            # print(f"dpu_nodeNum {dpu_nodeNum}")
            if all_nodeNum != "N/A" and dpu_nodeNum != "N/A":
                dpu_ratio = round((int(dpu_nodeNum) / int(all_nodeNum)) * 100, 2)
            else:
                dpu_ratio = "N/A"
        subg_td = (
            "\n <br> Ops on IPU ratio: %s" % dpu_ratio + "%"
            if dpu_ratio != "N/A"
            else "\n <br> Ops on IPU ratio: N/A"
        )
        # print(f"dpu_ratio {dpu_ratio}")
        return subg_td
    except Exception as e:
        logging.warning(f"open vitisai_ep_report.json failed!! --->  {e}.)")


def compare(
    model_list, log_path, compare_html, modelzoo_list, suite_run_elapsed="", run_date=""
):
    try:
        modelzoo_results = {}
        if not model_list:
            model_list = [x for x in os.listdir(log_path)]

        tbody = ""
        mismatch_num = cpu_all_num = rounding_mode_num = bad_result_num = 0
        compiler_fatal = dpu_multi_subgraph = 0
        # analyzer = onnx_ops_parse.OnnxModelOpsAnalyzer()
        for model_name in model_list:
            if not os.path.isdir(os.path.join(log_path, model_name)):
                continue
            print("compare method making model %s" % model_name, flush=True)
            modelzoo_results[model_name] = {}
            result_file = os.path.join(log_path, model_name, "build.log")
            # analyzer.analyze_model(os.path.join(log_path, model_name), model_name)
            if not os.path.exists(result_file):
                print("ERROR: %s no build.log" % model_name)
                continue
            model_tr = "<tr>"
            model_tr += "<td>%s</td>\n" % model_name
            onnx_model_info = utility.get_onnx_model_info(modelzoo_list, model_name)
            # check local onnx model md5
            onnx_model = onnx_model_info.get("onnx_model", "")
            onnx_model_md5 = onnx_model_info.get("md5sum", "")
            onnx_model_md5_td = onnx_model_md5[:7]
            onnx_model_name = os.path.split(onnx_model)[-1]
            onnx_model_local_path = os.path.join(log_path, model_name, onnx_model_name)
            if os.path.exists(onnx_model_local_path):
                onnx_model_local_md5 = utility.md5sum(onnx_model_local_path)
                if onnx_model_local_md5 != onnx_model_md5:
                    print(
                        "ERROR: onnx model md5 id not same with md5sum in modelzoo.json, %s"
                        % onnx_model_local_md5
                    )
                    onnx_model_md5_td = (
                        '<b><font color="red">%s(Need Update json!)</font></b>'
                        % onnx_model_local_md5[:7]
                    )

            # parse result from build.log
            result_lines = utility.read_lines(result_file)
            results = parse_result(result_lines)
            modelzoo_results[model_name]["Result"] = results
            build_url = os.environ.get(
                "BUILD_URL", "http://xcdl190259:8080/view/software/job/dw/"
            )
            modelzoo_results[model_name]["TEST_PATH"] = build_url
            compute_times = get_compute_time(result_lines)
            modelzoo_results[model_name]["COMPUTE_TIMES"] = compute_times

            # find the DPU subgraphnumber in result
            subg_num_in_xmodel = ""
            subg_num_on_ipu = 0
            dpu_subgraph_num_match = utility.pattern_match(
                r".*DPU\ssubgraph\snumber\s(\d+)", result_lines
            )
            if dpu_subgraph_num_match:
                subg_num_in_xmodel = dpu_subgraph_num_match.group(1)
            subg_num_on_ipu_match = utility.pattern_match(
                r".*\[XLNX_ONNX_EP_VERBOSE\]\sdpu\ssubgraph:\s(\d+)", result_lines
            )
            if subg_num_on_ipu_match:
                subg_num_on_ipu = int(subg_num_on_ipu_match.group(1).strip())
            subg_td = (
                "" if not subg_num_in_xmodel else "dpu subgraph:%s" % subg_num_in_xmodel
            )
            subg_td += "" if not subg_num_on_ipu else " - %s" % subg_num_on_ipu
            if utility.key_word_found(
                "[VITIS AI EP] This model is not a supported CNN model", result_lines
            ):
                subg_td += "num of conv <= 1"
            add_str = add_dpu_ratio(log_path, model_name)
            print("add_str: ", add_str)
            if add_str is not None and isinstance(add_str, str):
                subg_td += add_str
            else:
                print("ERROR! add_str!!")

            # no results or result format not correct
            if (not results) or (len(results.keys()) != 3):
                elapse = get_test_time(modelzoo_results[model_name], result_lines)
                run_time = get_run_time(modelzoo_results[model_name], result_lines)
                error_type = analyze_parsed_failed_cases(results, result_lines)
                if "timeout" in error_type:
                    modelzoo_results[model_name]["TEST_STATUS"] = "TIMEOUT"
                elif (
                    "Exception type" in error_type
                    or "Check failed" in error_type
                    or "Fatal error" in error_type
                    or "Error:" in error_type
                ):
                    modelzoo_results[model_name]["TEST_STATUS"] = "FAIL"
                else:
                    modelzoo_results[model_name]["TEST_STATUS"] = error_type
                model_tr += (
                    '<td><b><font color="red">%s</font></b></td>\n'
                    % modelzoo_results[model_name]["TEST_STATUS"]
                )
                error_type = (
                    error_type
                    if not subg_td
                    else f"{error_type}" + "<br>" + f"{subg_td}"
                )
                model_tr += "<td>%s</td>\n" % error_type
                model_tr += (
                    "<td> N/A </td>\n"
                    if not results or not results.get("cpu", [])
                    else "<td>%s</td>\n"
                    % [x.get("line", "") for x in results.get("cpu", [])][0]
                )
                model_tr += (
                    "<td> N/A </td>\n"
                    if not results or not results.get("cpu_runner", [])
                    else "<td>%s</td>\n"
                    % [x.get("line", "") for x in results.get("cpu_runner", [])][0]
                )
                model_tr += (
                    "<td> N/A</td>\n"
                    if not results or not results.get("ipu", [])
                    else "<td>%s</td>\n"
                    % [x.get("line", "") for x in results.get("ipu", [])][0]
                )
                model_tr += "<td>%s</td>\n" % elapse
                model_tr += "<td>%s</td>\n" % run_time
                model_tr += "<td>%s</td>\n" % onnx_model_md5_td
                model_tr += "</tr>"
                tbody += model_tr
                bad_result_num += 1

                modelzoo_results[model_name]["EXIT_CODE"] = 1
                modelzoo_results[model_name]["TEST_FIRST_ERROR_STRING"] = (
                    "Result Parse Fail:%s" % error_type
                )
                continue

            # make details table
            onnxep_match_cpurunner, onnxep_diff_cpuep = is_match(results)
            err_msg = ""
            if not onnxep_match_cpurunner:
                status = "FAIL"
                exit_code = 1
                mismatch_num += 1
                err_msg = "mismatch"
                model_tr += '<td ><b><font color="red">compiler functional mismatch</font></b></td>\n'
                model_tr += (
                    "<td > </td>\n" if not subg_td else "<td >%s</td>\n" % subg_td
                )
            elif onnxep_diff_cpuep:
                status = "PASS"
                exit_code = 0
                rounding_mode_num += 1
                model_tr += "<td ><b><font>rounding mode</font></b></td>\n"
                model_tr += f"<td >{subg_td}</td>\n"
            else:
                status = "PASS"
                exit_code = 0
                # check if compiler check fatal
                if utility.key_word_found(
                    "[XLNX_ONNX_EP_VERBOSE] dpu subgraph: 0", result_lines
                ) or (
                    not utility.key_word_found(
                        "[XLNX_ONNX_EP_VERBOSE] dpu subgraph", result_lines
                    )
                ):
                    err_msg = "Fall Back CPU"
                    if utility.key_word_found(
                        "Catch fatal exception", result_lines
                    ) or utility.key_word_found("catch other exception", result_lines):
                        cpu_all_num += 1
                        compiler_fatal += 1
                        error_str = "Catch fatal exception"
                        m = utility.pattern_match(
                            r"F.*\s(.*?\.cpp):(\d+).*?", result_lines
                        )
                        n = utility.pattern_match(
                            r".*\s(.*?\.cpp):(\d+).*catch\sother\sexception.*?",
                            result_lines,
                        )
                        if m:
                            err_msg = "Fall Back CPU:%s:%s" % (m.group(1), m.group(2))
                            error_str = "%s:%s" % (m.group(1), m.group(2))
                        if n:
                            err_msg = "Fall Back CPU:%s:%s" % (n.group(1), n.group(2))
                            error_str = "%s:%s" % (n.group(1), n.group(2))
                        model_tr += (
                            '<td ><b><font color="red">CPU All</font></b></td>\n'
                        )
                        model_tr += (
                            "<td style=text-align:left><b><font>xcompiler check fatal:%s</font></b></td>\n"
                            % error_str
                        )
                    else:
                        cpu_all_num += 1
                        dpu_multi_subgraph += 1
                        model_tr += (
                            '<td ><b><font color="red">CPU All</font></b></td>\n'
                        )
                        model_tr += (
                            f"<td >{subg_td}</td>\n"
                            if subg_td
                            else "<td >unknown CPU ALL type</td>\n"
                        )
                        err_msg = (
                            f"Fall Back CPU:{subg_td}" if subg_td else "Fall Back CPU"
                        )
                # if got [XLNX_ONNX_EP_VERBOSE] dpu subgraph: 1, status is rounding mode
                elif subg_num_on_ipu_match and subg_num_on_ipu > 0:
                    rounding_mode_num += 1
                    model_tr += "<td ><b><font>rounding mode</font></b></td>\n"
                    model_tr += f"<td >{subg_td}</td>\n" if subg_td else "<td ></td>\n"
                else:
                    err_msg = "Fall Back CPU"
                    cpu_all_num += 1
                    dpu_multi_subgraph += 1
                    model_tr += '<td ><b><font color="red">CPU All</font></b></td>\n'
                    model_tr += "<td style=text-align:left><b><font>unknown error type</font></b></td>\n"

            modelzoo_results[model_name]["TEST_STATUS"] = status
            modelzoo_results[model_name]["EXIT_CODE"] = exit_code
            model_tr += (
                "<td style=text-align:left>%s</td>\n"
                % [x.get("line", "") for x in results.get("cpu", [])][0]
            )
            model_tr += (
                "<td style=text-align:left>%s</td>\n"
                % [x.get("line", "") for x in results.get("cpu_runner", [])][0]
            )
            model_tr += (
                "<td style=text-align:left>%s</td>\n"
                % [x.get("line", "") for x in results.get("ipu", [])][0]
            )
            elapse = get_test_time(modelzoo_results[model_name], result_lines)
            run_time = get_run_time(modelzoo_results[model_name], result_lines)
            model_tr += "<td ><font>%s</font></td>\n" % elapse
            model_tr += "<td ><font>%s</font></td>\n" % run_time
            model_tr += "<td ><font>%s</font></td>\n" % onnx_model_md5_td

            if err_msg:
                modelzoo_results[model_name]["TEST_FIRST_ERROR_STRING"] = err_msg
            model_tr += "</tr>"
            tbody += model_tr
        # make the error type in CPU ALL
        error_type_info = ""
        if cpu_all_num:
            error_type_info += '<h3>Error Type in "CPU All"</h3>'
            if compiler_fatal:
                error_type_info += (
                    "<font>xcompiler check fatal :  %s </font> <br>" % compiler_fatal
                )
            if dpu_multi_subgraph:
                error_type_info += (
                    "<font>other CPU ALL cases :  %s </font> <br>" % dpu_multi_subgraph
                )

        # op_analyze_table = make_op_analyze_table(analyzer.total_op_on_cpu_counts, analyzer.total_op_on_ipu_counts)
        op_analyze_table = ""

        target_info = utility.make_target_info(modelzoo_list)
        target_type = (
            "STRIX"
            if not os.environ.get("TARGET_TYPE", "")
            else os.environ["TARGET_TYPE"]
        )
        dpu_type = os.environ.get("XLNX_VART_FIRMWARE", "1x4")
        opt_level = os.environ.get("OPT_LEVEL", "0")
        modelzoo = os.environ.get("MODEL_ZOO", "")
        report_time = time.strftime("%Y-%m-%d %H:%M", time.localtime())
        subgraph_num = os.environ.get("DPU_SUBGRAPH_NUM", "")
        run_type = f"{target_type} {modelzoo.upper()} {dpu_type} OPT{opt_level} {subgraph_num} {report_time}"
        run_date_str = "" if not run_date else "Test Date: %s\n" % run_date
        elapsed_str = (
            "" if not suite_run_elapsed else "Suite Run Elapsed: %s" % suite_run_elapsed
        )
        # write report
        jenkins_url = (
            os.environ.get("BUILD_URL", "http://xcdl190259:8080/view/software/job/dw/")
            + "artifact/"
            + compare_html
        )
        total = mismatch_num + cpu_all_num + rounding_mode_num + bad_result_num
        paras = {
            "title": "match compare",
            "tbody_content": tbody,
            "total": total,
            "mismatch_num": mismatch_num,
            "cpu_all_num": cpu_all_num,
            "rounding_mode_num": rounding_mode_num,
            "bad_result_num": bad_result_num,
            "error_type_info": error_type_info,
            "target_info": target_info,
            "jenkins_url": jenkins_url,
            "run_type": run_type,
            "run_date": run_date_str,
            "elapsed": elapsed_str,
            "op_analyze_table": op_analyze_table,
        }
        utility.write_file(compare_html, constant.TABLE_HTML_TEMPLATE.format(**paras))

        print(modelzoo_results)
        return modelzoo_results
    except Exception:
        tb = traceback.format_exc()
        if not tb is None:
            print(tb)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log_path", type=str, default="", required=True, help="log path."
    )
    parser.add_argument(
        "--compare_html", type=str, default="test.html", help="match compare report."
    )
    parser.add_argument("--model_list", type=str, default="", help="model list.")
    parser.add_argument("--modelzoo", type=str, default="", help="model list.")
    args = parser.parse_args()
    model_list = None if not args.model_list else args.model_list.split(" ")
    modelzoo_list = []
    with open(args.modelzoo, "r") as mz:
        modelzoo_list = json.load(mz)

    compare(model_list, args.log_path, args.compare_html, modelzoo_list)
