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


import traceback
import json
import sys
import os
import logging
import numpy as np
from pathlib import Path
import time

CWD = Path.cwd()
from . import utility
from . import parse_ops
from . import compare_with_baseline


def post_process_of_model_L(log_path, model_name, output_checking, mode=""):
    psnr = ""
    lpips = ""

    output_checking_list = [x.strip() for x in output_checking.split(",")]
    try:
        utility.pip_install_repos(["opencv-python", "lpips", "natsort", "scikit-image"])
    except Exception as e:
        print(e)

    if output_checking == "cpu_ep,cpu_ep":
        return psnr, lpips
    # convert the other test mode result vitisi_ep or cpu_runner
    if "cpu_ep" in output_checking_list:
        output_checking_list.remove("cpu_ep")
    cmp_mode = output_checking_list[0]
    if cmp_mode == "onnx_ep":
        cmp_mode = mode if mode else "vitisai_ep"
    cmp_mode_png = os.path.join(log_path, model_name, cmp_mode + "_png")
    if os.path.exists(os.path.join(log_path, model_name, cmp_mode)):
        utility.bin2png(os.path.join(log_path, model_name, cmp_mode), cmp_mode_png)
    else:
        print(f"{os.path.join(log_path, model_name, cmp_mode)} non-existent")

    vart_hello_world_path = (
        Path(os.environ.get("WORKSPACE", ""))
        / "win24_drop"
        / "examples"
        / "apps"
        / "vart_hello_world"
    )
    if not os.path.exists(vart_hello_world_path):
        print("ERROR:not found vart_hello_world script")
        return psnr, lpips

    try:
        sys.path.append(str(vart_hello_world_path).replace("/", "\\"))

        import post_process_of_model_L

        L_locol_golden = r"c:\win24_models_golden\L_v_1_0_golden"
        if not os.path.exists(r"c:\win24_models_golden"):
            os.makedirs(r"c:\win24_models_golden")
        if not os.path.exists(L_locol_golden):
            L_host_golden = "/group/dphi_software/win24_cpu_ep_golden/L_v_1_0_golden"
            utility.scp_dir("xcdl190074.xilinx.com", L_host_golden, L_locol_golden)

        assert os.path.exists(L_locol_golden)
        print("measure %s accuracy" % model_name)
        results, per_image_res = post_process_of_model_L.measure_dirs(
            L_locol_golden,
            cmp_mode_png,
            use_gpu=False,
            verbose=True,
            use_im8=True,
            img_ext="png",
        )

        psnr = round(np.mean([result["psnr"] for result in results]), 4)
        lpips = round(np.mean([result["lpips"] for result in results]), 5)
        return psnr, lpips
    except Exception as e:
        print(e)
    finally:
        return psnr, lpips


def json_to_html_summary(json_data, jenkins_url):
    html = f"<h4>Jenkins Pipeline URL: {jenkins_url}</h4>"
    html += '<p>Summary:</p> <table border="1">'
    data = json.loads(json_data)
    # head
    html += "<tr>"
    html += "<th>Type</th>"
    html += "<th>Amount</th>"
    html += "</tr>"

    # content
    dic = {}
    for item in data:
        key = item.get("result", "NA").split("::")[0]
        if key.find("compile") != -1:
            key = "FAIL@xcompiler_compile_xmodel"
        if key in dic:
            dic[key] += 1
        else:
            dic[key] = 1
    for val in dic.keys():
        html += "<tr>"
        html += f"<td>{val}</td>"
        html += f"<td>{dic[val]}</td>"
        html += "</tr>"
    html += "</table>"
    return html


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
                if item["name"] == "DPU":
                    dpu_nodeNum = item["nodeNum"] if item["nodeNum"] else "N/A"
            # print(f"dpu_nodeNum {dpu_nodeNum}")
            if all_nodeNum != "N/A" and dpu_nodeNum != "N/A":
                dpu_ratio = round((int(dpu_nodeNum) / int(all_nodeNum)) * 100, 2)
            else:
                dpu_ratio = "N/A"
        subg_td = f"{dpu_ratio} %" if dpu_ratio != "N/A" else "N/A"
        # print(f"dpu_ratio {dpu_ratio}")
        return subg_td
    except Exception as e:
        logging.warning(f"open vitisai_ep_report.json failed!! --->  {e}.)")


def add_pdi(log_path, model_name, model_benchmark_dict):
    try:
        result_file = os.path.join(log_path, model_name, "build.log")
        if os.path.exists(result_file):
            model_tr = ""
            result_lines = utility.read_lines(result_file)
            # real DPU subg
            subg_num_on_ipu_match = utility.pattern_match(
                r".*\[XLNX_ONNX_EP_VERBOSE\]\sdpu\ssubgraph:\s(\d+)", result_lines
            )
            if subg_num_on_ipu_match:
                real_dpu = subg_num_on_ipu_match.group(1).strip()
                real_dpu_num = int(real_dpu)
                if real_dpu_num <= 0:
                    real_dpu_num = '<b><font color="red">0</font></b>'
            else:
                real_dpu_num = '<b><font color="red">NA</font></b>'
            # print("real DPU  ------------->", real_dpu_num)
            subg_num_on_ipu_td = '<td align ="center"><b><font>%s</font></b></td>' % (
                real_dpu_num
            )
            model_tr += subg_num_on_ipu_td
            model_benchmark_dict["Summary"]["DPU_Subgraph"] = {
                "On_IPU": real_dpu_num,
            }

            #  xcompiler DPU subg
            subg_total_DPU_match = utility.pattern_match(
                r".*\sDPU\ssubgraph\snumber\s(\d+)", result_lines
            )
            if subg_total_DPU_match:
                # print(subg_total_DPU_match.group(1), "-----------------------------")
                total_dpu = subg_total_DPU_match.group(1).strip()
                total_dpu_num = int(total_dpu)
            else:
                total_dpu_num = '<b><font color="red">NA</font></b>'
            # print("xcompiler DPU  ------------->", total_dpu_num)
            subg_total_dpu_td = '<td align ="center"><b><font>%s</font></b></td>' % (
                total_dpu_num
            )
            model_tr += subg_total_dpu_td + "\n"
            model_benchmark_dict["Summary"]["DPU_Subgraph"]["xcompiler"] = total_dpu_num

            #  real pdi
            subg_vai_PDI_match = utility.pattern_match(
                r".*\stotal_pdi_swaps:\s(\d+)", result_lines
            )
            if subg_vai_PDI_match:
                # print(subg_vai_PDI_match.group(1), "-----------------------------")
                vai_pdi = subg_vai_PDI_match.group(1).strip()
                vai_pdi_num = int(vai_pdi)
            else:
                vai_pdi_num = "NA"
            # print("real pdi ------------->", vai_pdi_num)
            subg_vai_PDI_td = '<td align="center"><b><font>%s</font></b></td>' % (
                vai_pdi_num
            )
            model_tr += subg_vai_PDI_td + "\n"
            model_benchmark_dict["Summary"]["PDI_Number"] = {
                "On_IPU": vai_pdi_num,
            }

            #   xcompiler pdi
            subg_num_on_PDI_match = utility.pattern_match(
                r".*\stotal\snumber\sof\sPDI\sswaps\s(\d+)", result_lines
            )
            if subg_num_on_PDI_match:
                # print(subg_num_on_PDI_match.group(1), "-----------------------------")
                xcompiler_pdi = subg_num_on_PDI_match.group(1).strip()
                xcompiler_pdi_num = int(xcompiler_pdi)
            else:
                xcompiler_pdi_num = "NA"
            # print("xcompiler pdi ------------->", xcompiler_pdi_num)
            subg_total_PDI_td = '<td align="center"><b><font>%s</font></b></td>' % (
                xcompiler_pdi_num
            )
            model_tr += subg_total_PDI_td + "\n"
            model_benchmark_dict["Summary"]["PDI_Number"][
                "xcompiler"
            ] = xcompiler_pdi_num

            #  xcompiler CPU subg
            subg_total_CPU_match = utility.pattern_match(
                r".*\sCPU\ssubgraph\snumber\s(\d+)", result_lines
            )
            if subg_total_CPU_match:
                # print(subg_total_CPU_match.group(1), "-----------------------------")
                total_cpu = subg_total_CPU_match.group(1).strip()
                total_cpu_num = int(total_cpu)
            else:
                total_cpu_num = '<b><font color="red">NA</font></b>'
            # print("xcompiler CPU  ------------->", total_cpu_num)
            subg_num_on_cpu_td = '<td align ="center"><b><font>%s</font></b></td>' % (
                total_cpu_num
            )
            model_tr += subg_num_on_cpu_td + "\n"
            model_benchmark_dict["Summary"]["CPU_Subgraph"] = total_cpu_num

            return model_tr
    except Exception as e:
        logging.warning(f"open build.log failed!! --->  {e}.)")


def json_to_html(
    json_data,
    target_info,
    log_path,
    output_checking,
    system_version,
    verbose,
    xcompiler_optimizations,
    modelzoo_results_dict=None,
):
    data = json.loads(json_data)
    html = "<h3>Target Info:</h3>"
    html += f"{target_info}"
    html += "<br><span>System Version:</span>"
    html += f"{system_version}"
    html += "<br><br>"
    html += f"{verbose}"
    html += "<br><br>"
    html += f"{xcompiler_optimizations}"

    tol_val = (
        data[0].get("tolerance", "NA").split("::")[0]
        if data[0].get("tolerance", "NA") is not None
        else "NA"
    )
    if tol_val != "NA":
        html += f"<p>Tolerance:{tol_val}</p> "
    html += f'<p>Details:</p> <table border="1">'
    # print("data------->", data)
    output_checking_list = output_checking.upper().replace(",", " VS ")
    if os.environ.get("OUTPUT_CHECKING", "cpu_ep,onnx_ep").find("cpu_ep") != -1:
        test_mode2 = "CPU_EP"
    else:
        test_mode2 = "CPU_RUNNER"

    # head
    html += "<thead>"
    html += "<tr>"
    html += '<th rowspan="2">ModelName</th>'
    html += '<th rowspan="2">Result</th>'
    html += '<th rowspan="2">Md5 Compare</th>'
    html += '<th colspan="2" style="width:200px;">TARGET</th>'
    html += f'<th colspan="2" style="width:200px;">{output_checking_list}</th>'
    if (
        "onnx_ep" in output_checking
        and os.environ.get("TEST_HELLO_WORLD", "true") == "true"
    ):
        html += f'<th colspan="2">Hello World VS {test_mode2}</th>'
    html += '<th rowspan="2">Ops on IPU ratio</th>'
    html += '<th colspan="2" style="width:200px;">DPU subgraph</th>'
    html += '<th colspan="2" style="width:200px;">PDI SWAP</th>'
    html += '<th style="width:200px;"> CPU subgraph</th>'
    html += "</tr>"
    html += "<tr>"
    html += '<th style="width:150px;">psnr/snr</th>'
    html += '<th style="width:150px;">l2norm/lpips</th>'
    html += '<th style="width:150px;">psnr/snr</th>'
    html += '<th style="width:150px;">l2norm/lpips</th>'
    if (
        "onnx_ep" in output_checking
        and os.environ.get("TEST_HELLO_WORLD", "true") == "true"
    ):
        html += '<th style="width:150px;">psnr/snr</th>'
        html += '<th style="width:150px;">l2norm/lpips</th>'

    html += '<th style="width:150px;">On IPU</th>'
    html += '<th style="width:150px;">Xcompiler</th>'
    html += '<th style="width:150px;">On IPU</th>'
    html += '<th style="width:150px;">Xcompiler</th>'
    html += '<th style="width:150px;">Xcompiler</th>'
    html += "</tr>"
    html += "</thead>"

    # content
    for item in data:
        # print("start to parse model:%s" % item["id"])
        model_benchmark_dict = {}
        modelzoo_results_dict[item["id"]] = model_benchmark_dict

        html += "<tr>"
        html += f'<td>{item["id"]}</td>'
        res_val = (
            item.get("result", "NA").split("::")[0]
            if item.get("result", "NA") is not None
            else "NA"
        )
        md5_val = (
            item.get("md5_compare", "NA").split("::")[0]
            if item.get("md5_compare", "NA") is not None
            else "NA"
        )
        snr_target = (
            item.get("snr_target", "NA")
            if item.get("snr_target", "NA") is not None
            else "NA"
        )
        if snr_target == "":
            snr_target = "NA"

        psnr_target = (
            item.get("psnr_target", "NA")
            if item.get("psnr_target", "NA") is not None
            else "NA"
        )
        if psnr_target == "":
            psnr_target = "NA"

        l2norm_target = (
            item.get("l2norm_target", "NA")
            if item.get("l2norm_target", "NA") is not None
            else "NA"
        )

        snr_val = item.get("snr", "NA") if item.get("snr", "NA") is not None else "NA"
        psnr_val = (
            item.get("psnr", "NA") if item.get("psnr", "NA") is not None else "NA"
        )
        l2norm_val = (
            item.get("l2norm", "NA") if item.get("l2norm", "NA") is not None else "NA"
        )
        l2norm_hw_val = (
            item.get("l2norm_hw", "NA")
            if item.get("l2norm_hw", "NA") is not None
            else "NA"
        )
        snr_hw_val = (
            item.get("snr_hw", "NA") if item.get("snr_hw", "NA") is not None else "NA"
        )
        psnr_hw_val = (
            item.get("psnr_hw", "NA") if item.get("psnr_hw", "NA") is not None else "NA"
        )
        html += f"<td>{res_val}</td>"
        html += f"<td>{md5_val}</td>"
        if snr_target != "NA":
            html += f"<td>{snr_target}</td>"
        else:
            html += f"<td>{psnr_target}</td>"

        # print("item ---> ", item, flush=True)
        if item.get("acc_type", "") == "lpips":
            # for L v1.0, get lpips instead of l2norm
            L_psnr, L_lpips = post_process_of_model_L(
                log_path, item["id"], output_checking
            )
            psnr_val = L_psnr if L_psnr else "NA"
            l2norm_val = L_lpips if L_lpips else "NA"
            l2norm_target = item.get("lpips_target", "")
            # print(item)
            if (
                "onnx_ep" in output_checking
                and os.environ.get("TEST_HELLO_WORLD", "true") == "true"
            ):
                L_psnr, L_lpips = post_process_of_model_L(
                    log_path, item["id"], output_checking, "hello_world"
                )
                psnr_hw_val = L_psnr if L_psnr else "NA"
                l2norm_hw_val = L_lpips if L_lpips else "NA"

        if l2norm_target == "":
            l2norm_target = "NA"
        html += f"<td>{l2norm_target}</td>"

        accuracy_result_dict = {}
        if snr_val != "NA":
            html += f"<td>{snr_val}</td>"
            accuracy_result_dict["snr"] = snr_val
        else:
            html += f"<td>{psnr_val}</td>"
            accuracy_result_dict["psnr"] = psnr_val
        html += f"<td>{l2norm_val}</td>"
        accuracy_result_dict["l2norm"] = l2norm_val

        if (
            "onnx_ep" in output_checking
            and os.environ.get("TEST_HELLO_WORLD", "true") == "true"
        ):
            if snr_hw_val and snr_hw_val != "NA":
                html += f"<td>{snr_hw_val}</td>"
                accuracy_result_dict["snr_helloworld"] = snr_hw_val
            else:
                html += f"<td>{psnr_hw_val}</td>"
                accuracy_result_dict["psnr_helloworld"] = psnr_hw_val
            html += f"<td>{l2norm_hw_val}</td>"
        accuracy_result_dict["l2norm_helloworld"] = l2norm_hw_val
        model_benchmark_dict["Summary"] = {
            "Accuracy": accuracy_result_dict,
        }

        ops = parse_ops.parse_ops(item["id"], log_path)
        html += f"<td>{ops}%</td>" if ops else f"<td>NA</td>"
        model_benchmark_dict["Summary"]["OPS"] = ops

        html += f'{add_pdi(log_path, item["id"], model_benchmark_dict)}'
        html += "</tr>"

    html += "</table>"
    return html


def json_to_html_l2table(json_data, log_path):
    try:
        data = json.loads(json_data)
        html = ""
        # head
        for item in data:
            print(item["id"], flush=True)
            html += f'<p>{item["id"]} l2norm Details:</p> <table border="1">'
            html += "<thead>"
            html += "<tr>"
            html += "<th>Id</th>"
            l2norm_detail = item.get("l2norm_detail")
            if l2norm_detail is None:
                l2norm_detail = {}
            for key in l2norm_detail.keys():
                html += f"<th>output{key}</th>"
            html += '<th style="width:500px;">Input</th>'
            html += "</tr>"
            html += "</thead>"

            input_seq = str(os.environ.get("SET_SEQ", "0"))
            test_mode = os.environ.get("TEST_MODE", "performance")
            if input_seq != "0" or test_mode == "functionality":
                html += "<tr>"
                html += f"<td>{int(input_seq)}</td>"
                for key, vals in l2norm_detail.items():
                    val = str(vals[0])
                    print("val---->", val, flush=True)
                    html += f"<td>{val}</td>"
                if item.get("accuracy_input"):
                    inputs_str = "\n".join(item.get("accuracy_input")[input_seq])
                else:
                    inputs_str = ""
                html += f"<td>{inputs_str}</td>"
                html += "</tr>"
            elif (
                "accuracy" in test_mode
                or os.environ.get("ACCURACY_TEST", "false") == "true"
            ):
                con = 0
                for num, inputs in item.get("accuracy_input", {}).items():
                    html += "<tr>"
                    html += f"<td>{num}</td>"
                    for key, vals in l2norm_detail.items():
                        # val = vals[int(num)] if int(num) < len(vals) else "NA"
                        val = vals[int(con)] if int(con) < len(vals) else "NA"
                        html += f"<td>{val}</td>"
                    con += 1
                    if inputs is None:
                        inputs = []
                    inputs_str = "\n".join(inputs)
                    html += f"<td>{inputs_str}</td>"
                    html += "</tr>"

            html += "</table>"
            html += "<b></b>"
        return html
    except Exception as e:
        tb = traceback.format_exc()
        if not tb is None:
            print(tb)
        logging.warning(f"write l2norm detail failed!! --->  {e}.)")


def save_html_file(html_content, file_path, type):
    with open(file_path, type) as file:
        file.write(html_content)


def main(json_file, html_file, log_path="", baseline_benchmark_data=None):
    # read json
    print(json_file, flush=True)
    with open(json_file, "r") as file:
        list_json_data = json.load(file)

    job_base_name = os.environ.get("JOB_BASE_NAME", "")
    print(f"job_base_name is {job_base_name}", flush=True)

    for item in list_json_data:
        # print(f"item is {item}", flush=True)
        if not isinstance(item, dict):
            print(f"Skipping non-dict item: {item}")
            continue
        target_path = f"C:\\IPU_workspace\\{job_base_name}\\vaip_regression\\{item['id']}\\fast_accuracy_summary.json"
        print(f"target_path is {target_path}", flush=True)
        if os.path.exists(target_path):
            print(f"matched")
            with open(target_path, "r") as file:
                fast_acccuracy_data = json.load(file)
            item["snr"] = fast_acccuracy_data.get("snr")
            item["psnr"] = fast_acccuracy_data.get("psnr")
            item["l2norm"] = fast_acccuracy_data.get("l2norm")
            item["l2norm_detail"] = fast_acccuracy_data.get("l2norm_detail")
            # print(f"replaced item is {item}", flush=True)

    with open(json_file, "w") as file:
        json.dump(list_json_data, file, indent=4)
    with open(json_file, "r") as file:
        json_data = file.read()

    # print(json_data, flush=True)
    # json to html

    test_mode = os.environ.get("TEST_MODE", "performance")
    default_output_checking = {"performance": "cpu_runner,onnx_ep"}
    output_checking_env = os.environ.get("OUTPUT_CHECKING")
    if output_checking_env == "true":
        output_checking = default_output_checking.get(test_mode, "cpu_ep,onnx_ep")
    else:
        output_checking = (
            output_checking_env
            if output_checking_env
            else default_output_checking.get(test_mode, "cpu_ep,onnx_ep")
        )
    target_info = ""
    modelzoo_list = []
    modelzoo = os.environ.get("MODEL_ZOO", "")
    modelzoo_json = os.path.join(CWD, "ci", modelzoo + ".json")
    if (not modelzoo) or (not os.path.exists(modelzoo_json)):
        logging.warning("No MODEL_ZOO got in env!")
    else:
        with open(modelzoo_json, "r") as mz:
            modelzoo_list = json.load(mz)
        target_info = utility.make_target_info(modelzoo_list)
    jenkins_url = (
        os.environ.get("BUILD_URL", "http://xcdl190259:8080/view/software/job/dw/")
        + "artifact/"
        + html_file
    )
    modelzoo_results_dict = {}
    system_version, verbose, xcompiler_optimizations = utility.get_verbose()
    html_summary = json_to_html_summary(json_data, jenkins_url)
    html_content = json_to_html(
        json_data,
        target_info,
        log_path,
        output_checking,
        system_version,
        verbose,
        xcompiler_optimizations,
        modelzoo_results_dict,
    )
    l2norm_detail = json_to_html_l2table(json_data, log_path)
    dpu_type = os.environ.get("XLNX_VART_FIRMWARE", "").strip().split("\\")[-1]
    target_type = os.environ.get("TARGET_TYPE", "STRIX")
    run_date = time.strftime("%Y-%m-%d", time.localtime())
    job_name = os.environ.get("JOB_BASE_NAME", "dw")
    build_id = os.environ.get("BUILD_ID", "999")
    modelzoo_results_dict["VERSION_INFO"] = utility.get_version_info()
    modelzoo_results_dict["XCOMPILER_ATTRS"] = utility.html_to_json(verbose)

    modelzoo_results_dict["TEST_MODE"] = os.environ.get("TEST_MODE", "")
    modelzoo_results_dict["REGRESSION_TYPE"] = os.environ.get("REGRESSION_TYPE", "")
    modelzoo_results_dict["MODEL_GROUP"] = os.environ.get("MODEL_GROUP", "")
    modelzoo_results_dict["TARGET_TYPE"] = target_type.capitalize()
    if dpu_type:
        modelzoo_results_dict["TARGET_NAME"] = os.path.splitext(dpu_type)[0]
    modelzoo_results_dict["RUN_DATE"] = run_date
    modelzoo_results_dict["MODEL_ZOO"] = os.environ.get("MODEL_ZOO", "")
    modelzoo_results_dict["BUILD_ID"] = build_id
    modelzoo_results_dict["JOB_BASE_NAME"] = job_name
    modelzoo_results_dict["BUILD_USER"] = os.environ.get("BUILD_USER", "")
    modelzoo_results_dict["BUILD_URL"] = os.environ.get("BUILD_URL", "")
    print("modelzoo_results ----------------------", modelzoo_results_dict)

    json_to_save_data = os.environ.get(
        "BENCHMARK_RESULT_JSON",
        f"benchmark_result_{job_name}_{build_id}_{run_date}.json",
    )
    print("benchmark result json ----------------------", json_to_save_data)
    utility.dump_dict_to_json(modelzoo_results_dict, json_to_save_data)

    print(f"baseline_benchmark_data {baseline_benchmark_data}")
    compare_table = ""
    if (
        baseline_benchmark_data
        and os.path.exists(baseline_benchmark_data)
        and os.path.exists(json_to_save_data)
    ):
        if "performance" in test_mode:
            compare_table, compare_data = compare_with_baseline.compare_with_baseline(
                baseline_benchmark_data, json_to_save_data, "performance"
            )
            if test_mode != "performance":
                opera_json(json_to_save_data, compare_data)
        if "accuracy" or "functionality" in test_mode:
            compare_table, compare_data = compare_with_baseline.compare_with_baseline(
                baseline_benchmark_data, json_to_save_data, "accuracy"
            )
            opera_json(json_to_save_data, compare_data)
    # save
    save_html_file(html_summary, html_file, "w")
    save_html_file(html_content, html_file, "a")
    save_html_file(compare_table, html_file, "a")
    save_html_file(l2norm_detail, html_file, "a")

    print(f"done ------> {html_file}")


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


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 jsonToHtml.py json_file [result_file].")
        sys.exit()

    main(sys.argv[1], sys.argv[2])
