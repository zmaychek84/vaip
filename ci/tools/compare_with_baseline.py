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
import re
import logging
import json

from . import constant


def make_thead(test_mode, target_type="", dpu_type=""):
    thead_tr_col1 = f'<th rowspan="3" style="width:200px;">{target_type} {dpu_type}<br>Model Name</th>\n'

    if "performance" in test_mode:
        # latency compare thead
        thead_tr_col1 += '<th colspan="4" style="width:200px;">Latency(ms)</th>\n'
        thead_tr_col2 = '<th colspan="2" style="width:100px;">Silicon Time</th>\n'
        thead_tr_col2 += '<th colspan="2" style="width:100px;">vart</th>\n'
        thead_tr_col3 = '<th style="width:100px;">Baseline</th>\n'
        thead_tr_col3 += '<th style="width:100px;">Current (ratio)</th>\n'
        thead_tr_col3 += '<th style="width:100px;">Baseline</th>\n'
        thead_tr_col3 += '<th style="width:100px;">Current (ratio)</th>\n'

        # throughput compare thead
        thead_tr_col1 += (
            '<th rowspan="2" colspan="2" style="width:300px;">Throughput(fps)</th>\n'
        )
        thead_tr_col3 += '<th style="width:100px;">Baseline</th>\n'
        thead_tr_col3 += '<th style="width:100px;">Current (ratio)</th>\n'
    elif "accuracy" or "functionality" in test_mode:
        # latency compare thead
        thead_tr_col1 += '<th colspan="4" style="width:200px;">Accuracy</th>\n'
        thead_tr_col2 = '<th colspan="2" style="width:100px;">psnr/snr</th>\n'
        thead_tr_col2 += '<th colspan="2" style="width:100px;">l2norm/lpips</th>\n'
        thead_tr_col3 = '<th style="width:100px;">Baseline</th>\n'
        thead_tr_col3 += '<th style="width:100px;">Current</th>\n'
        thead_tr_col3 += '<th style="width:100px;">Baseline</th>\n'
        thead_tr_col3 += '<th style="width:100px;">Current</th>\n'
    else:
        print("ERROR: not valid test mode")
        return

    # make subgraph thead
    thead_tr_col1 += (
        '<th rowspan="2" colspan="2" style="width:200px;">DPU subgraph</th>\n'
    )
    thead_tr_col1 += '<th rowspan="2" colspan="2" style="width:200px;">PDI SWAP</th>\n'
    thead_tr_col1 += '<th rowspan="2" style="width:200px;"> CPU subgraph</th>\n'

    thead_tr_col3 += f'<th style="width:150px;">On IPU</th>'
    thead_tr_col3 += f'<th style="width:150px;">Xcompiler</th>'
    thead_tr_col3 += f'<th style="width:150px;">On IPU</th>'
    thead_tr_col3 += f'<th style="width:150px;">Xcompiler</th>'
    thead_tr_col3 += f'<th style="width:150px;">Xcompiler</th>'

    thead_tr_col1 += '<th rowspan="3" style="width:200px;">OPS on IPU</th>\n'
    if "performance" in test_mode:
        thead_tr_col1 += '<th rowspan="3" style="width:200px;">onnx md5</th>\n'
    return thead_tr_col1, thead_tr_col2, thead_tr_col3


def make_model_tr(
    model_name,
    test_mode,
    model_result_benchmark,
    model_baseline_benchmark,
    compare_item,
):
    compare_con = {}
    model_tr = "<tr>"
    model_tr += f"<td>{model_name}</td>\n"
    if test_mode == "performance":
        # silicon tbody
        compare_con["Latency(ms)"] = {}
        baseline_latency = model_baseline_benchmark.get("Summary", {}).get(
            "Latency(ms)", {}
        )
        result_latency = model_result_benchmark.get("Summary", {}).get(
            "Latency(ms)", {}
        )
        silicon_baseline = baseline_latency.get("silicon_time", "")
        model_tr += f"<td>{silicon_baseline}</td>\n"
        silicon_result = result_latency.get("silicon_time", "")
        compare_con["Latency(ms)"]["silicon_baseline"] = silicon_baseline
        compare_con["Latency(ms)"]["silicon_result"] = silicon_result
        compare_con["Latency(ms)"]["status"] = "PASS"
        if silicon_result and silicon_baseline:
            latency_ratio = (silicon_baseline - silicon_result) / silicon_baseline
            latency_ratio = round(latency_ratio * 100, 2)
            if latency_ratio < -3:
                model_tr += '<td>%s(<b><font color="red">%s%%</font></b>)</td>\n' % (
                    silicon_result,
                    latency_ratio,
                )
                compare_con["Latency(ms)"]["status"] = "FAIL"
            elif latency_ratio > 3:
                model_tr += '<td>%s(<b><font color="green">%s%%</font></b>)</td>\n' % (
                    silicon_result,
                    latency_ratio,
                )
            else:
                model_tr += "<td>%s(%s%%)</td>\n" % (silicon_result, latency_ratio)
        else:
            model_tr += f"<td>{silicon_result}</td>\n"
        # vart tbody
        compare_con["vart"] = {}
        vart_baseline = baseline_latency.get("vart", "")
        model_tr += f"<td>{vart_baseline}</td>\n"
        vart_result = result_latency.get("vart", "")
        compare_con["vart"]["vart_baseline"] = vart_baseline
        compare_con["vart"]["vart_result"] = vart_result
        compare_con["vart"]["status"] = "PASS"
        if vart_result and vart_baseline:
            vart_ratio = (vart_baseline - vart_result) / vart_baseline
            vart_ratio = round(vart_ratio * 100, 2)
            if vart_ratio < -3:
                model_tr += '<td>%s(<b><font color="red">%s%%</font></b>)</td>\n' % (
                    vart_result,
                    vart_ratio,
                )
                compare_con["vart"]["status"] = "FAIL"
            elif vart_ratio > 3:
                model_tr += '<td>%s(<b><font color="green">%s%%</font></b>)</td>\n' % (
                    vart_result,
                    vart_ratio,
                )
            else:
                model_tr += "<td>%s(%s%%)</td>\n" % (vart_result, vart_ratio)
        else:
            model_tr += f"<td>{vart_result}</td>\n"

        # througnput tbody
        compare_con["Throughput(fps)"] = {}
        fps_baseline = (
            model_baseline_benchmark.get("Summary", {})
            .get("Throughput(fps)", {})
            .get("onnxep", "")
        )
        model_tr += f"<td>{fps_baseline}</td>\n"
        fps_result = (
            model_result_benchmark.get("Summary", {})
            .get("Throughput(fps)", {})
            .get("onnxep", "")
        )
        compare_con["Throughput(fps)"]["fps_baseline"] = fps_baseline
        compare_con["Throughput(fps)"]["fps_result"] = fps_result
        compare_con["Throughput(fps)"]["status"] = "PASS"
        if fps_result and fps_baseline:
            fps_ratio = (fps_result - fps_baseline) / fps_baseline
            fps_ratio = round(fps_ratio * 100, 2)
            if fps_ratio < -3:
                model_tr += '<td>%s(<b><font color="red">%s%%</font></b>)</td>\n' % (
                    fps_result,
                    fps_ratio,
                )
                compare_con["Throughput(fps)"]["status"] = "FAIL"
            elif fps_ratio > 3:
                model_tr += '<td>%s(<b><font color="green">%s%%</font></b>)</td>\n' % (
                    fps_result,
                    fps_ratio,
                )
            else:
                model_tr += "<td>%s(%s%%)</td>\n" % (fps_result, fps_ratio)
        else:
            model_tr += f"<td>{fps_result}</td>\n"

    if test_mode in ("functionality", "accuracy"):
        compare_con["psnr"] = {}
        baseline_accuracy = model_baseline_benchmark.get("Summary", {}).get(
            "Accuracy", {}
        )
        current_accuracy = model_result_benchmark.get("Summary", {}).get("Accuracy", {})
        baseline_psnr = baseline_accuracy.get("snr", "")
        baseline_psnr = (
            baseline_accuracy.get("psnr", "") if not baseline_psnr else baseline_psnr
        )
        model_tr += f'<td align="center">{baseline_psnr}</td>\n'
        current_psnr = current_accuracy.get("snr", "")
        current_psnr = (
            current_accuracy.get("psnr", "") if not current_psnr else current_psnr
        )
        compare_con["psnr"]["baseline_psnr"] = baseline_psnr
        compare_con["psnr"]["current_psnr"] = current_psnr
        compare_con["psnr"]["status"] = "PASS"
        if current_psnr == baseline_psnr:
            model_tr += f'<td align="center">{current_psnr}</td>\n'
        else:
            model_tr += f'<td align="center"><b><font color="red">{current_psnr}</font></b></td>\n'
            compare_con["psnr"]["status"] = "FAIL"

        compare_con["l2norm"] = {}
        baseline_l2norm = baseline_accuracy.get("lpips", "")
        baseline_l2norm = (
            baseline_accuracy.get("l2norm", "")
            if not baseline_l2norm
            else baseline_l2norm
        )
        model_tr += f'<td align="center">{baseline_l2norm}</td>\n'
        current_l2norm = current_accuracy.get("lpips", "")
        current_l2norm = (
            current_accuracy.get("l2norm", "") if not current_l2norm else current_l2norm
        )
        compare_con["l2norm"]["baseline_l2norm"] = baseline_l2norm
        compare_con["l2norm"]["current_l2norm"] = current_l2norm
        compare_con["l2norm"]["status"] = "PASS"
        if current_l2norm == baseline_l2norm:
            model_tr += f'<td align="center">{current_l2norm}</td>\n'
        else:
            model_tr += f'<td align="center"><b><font color="red">{current_l2norm}</font></b></td>\n'
            compare_con["l2norm"]["status"] = "FAIL"

    # subgraph tbody
    compare_con["DPU_Subgraph_on_ipu"] = {}
    baseline_subgraph_on_ipu = (
        model_baseline_benchmark.get("Summary", {})
        .get("DPU_Subgraph", {})
        .get("On_IPU", "")
    )
    result_subgraph_on_ipu = (
        model_result_benchmark.get("Summary", {})
        .get("DPU_Subgraph", {})
        .get("On_IPU", "")
    )
    compare_con["DPU_Subgraph_on_ipu"][
        "baseline_subgraph_on_ipu"
    ] = baseline_subgraph_on_ipu
    compare_con["DPU_Subgraph_on_ipu"][
        "result_subgraph_on_ipu"
    ] = result_subgraph_on_ipu
    compare_con["DPU_Subgraph_on_ipu"]["status"] = "PASS"
    if (
        result_subgraph_on_ipu in (0, "", "NA")
        or baseline_subgraph_on_ipu in (0, "", "NA")
        or result_subgraph_on_ipu > baseline_subgraph_on_ipu
    ):
        model_tr += (
            '<td align="center">%s(<b><font color="red"> Baseline:%s</font></b>)</td>\n'
            % (
                result_subgraph_on_ipu,
                baseline_subgraph_on_ipu,
            )
        )
        compare_con["DPU_Subgraph_on_ipu"]["status"] = "FAIL"
    else:
        model_tr += f'<td align="center">{result_subgraph_on_ipu}</td>\n'

    compare_con["DPU_Subgraph_xcompiler"] = {}
    baseline_subgraph_xcompiler = (
        model_baseline_benchmark.get("Summary", {})
        .get("DPU_Subgraph", {})
        .get("xcompiler", "")
    )
    result_subgraph_xcompiler = (
        model_result_benchmark.get("Summary", {})
        .get("DPU_Subgraph", {})
        .get("xcompiler", "")
    )
    compare_con["DPU_Subgraph_xcompiler"][
        "baseline_subgraph_xcompiler"
    ] = baseline_subgraph_xcompiler
    compare_con["DPU_Subgraph_xcompiler"][
        "result_subgraph_xcompiler"
    ] = result_subgraph_xcompiler
    compare_con["DPU_Subgraph_xcompiler"]["status"] = "PASS"
    if (
        result_subgraph_xcompiler in (0, "", "NA")
        or baseline_subgraph_xcompiler in (0, "", "NA")
        or result_subgraph_xcompiler > baseline_subgraph_xcompiler
    ):
        model_tr += (
            '<td align="center">%s(<b><font color="red"> Baseline:%s</font></b>)</td>\n'
            % (
                result_subgraph_xcompiler,
                baseline_subgraph_xcompiler,
            )
        )
        compare_con["DPU_Subgraph_xcompiler"]["status"] = "FAIL"
    else:
        model_tr += f'<td align="center">{result_subgraph_xcompiler}</td>\n'

    compare_con["pdi_on_ipu"] = {}
    baseline_pdi_on_ipu = (
        model_baseline_benchmark.get("Summary", {})
        .get("PDI_Number", {})
        .get("On_IPU", "")
    )
    result_pdi_on_ipu = (
        model_result_benchmark.get("Summary", {})
        .get("PDI_Number", {})
        .get("On_IPU", "")
    )
    compare_con["pdi_on_ipu"]["baseline_pdi_on_ipu"] = baseline_pdi_on_ipu
    compare_con["pdi_on_ipu"]["result_pdi_on_ipu"] = result_pdi_on_ipu
    compare_con["pdi_on_ipu"]["status"] = "PASS"
    if (
        result_pdi_on_ipu in ("", "NA")
        or baseline_pdi_on_ipu in ("", "NA")
        or result_pdi_on_ipu > baseline_pdi_on_ipu
    ):
        model_tr += (
            '<td align="center">%s(<b><font color="red"> Baseline:%s</font></b>)</td>\n'
            % (
                result_pdi_on_ipu,
                baseline_pdi_on_ipu,
            )
        )
        compare_con["pdi_on_ipu"]["status"] = "FAIL"
    else:
        model_tr += f'<td align="center">{result_pdi_on_ipu}</td>\n'

    compare_con["pdi_on_xcompiler"] = {}
    baseline_pdi_xcompiler = (
        model_baseline_benchmark.get("Summary", {})
        .get("PDI_Number", {})
        .get("xcompiler", "")
    )
    result_pdi_xcompiler = (
        model_result_benchmark.get("Summary", {})
        .get("PDI_Number", {})
        .get("xcompiler", "")
    )
    compare_con["pdi_on_xcompiler"]["baseline_pdi_xcompiler"] = baseline_pdi_xcompiler
    compare_con["pdi_on_xcompiler"]["result_pdi_xcompiler"] = result_pdi_xcompiler
    compare_con["pdi_on_xcompiler"]["status"] = "PASS"
    if (
        result_pdi_xcompiler in ("", "NA")
        or baseline_pdi_xcompiler in ("", "NA")
        or result_pdi_xcompiler > baseline_pdi_xcompiler
    ):
        model_tr += (
            '<td align="center">%s(<b><font color="red"> Baseline:%s</font></b>)</td>\n'
            % (
                result_pdi_xcompiler,
                baseline_pdi_xcompiler,
            )
        )
        compare_con["pdi_on_xcompiler"]["status"] = "FAIL"
    else:
        model_tr += f'<td align="center">{result_pdi_xcompiler}</td>\n'

    compare_con["CPU_Subgraph"] = {}
    baseline_cpu_subgraph = model_baseline_benchmark.get("Summary", {}).get(
        "CPU_Subgraph", ""
    )
    result_cpu_subgraph = model_result_benchmark.get("Summary", {}).get(
        "CPU_Subgraph", ""
    )
    compare_con["CPU_Subgraph"]["baseline_cpu_subgraph"] = baseline_cpu_subgraph
    compare_con["CPU_Subgraph"]["result_cpu_subgraph"] = result_cpu_subgraph
    compare_con["CPU_Subgraph"]["status"] = "PASS"
    if (
        result_cpu_subgraph in (0, "", "NA")
        or baseline_cpu_subgraph in (0, "", "NA")
        or result_cpu_subgraph > baseline_cpu_subgraph
    ):
        model_tr += (
            '<td align="center">%s(<b><font color="red"> Baseline:%s</font></b>)</td>\n'
            % (
                result_cpu_subgraph,
                baseline_cpu_subgraph,
            )
        )
        compare_con["CPU_Subgraph"]["status"] = "FAIL"
    else:
        model_tr += f'<td align="center">{result_cpu_subgraph}</td>\n'

    compare_con["OPS"] = {}
    baseline_ops = model_baseline_benchmark.get("Summary", {}).get("OPS", "")
    result_ops = model_result_benchmark.get("Summary", {}).get("OPS", "")
    compare_con["OPS"]["baseline_ops"] = baseline_ops
    compare_con["OPS"]["result_ops"] = result_ops
    compare_con["OPS"]["status"] = "PASS"
    if (
        result_ops in (0, "", "NA")
        or result_ops is None
        or baseline_ops in (0, "", "NA")
        or result_ops < baseline_ops
    ):
        model_tr += (
            '<td align="center">%s%%(<b><font color="red"> Baseline:%s%%</font></b>)</td>\n'
            % (result_ops, baseline_ops)
        )
        compare_con["OPS"]["status"] = "FAIL"
    else:
        model_tr += f'<td align="center">{result_ops}%</td>\n'

    if test_mode == "performance":
        baseline_onnx_md5 = model_baseline_benchmark.get("Summary", {}).get(
            "onnx_model_md5", ""
        )
        result_onnx_md5 = model_result_benchmark.get("Summary", {}).get(
            "onnx_model_md5", ""
        )
        if result_onnx_md5 == baseline_onnx_md5:
            model_tr += f'<td align="center">{result_onnx_md5}</td>\n'
        else:
            model_tr += (
                '<td align="center">%s(<b><font color="red"> Baseline:%s</font></b>)</td>\n'
                % (
                    result_onnx_md5,
                    baseline_onnx_md5,
                )
            )

    model_tr += "</tr>"
    compare_item[model_name] = compare_con
    return model_tr


def compare_with_baseline(
    baseline_json, result_json, test_mode, target_type="", dpu_type=""
):
    if not (os.path.exists(baseline_json) and os.path.exists(result_json)):
        return
    try:
        with open(baseline_json) as b, open(result_json) as r:
            baseline_benchmek = json.load(b)
            result_benchmark = json.load(r)
    except Exception as e:
        print("ERROR: %s " % e)
        return

    table_heads = make_thead(test_mode, target_type, dpu_type)

    table_content = ""

    compare_data = {"COMPARE_DATA": {}}
    for model_name, model_result_benchmark in result_benchmark.items():
        if not isinstance(model_result_benchmark, dict):
            continue
        if not "Summary" in model_result_benchmark.keys():
            continue
        model_baseline_benchmark = baseline_benchmek.get(model_name, {})
        compare_item = {}
        table_content += make_model_tr(
            model_name,
            test_mode,
            model_result_benchmark,
            model_baseline_benchmark,
            compare_item,
        )

        compare_data["COMPARE_DATA"].update(compare_item)

    # compare_summary = make_summary(result_benchmark)
    baseline_json_original_url = os.environ.get("BASELINE_BENCHMARK_DATA", "")
    if baseline_json_original_url:
        json_detail = "<b>Baseline: </b>" + '<a href="%s">%s</a><br>\n' % (
            baseline_json_original_url,
            baseline_json_original_url,
        )
    else:
        json_detail = "<b>Baseline: </b><font> %s</font><br>\n" % baseline_json
    result_benchmark_artifact_json = (
        os.environ.get(
            "BUILD_URL",
            "http://xsjvitisaijenkins:8080/view/ONNX/job/daily_regression/job/daily_performance_4x4",
        )
        + "artifact/"
        + result_json
    )

    json_detail += "<b>Current: </b>" + '<a href="%s">%s</a><br>\n' % (
        result_benchmark_artifact_json,
        result_benchmark_artifact_json,
    )
    paras = {
        "json_to_compare": json_detail,
        "thead_tr1": table_heads[0],
        "thead_tr2": table_heads[1],
        "thead_tr3": table_heads[2],
        "title": "performance report ",
        "tbody_content": table_content,
    }

    return constant.BASELINE_COMPARE_TABLE.format(**paras), compare_data


if __name__ == "__main__":
    baseline_json = sys.argv[1]
    result_json = sys.argv[2]
    compare_with_baseline(baseline_json, result_json)
