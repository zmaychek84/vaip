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

MONGODB_HOST = "xsjncuph07.xilinx.com"
MONGODB_PORT = "27017"
MONGO_DAILY_USER = "ntsDaily"
MONGO_DAILY_PASSWORD = "ntsDaily"
MONGO_TEST_USER = "ntsTest"
MONGO_TEST_PASSWORD = "ntsTest"


DRIVER_MAP = {
    "32.0.202.174": "https://xcoartifactory.xilinx.com/artifactory/PHX_test_case_package/driver/NPU_MCDM_STX1.0_MSFT_172_R24.06.11_RC4_174_hacked.zip",
    "32.0.202.195": "https://xcoartifactory.xilinx.com/artifactory/PHX_test_case_package/driver/NPU_MCDM_STX1.0_MSFT_192_R24.07.15_RC2_195_hacked.zip",
    "10.1.0.1": "https://xcoartifactory.xilinx.com/artifactory/PHX_test_case_package/driver/mcdm_stack_rel-run1364_hacked.zip",
    "32.0.202.206": "https://xcoartifactory.xilinx.com/artifactory/PHX_test_case_package/driver/NPU_MCDM_WIN24_R24.08.05_RC1_206_hacked.zip",
}

MODELZOO_MAP = {
    "ipu_benchmark_real_data": "MEP",
    "ipu_benchmark": "Procyon",
    "ipu_benchmark_PSO_PSA": "SHELL",
    "ipu_benchmark_pss_pst": "StableDifusion",
}

XRT_INI_VERBOSITY = """
[Runtime]
verbosity=5
"""

BIG_BUFFER_AIE_TRACE_DLL = (
    "/group/dphi_software/software/workspace/yanjunz/ipu_test/aie_trace/xrt-9048a"
)
RC_AIE_TRACE_DLL = "/group/dphi_software/software/workspace/yanjunz/ipu_test/aie_trace/xrt_mcdm_dll_1904"

XRT_INI_AIE_TRACE = """
[Runtime]
verbosity=5

[Debug]
ml_timeline=true
aie_trace=true
aie_debug=true

[AIE_trace_settings]
tile_based_interface_tile_metrics=all:input_output_ports_stalls

[AIE_debug_settings]
interface_registers = 0x000340D0, 0x000340D4, 0x000340D8, 0x000340E0, 0x000340E4, 0x0003FF00, 0x0003FF04
"""

TABLE_HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
    <style>
        table {{
            border-collapse: collapse;
            width: 50%%;
            max-width:90%%;
        }}
        table, td, th {{
            line-height: 1.8em;
            vertical-align: middle;
            text-align: center;
        }}
    </style>
</head>
<body>
<h1>IPU Regression XBJ {run_type}</h1> 

<h4>XOAH URL: http://xoah/summary?superSuiteName=IPU_TEST_PIPELINE&relBranch=1.0.0-dev</h4> 
<h4>Jenkins Pipeline URL: {jenkins_url}</h4> 
{run_date}
<br>
{elapsed}
<h3>Summary:</h3>
<table border="1">
<thead>
<tr>
  <th style="width:100px;">Total</th>
  <th style="width:100px;">Mismatch</th>
  <th style="width:100px;">CPU All</th>
  <th style="width:100px;">Result Parse Fail</th>
  <th style="width:100px;">Rounding Mode</th>
</tr>
</thead>
<tbody>
<tr>
<td>{total}</td>
<td>{mismatch_num}</td>
<td>{cpu_all_num}</td>
<td>{bad_result_num}</td>
<td>{rounding_mode_num}</td>
</tr>
</tbody>
</table>
{error_type_info}

<h3>Target Info:</h3>
  {target_info}

<h3>Details:</h3>
<table border="1">
<thead>
<tr>
  <th style="width:200px;">Model Name</th>
  <th style="width:200px;">ONNX_EP VS cpu_runner</th>
  <th style="width:200px;">Error Type</th>
  <th style="width:350px;">CPU EP</th>
  <th style="width:350px;">CPU Runner</th>
  <th style="width:350px;">ONNX VAI EP</th>
  <th style="width:350px;">Time (seconds)</th>
  <th style="width:350px;">Run Time (seconds)</th>
  <th style="width:350px;">onnx model md5</th>
</tr>
</thead>
<tbody>

  {tbody_content}

</tbody>
</table>

{op_analyze_table}

</body>
</html>
"""

PERFORMANCE_HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
    <style>
        table {{
            border-collapse: collapse;
            width: 50%%;
            max-width:90%%;
        }}
        table, td, th {{
            line-height: 1.8em;
            vertical-align: middle;
            text-align: center;
        }}
    </style>
</head>
<body>
<h1>IPU Regression XBJ Performance {run_type}</h1> 

<h4>Jenkins Pipeline URL: {jenkins_url}</h4> 
{run_date}
<br>
{elapsed}
<h3>Target Info:</h3>
  {target_info}

<b>System Version:</b>
{system_version}
<br>
<br>
{verbose_list}
<br>
{xcompiler_optimizations}

{result_summary}
<h3>Performance:</h3>
{procyon_score}
<table border="1">
<thead>
<tr>
  {thead_tr1}
</tr>
<tr>
  {thead_tr2}
</tr>
</thead>
<tbody>

  {tbody_content}
</tbody>
</table>
<br>
{compare_table}
<b><font>END</font></b>
<br>
</body>
</html>
"""

OP_TABLE = """
<h3>ONNX OPs Summary:</h3>
<table border="1">
<thead>
<tr>
  <th style="width:200px;">Op Type</th>
  <th style="width:200px;">IPU Ratio (%)</th>
  <th style="width:200px;">IPU Count</th>
  <th style="width:200px;">CPU Count</th>
  <th style="width:200px;">TOTAL</th>
</tr>
</thead>
<tbody>

  {op_tbody}

</tbody>
</table>
"""

BASELINE_COMPARE_TABLE = """
<h3>Compare With Baseline:</h3>
{json_to_compare}
<table border="1">
<thead>
<tr>
  {thead_tr1}
</tr>
<tr>
  {thead_tr2}
</tr>
<tr>
  {thead_tr3}
</tr>
</thead>
<tbody>

  {tbody_content}
</tbody>
</table>
<br>
"""
