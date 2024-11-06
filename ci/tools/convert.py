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
import logging
import re
import platform

ignore_list = [
    "DEBUG_GRAPH_RUNNER",
    "DEBUG_RUNNER",
    "DUMP_RESULT",
    "XLNX_ENABLE_CACHE",
    "XLNX_CACHE_DIR",
    "ENABLE_FAST_PM",
    "XLNX_ENABLE_DUMP",
    "DUMP_SUBGRAPH_OPS",
]


def write_bat(bat_file, bat_list):
    try:
        with open(bat_file, "w") as f:
            if "onnx_perf" in bat_file:
                f.write(f"set XLNX_ENABLE_CACHE=0\n")
            for item in bat_list:
                if "=" in item:
                    env = item.split("=")
                    if env[0] not in ignore_list:
                        f.write(f"set {item}")
                else:
                    f.write(f"{item}\n")
        f.close()
    except Exception as e:
        print(e, flush=True)


def main(build_log):
    try:
        bat_list = []
        firmware = ""
        cmd = ""
        trace = ""
        vart_perf_flag = False
        with open(build_log, "r") as f:
            lines = f.readlines()
            lines = [re.sub("#.*", "", line) for line in lines]
            lines = [line for line in lines if len(line) > 0]
            pattern_path = r"^PATH=(.*)"
            pattern_env = r"^(ENABLE|VAIP|XLNX|DUMP_|DEBUG_|NUM_|ONNX|VAI_AIE|VITISAI_EP|USE_CPU_)\S+=\S+.*"
            pattern_onnx_perf = r"running.*]\s+:\s+(\S+onnxruntime_perf_test.exe')(.*vitisai.*)(config.*vaip_config.json)(.*)"
            pattern_trace_perf = r"running.*]\s+:\s*(.*vaitrace.exe)(.*)\s+(\S+onnxruntime_perf_test.exe')(.*vitisai.*)(config.*vaip_config.json)(.*)"
            pattern_vart_perf = r"running.*]\s+:\s+(\S+vart_perf.exe')(.*)"
            pattern_trace_vart = (
                r"running.*]\s+:\s*(.*vaitrace.exe)(.*)\s+(\S+vart_perf.exe')(.*)"
            )
            pattern_mismatch = r"running.*]\s+:\s+(\S+classification.exe')(.*)"
            pattern_cpu_ep = r"running.*]\s+:\s+(\S+classification.exe')(.*-n)(.*)"
            pattern_onnx_runner_ctx = (
                r"running.*]\s+:\s+(\S+test_onnx_runner.exe')(.*.onnx)"
            )
            pattern_onnx_runner = (
                r"running.*]\s+:\s+(\S+test_onnx_runner.exe')(.*.onnx)(.*.bin|.*.raw)"
            )
            path = ""
            set_config = f"config_file|.\\vaip_config.json "

            mismatch_flag = False
            for line in lines:
                m = re.search(pattern_path, line)
                if m:
                    path_list = m.group(1).strip().split(";")[0:3]
                    path = (";").join(path_list)
                if re.search(pattern_env, line):
                    if line.find("XLNX_VART_FIRMWARE") != -1:
                        firmware = line
                    bat_list.append(line)
                m = re.search(pattern_onnx_runner_ctx, line)
                n = re.search(pattern_onnx_runner, line)
                if m and not n:
                    bat_list.append("REM generate_ep_context")
                    bat_list.append("test_onnx_runner.exe" + m.group(2))

                m = re.search(pattern_mismatch, line)
                n = re.search(pattern_cpu_ep, line)
                if m and not n:
                    mismatch_flag = True
                    bat_list.append("REM test_dpu_ep/cpu_runner")
                    dpu_ep_cmd = "classification.exe" + m.group(2).replace("'", "")
                    bat_list.append(dpu_ep_cmd)
                if n:
                    bat_list.append("REM test_cpu_ep")
                    cpu_ep_cmd = (
                        "classification.exe" + n.group(2) + n.group(3).replace("'", "")
                    )
                    bat_list.append(cpu_ep_cmd)
                    continue
                m = re.search(pattern_trace_perf, line)
                if m:
                    trace = "vaitrace.exe " + m.group(2).strip().replace("'", "")
                    cmd = (
                        " onnxruntime_perf_test.exe "
                        + m.group(4).strip().replace("'", '"')
                        + set_config
                        + m.group(6).strip().replace("'", '"')
                    )
                    bat_list.append(trace + cmd)
                    continue
                m = re.search(pattern_onnx_perf, line)
                if m:
                    cmd = (
                        "onnxruntime_perf_test.exe "
                        + m.group(2).strip().replace("'", '"')
                        + set_config
                        + m.group(4).strip().replace("'", '"')
                    )
                    bat_list.append(cmd)
                    continue
                m = re.search(pattern_trace_vart, line)
                if m:
                    vart_perf_flag = True
                    trace = "vaitrace.exe " + m.group(2).strip().replace("'", "")
                    cmd = " vart_perf.exe" + m.group(4)
                    bat_list.append(trace + cmd)
                    continue
                m = re.search(pattern_vart_perf, line)
                if m:
                    vart_perf_flag = True
                    cmd = "vart_perf.exe" + m.group(2)
                    bat_list.append(cmd)

        set_path = f"PATH={path};%PATH%\n"
        bat_list.insert(0, set_path)
        if mismatch_flag == True:
            classification_bat = "run_classification.bat"
            write_bat(classification_bat, bat_list)
            print(f"Generate: {classification_bat} success.")
        elif trace != "":
            if vart_perf_flag == True:
                new_bat_list = [set_path]
                new_bat_list.append(firmware)
                for line in bat_list:
                    if "vart_perf.exe" in line:
                        new_bat_list.append(line)
                trace_vart_perf_bat = "run_vaitrace_vart_perf.bat"
                write_bat(trace_vart_perf_bat, new_bat_list)
                print(f"Generate: {trace_vart_perf_bat} success.")
            else:
                trace_perf_bat = "run_vaitrace_onnx_perf.bat"
                write_bat(trace_perf_bat, bat_list)
                print(f"Generate: {trace_perf_bat} success.")

                trace_vart_perf_bat = "run_vaitrace_vart_perf.bat"
                with open(trace_vart_perf_bat, "w") as f:
                    f.write(f"set PATH={path};%PATH%\n")
                    f.write(f"set {firmware}")
                    f.write(f"{trace} vart_perf.exe -x compiled.xmodel -r 500")
                print(f"Generate: {trace_vart_perf_bat} success.")
        else:
            if vart_perf_flag == True:
                new_bat_list = [set_path]
                new_bat_list.append(firmware)
                for line in bat_list:
                    if "vart_perf.exe" in line:
                        new_bat_list.append(line)
                vart_perf_bat = "run_vart_perf.bat"
                write_bat(vart_perf_bat, new_bat_list)
                print(f"Generate: {vart_perf_bat} success.")
            else:
                onnx_perf_bat = "run_onnx_perf.bat"
                write_bat(onnx_perf_bat, bat_list)
                print(f"Generate: {onnx_perf_bat} success.")

                vart_perf_bat = "run_vart_perf.bat"
                with open(vart_perf_bat, "w") as f:
                    f.write(f"set PATH={path};%PATH%\n")
                    f.write(f"set {firmware}")
                    f.write("vart_perf.exe -x compiled.xmodel -r 500")
                print(f"Generate: {vart_perf_bat} success.")

    except Exception as e:
        print(e)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 convert.py build.log")
        sys.exit()

    main(sys.argv[1])
