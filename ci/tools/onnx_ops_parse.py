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

#!/usr/bin/env python3

import os
import logging
import time
import sys
import re
import json
import subprocess
import pandas as pd
from collections import defaultdict

TYPE_PATTERN = re.compile(r"type = (\w+)")


def time_logged(func):
    func_name = func.__name__

    def wrap(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        elapse = end - start
        print(f"{func_name} took {elapse:.4f} seconds")
        return result

    return wrap


@time_logged
def save_excel_sheet(writer, df, sheet_name, target=None, models=None):
    if target is not None:
        df["Target"] = target

    if models is not None:
        df["Total Relevant Models"] = df["ONNX operator"].map(models).apply(len)
        df["Relevant Models"] = (
            df["ONNX operator"].map(models).apply(lambda x: ",  ".join(x))
        )

    df.to_excel(writer, sheet_name=sheet_name, index=False)
    worksheet = writer.sheets[sheet_name]
    column_widths = [
        max(df[column].astype(str).apply(len).max(), len(column))
        for column in df.columns
    ]
    for i, width in enumerate(column_widths):
        worksheet.set_column(i, i, width)


class XirModelOpsAnalyzer:
    def __init__(self):
        self.unsupported_ops = []
        self.supported_ops = []
        self.s_summary_data = []
        self.summary_data = []

    def extract_info(self, lines, formatted_d):
        ops = []
        for line in lines:
            model_name = formatted_d.strip()
            info = line.strip()
            optype = ""
            type_match = TYPE_PATTERN.search(info)
            if type_match:
                optype = type_match.group(1)
            ops.append({"Model Name": model_name, "OpType": optype, "Info": info})
        return ops

    @time_logged
    def run(self):
        for subdir in os.listdir("."):
            if os.path.isdir(subdir) and not os.path.exists(
                os.path.join(subdir, "*.html")
            ):
                formatted_d = f"{subdir}, "
                os.chdir(subdir)
                grep_command = (
                    r"grep 'is not supported by current target' -nr . --exclude='*.txt' --exclude='*.sh' --exclude='*.py' | "
                    r"awk -F ']' '{print $2}'"
                )
                extracted = subprocess.run(
                    grep_command, shell=True, capture_output=True, text=True
                ).stdout.strip()
                lines = extracted.split("\n")
                self.unsupported_ops.extend(self.extract_info(lines, formatted_d))
                os.chdir("..")

        for d in os.listdir():
            if os.path.isdir(d) and not d.endswith(".html"):
                formatted_d = f"{d}, "
                extracted = subprocess.check_output(
                    f'grep "has been assigned to CPU" -nr {d} --exclude="*.txt" --exclude="*.sh" --exclude="*.py" | '
                    f"awk -F \"]\" '{{print $2}}'",
                    shell=True,
                    text=True,
                ).strip()
                self.supported_ops.extend(
                    self.extract_info(extracted.split("\n"), formatted_d)
                )

        summary_command = (
            r"grep 'is not supported by current target' -nr --exclude=*.txt --exclude=*.sh --exclude=*.py . | "
            r"awk '{print $10}' | sed 's/}//' | sed '/^$/d' | sort -n | uniq -c | sort -nr"
        )
        summary_result = subprocess.check_output(summary_command, shell=True, text=True)
        for line in summary_result.strip().split("\n"):
            occurrences, op = line.split(maxsplit=1)
            self.summary_data.append(
                {"Xir OpType": op, "Occurrences": int(occurrences)}
            )

        s_summary_command = (
            r"grep 'has been assigned to CPU' -nr --exclude=*.txt --exclude=*.sh --exclude=*.py . | "
            r"awk '{print $10}' | sed 's/}//' | sed '/^$/d' | sort -n | uniq -c | sort -nr"
        )
        s_summary_result = subprocess.check_output(
            s_summary_command, shell=True, text=True
        )
        for line in s_summary_result.strip().split("\n"):
            occurrences, op = line.split(maxsplit=1)
            self.s_summary_data.append(
                {"Xir OpType": op, "Occurrences": int(occurrences)}
            )


class OnnxModelOpsAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.analyzed_models = []
        self.skipped_models = []
        self.sorted_counts = []
        self.sorted_ipu_counts = []
        self.total_op_on_cpu_counts = defaultdict(int)
        self.total_op_on_ipu_counts = defaultdict(int)
        self.op_on_cpu_models = defaultdict(set)
        self.op_on_ipu_models = defaultdict(set)
        self.all_data = []
        self.all_ipu_data = []
        self.summary_data = []

    def load_json_file(self, json_file_path):
        try:
            with open(json_file_path, "r") as file:
                return json.load(file)
        except FileNotFoundError:
            self.logger.error(f"File not found: {json_file_path}")
            return None
        except json.JSONDecodeError:
            self.logger.error(f"Error decoding JSON: {json_file_path}")
            return None

    def process_device_stat(self, data):
        all_node_num = None
        cpu_node_num = None

        for item in data.get("deviceStat", []):
            name = item.get("name")
            node_num = item.get("nodeNum")
            if name == "all":
                all_node_num = node_num
            elif name == "CPU":
                cpu_node_num = node_num

        return all_node_num, cpu_node_num

    def process_op_counts(self, entry, op_on_cpu_counts, op_on_ipu_counts, model_name):
        device = entry["device"]
        op_type = entry["opType"]
        if device == "CPU":
            op_on_cpu_counts[op_type] += 1
            self.op_on_cpu_models[op_type].add(model_name)
        elif device in ["DPU", "IPU"]:
            op_on_ipu_counts[op_type] += 1
            self.op_on_ipu_models[op_type].add(model_name)

    def accumulate_counts(self, op_counts, all_data, total_op_counts, model_name):
        for op_type, count in op_counts.items():
            all_data.append(
                {
                    "Model Name": model_name,
                    "ONNX operator": op_type,
                    "Occurrences": count,
                }
            )
            total_op_counts[op_type] += count

    def analyze_model(self, subdir, model=""):
        subdir_path = os.path.join(".", subdir)
        json_files = [
            filename for filename in os.listdir(subdir) if filename.endswith(".json")
        ]

        for json_file in json_files:
            json_file_path = os.path.join(subdir, json_file)
            data = self.load_json_file(json_file_path)
            model_name = subdir if not model else model
            all_node_num, cpu_node_num = self.process_device_stat(data)
            message = f"{model_name}: all_nodeNum = {all_node_num}, cpu_nodeNum = {cpu_node_num}"

            if (
                all_node_num is not None
                and cpu_node_num is not None
                and all_node_num == cpu_node_num
            ):
                self.skipped_models.append(model_name)
                message += ", fall back to CPU all, drop it"
                self.logger.info(message)
                break
            else:
                message += ", running"
                self.logger.info(message)

            self.analyzed_models.append(model_name)
            op_on_cpu_counts = defaultdict(int)
            op_on_ipu_counts = defaultdict(int)

            for entry in data["nodeStat"]:
                self.process_op_counts(
                    entry, op_on_cpu_counts, op_on_ipu_counts, model_name
                )

            self.accumulate_counts(
                op_on_cpu_counts, self.all_data, self.total_op_on_cpu_counts, model_name
            )
            self.accumulate_counts(
                op_on_ipu_counts,
                self.all_ipu_data,
                self.total_op_on_ipu_counts,
                model_name,
            )

    @time_logged
    def run(self):
        import pandas as pd

        subdirs = [name for name in os.listdir(".") if os.path.isdir(name)]
        for model_num, subdir in enumerate(subdirs, start=1):
            self.analyze_model(subdir)

        self.sorted_counts = sorted(
            self.total_op_on_cpu_counts.items(), key=lambda x: x[1], reverse=True
        )
        self.sorted_ipu_counts = sorted(
            self.total_op_on_ipu_counts.items(), key=lambda x: x[1], reverse=True
        )

        ipu_counts_dict = dict(self.sorted_ipu_counts)

        for cpu_item in self.sorted_counts:
            cpu_op_type, cpu_count = cpu_item

            if cpu_op_type in ipu_counts_dict:
                ipu_count = ipu_counts_dict[cpu_op_type]
                total_count = ipu_count + cpu_count
                ipu_ratio = (ipu_count / total_count) * 100 if total_count > 0 else 0

                self.summary_data.append(
                    {
                        "ONNX Op Type": cpu_op_type,
                        "IPU Count": ipu_count,
                        "CPU Count": cpu_count,
                        "IPU Ratio (%)": ipu_ratio,
                    }
                )
            else:
                self.summary_data.append(
                    {
                        "ONNX Op Type": cpu_op_type,
                        "IPU Count": 0,
                        "CPU Count": cpu_count,
                        "IPU Ratio (%)": 0.00,
                    }
                )

        num_analyzed_models = len(self.analyzed_models)
        num_skipped_models = len(self.skipped_models)
        self.logger.info(
            f"Analyzed models: {num_analyzed_models}, Skipped models: {num_skipped_models}"
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    excel_filename = "ONNX_operator_counts.xlsx"
    onnx = OnnxModelOpsAnalyzer()
    xir = XirModelOpsAnalyzer()
    onnx.run()
    xir.run()
    with pd.ExcelWriter(excel_filename) as writer:
        summary_df = pd.DataFrame(onnx.summary_data)
        df = pd.DataFrame(
            onnx.sorted_counts, columns=["ONNX operator", "Total Op Counts"]
        )
        df_ipu = pd.DataFrame(
            onnx.sorted_ipu_counts, columns=["ONNX operator", "Total Op Counts"]
        )
        all_df = pd.DataFrame(onnx.all_data)
        all_ipu_df = pd.DataFrame(onnx.all_ipu_data)

        df_summary = pd.DataFrame(xir.summary_data)
        df_s_summary = pd.DataFrame(xir.s_summary_data)
        df_unsupported = pd.DataFrame(xir.unsupported_ops)
        df_supported = pd.DataFrame(xir.supported_ops)

        save_excel_sheet(writer, summary_df, "ONNX_Ops_IPU_vs_CPU_summary")
        save_excel_sheet(
            writer, df, "ONNX_Ops_on_CPU_summary", "CPU", onnx.op_on_cpu_models
        )
        save_excel_sheet(
            writer, df_ipu, "ONNX_Ops_on_IPU_summary", "IPU", onnx.op_on_ipu_models
        )
        save_excel_sheet(writer, df_summary, "XIR_Unsupported_OPs_summary")
        save_excel_sheet(writer, df_s_summary, "XIR_Limit_OPs_summary")

        save_excel_sheet(writer, all_df, "ONNX_Ops_on_cpu_for_each", "CPU")
        save_excel_sheet(writer, all_ipu_df, "ONNX_Ops_on_ipu_for_each", "IPU")
        save_excel_sheet(writer, df_unsupported, "XIR_Unsupported_OPs_for_each")
        save_excel_sheet(writer, df_supported, "XIR_Limit_OPs_for_each")
