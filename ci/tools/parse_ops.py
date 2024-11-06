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
import argparse
import sys
import os
from pathlib import Path


def parse_ops(model_name, log_path):
    operators_parser_path = (
        Path(os.environ.get("WORKSPACE", ""))
        / "tracer_analyze"
        / "onnx-ops-coverage"
        / "operators_parser"
    )
    if not os.path.exists(operators_parser_path):
        print("ERROR:not found operators_parser script")
        return

    try:
        sys.path.append(str(operators_parser_path).replace("/", "\\"))
        from onnx_ops_parse import OnnxModelOpsAnalyzer

        print("parse OPS of model %s" % model_name)
        model_log_path = os.path.join(log_path, model_name)
        try:
            debugfiles = os.listdir(model_log_path)
            print("Files in the directory:", debugfiles)
        except FileNotFoundError:
            print(f"The directory {model_log_path} does not exist.")
        except NotADirectoryError:
            print(f"{model_log_path} is not a directory.")
        except PermissionError:
            print(f"Permission denied for accessing {model_log_path}.")

        onnx_ops = OnnxModelOpsAnalyzer(False)
        onnx_ops.parsing(model_log_path)
        print("onnx_ops---->", onnx_ops, flush=True)
        onnx_model_name = set(
            [x.get("Model Name", "") for x in onnx_ops.on_cpu_ipu_for_each]
        )
        print("onnx_model_name---->", onnx_model_name, flush=True)
        if len(onnx_model_name) != 1:
            print("ERROR: not only one model found")
            return

        op_num_ipu = 0
        for onnx_op in onnx_ops.on_ipu_for_each:
            op_num_ipu += onnx_op.get("Occurrences", 0)
        op_num_cpu = 0
        for onnx_op in onnx_ops.on_cpu_for_each:
            op_num_cpu += onnx_op.get("Occurrences", 0)

        if op_num_ipu or op_num_cpu:
            OPS = (op_num_ipu / (op_num_ipu + op_num_cpu)) * 100
            if OPS == 100.0:
                return 100
            OPS = round(OPS, 2)
            # print("%s%%" % OPS)
            return OPS
    except Exception as e:
        print("ERROR: parsing OPS failed: %s" % e)


if __name__ == "__main__":
    pass
