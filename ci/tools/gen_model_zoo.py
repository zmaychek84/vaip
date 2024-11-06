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
import json
import onnx
import os
import argparse
from pathlib import Path
import warnings
import hashlib


def calculate_md5(file_path):
    md5 = hashlib.md5()
    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(4096), b""):
            md5.update(chunk)
    return md5.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser("Generator json file for model zoo")
    parser.add_argument("models_dir", type=str, help="models directory")
    parser.add_argument("output_json_file", type=str, help="output model zoo json file")
    parser.add_argument("--golden_dir", type=str, help="golden directory")
    parser.add_argument(
        "--host_name",
        type=str,
        default="xcdl190074.xilinx.com",
        help="models and golden hostname",
    )
    args, _ = parser.parse_known_args()
    return args


class ModelZooJsonGenerator(object):
    def __init__(self, args):
        self.args = args
        self.models_dir = os.path.abspath(args.models_dir)
        if args.golden_dir:
            self.golden_dir = os.path.abspath(args.golden_dir)
        self.models = [
            file for file in os.listdir(args.models_dir) if file.endswith(".onnx")
        ]
        self.output_json_file = args.output_json_file
        self.model_zoo = []

    def dump_model_zoo(self):
        with open(self.output_json_file, "w") as file:
            json.dump(self.model_zoo, file, indent=4)

    def gen(self):
        for file in self.models:
            id, _ = os.path.splitext(file)
            onnx_model = self.models_dir + "/" + file

            model_json = {
                "id": id,
                "hostname": self.args.host_name,
                "onnx_model": onnx_model,
                "md5sum": calculate_md5(onnx_model),
            }

            if self.args.golden_dir:
                model = onnx.load(onnx_model)
                model_inputs = []
                for i in model.graph.input:
                    i_name = i.name.replace("/", "_").replace(":", "_")
                    input_file_name = self.golden_dir + "/" + i_name + ".bin"
                    if os.path.exists(input_file_name):
                        model_inputs.append(input_file_name)
                    else:
                        warnings.warn(f"can not find input file {input_file_name}")
                model_json["input"] = model_inputs

            self.model_zoo.append(model_json)


if __name__ == "__main__":
    args = parse_args()
    g = ModelZooJsonGenerator(args)
    g.gen()
    g.dump_model_zoo()
    print("Done")
