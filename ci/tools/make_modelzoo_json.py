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
import os
import sys
import json
import hashlib
from pathlib import Path
from glob import glob


def make_modelzoo_json(host, modelzoo_path, json_file):
    print(modelzoo_path)
    res = []
    postfix = "*.onnx"
    if (
        os.environ.get("MODEL_TYPE", "onnx") == "xmodel"
        or os.environ.get("TEST_MODE", "performance") == "vart_perf"
    ):
        postfix = "*.xmodel"
        if modelzoo_path.endswith("cache_zip"):
            postfix = "compiled*.xmodel"
        print("Searching for xmodel...", flush=True)
    else:
        print("Searching for onnx...", flush=True)
    all_models = Path(modelzoo_path).rglob(postfix)
    for f in all_models:
        onnx_model = str(f)
        print(f, flush=True)
        dirname = os.path.dirname(onnx_model)
        idname = os.path.basename(dirname)
        sub_models = Path(dirname).rglob(postfix)
        if dirname == modelzoo_path or len(list(sub_models)) > 1:
            idname = os.path.basename(f).replace(postfix[1:], "")
        md5 = ""
        with open(onnx_model, "rb") as fb:
            md5 = hashlib.md5(fb.read()).hexdigest()
        model_info = {
            "id": idname,
            "onnx_model": onnx_model,
            "hostname": host,
            "md5sum": md5,
        }
        print(model_info, flush=True)
        res.append(model_info)
        res = sorted(res, key=lambda x: x["id"])

        with open(json_file, "w") as file:
            file.write(json.dumps(res, indent=4))


if __name__ == "__main__":
    # make modelzoo json
    if len(sys.argv) < 4:
        print(
            "Usage: python3 make_modelzoo_json.py [host] [model_path] [model_zoo_json_file]."
        )
        sys.exit()

    make_modelzoo_json(sys.argv[1], sys.argv[2], sys.argv[3])
