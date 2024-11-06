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
import json
import pandas as pd
import logging
from pathlib import Path
import subprocess

csv_data = {}
CWD = Path.cwd()


def create_dict(key, ops, data):
    csv_data.update({key: {}})
    for op_name in ops:
        if op_name in data.keys():
            try:
                avg_runtime = data[op_name]["Result"]["total_runtime"][
                    "profiling_data"
                ]["aie"]["total_super_layer_runtime"]["avg_t"]
            except KeyError:
                avg_runtime = ""
        else:
            avg_runtime = ""
        csv_data[key].update({op_name: avg_runtime})


def get_silicon(data, txt_dir):
    build_id = os.environ.get("BUILD_ID", "999")
    try:
        test_case = os.environ.get("TEST_CASE", "")
        print("test_case ---->", test_case)
        silicon_time_file = f"{test_case}_silicon_time_{build_id}.csv"
        op_names = []
        txt_file = os.path.join(txt_dir, test_case + ".txt")
        print("txt_file ----->", txt_file)
        if os.path.exists(txt_file):
            with open(txt_file, "r") as file_name:
                op_names = [line.strip() for line in file_name]
            create_dict(test_case, op_names, data)
    except Exception as e:
        logging.warning(f"!!! warning : get silicon failed! {e}.)")
    print("csv_data -------->", csv_data)
    dfs = []
    for key, value in csv_data.items():
        df = pd.DataFrame.from_dict(
            csv_data[key], orient="index", columns=["Silicon Time"]
        )
        df.insert(0, "Name", df.index)
        df.to_csv(silicon_time_file, index=False)


if __name__ == "__main__":
    pass
