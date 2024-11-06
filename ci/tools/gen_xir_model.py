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
import subprocess
import glob
import os

t_found = 0
t_not_found = 0


def match_model_name(json_data, key):
    if key in json_data:
        return json_data[key]
    else:
        return None


def main(input_folder, input_file, output_file, mode):
    found = 0
    not_found = 0
    global t_found
    global t_not_found
    with open("xv3dpu_internal.json", "r", encoding="utf-8") as json_file:
        json_data = json.load(json_file)

    with open(output_file, mode, encoding="utf-8") as output:
        with open(
            os.path.join(input_folder, input_file), "r", encoding="utf-8"
        ) as file:
            folder = os.path.splitext(input_file)[0]
            f_path = os.path.join("./tmp", folder)
            folder_path = os.path.normpath(f_path)
            os.makedirs(folder_path, exist_ok=True)
            print(f"Processing {folder_path}......")
            for num, line in enumerate(file, 1):
                key = line.strip()
                meta = match_model_name(json_data, key)
                # if meta is not None and key.find('conv') != -1:
                if True:
                    xmodel = meta["meta"].get("xmodel")
                    found += 1
                    # print(f"\t{num:3} {xmodel}")

                    com = "xcompiler -i {} -o {} --profile 1 --target DPUCV3DX8G_ISA0_C8SP1"
                    file_name = os.path.basename(xmodel)
                    file_parts = file_name.split("/")
                    xmodel_name = file_parts[-1]
                    output_model = os.path.join(folder_path, key + ".xmodel")
                    command = com.format(xmodel, output_model)
                    print(command)

                    result = subprocess.run(
                        command,
                        shell=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                    if result.returncode == 0:
                        # print(result.stdout.decode('utf-8'))
                        print(result.stdout)
                        output.write(f"[{num}] '{key}' : {xmodel} SUCCESS\n")
                    else:
                        print(result.stderr)
                        output.write(f"[{num}] '{key}' : {xmodel} FAIL\n")
                        # print(result.stderr.decode('utf-8'))
                    output.flush()
                else:
                    not_found += 1
                    print(f"\t'{key}'")
                    output.write(f"'{key}' not found\n")

        print(f"founded: {found}")
        print(f"not founded: {not_found}")
        t_found += found
        t_not_found += not_found


if __name__ == "__main__":
    input_folder = "./"
    f_type = "*.txt"
    output_file = "gen_xmodel_output.log"
    os.system(f"rm -rf {output_file}")
    mode = "a+"

    txt_files = glob.glob(os.path.join(input_folder, f_type))
    for input_file in txt_files:
        main(input_folder, input_file, output_file, mode)

    print("-" * 30 + " summary " + "-" * 30)
    print(f"tatal founded: {t_found}")
    print(f"total not founded: {t_not_found}")
