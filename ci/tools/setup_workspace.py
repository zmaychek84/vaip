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
import traceback
import json
import subprocess
import urllib.request
from zipfile import ZipFile
from pathlib import Path
from glob import glob


arti_url = "ARTIFACT_URL"
sources = {
    "voe": "VOE_NAME",
    "xclbin": "XCLBIN_NAME",
    "ctl_pkts": "CTL_PKTS",
    "build_log": "BUILD_LOG",
}


def update_json_dict(json_dict, env_name, kw, val):
    try:
        for each in json_dict["passes"]:
            if "passDpuParam" in each:
                if "xcompilerAttrs" not in each["passDpuParam"]:
                    continue
                each["passDpuParam"]["xcompilerAttrs"][env_name.lower()] = {kw: val}
    except Exception as e:
        print(e)


def download():
    for key, each in sources.items():
        if each == "":
            print(f"Ignore download: {key}")
            continue
        tool = os.path.basename(each)
        if not os.path.isfile(tool):
            print(f"Download: {arti_url}{each}")
            urllib.request.urlretrieve(arti_url + each, each)
        if tool.endswith("zip") and not os.path.exists(key):
            print(f"Extract: {each}")
            with ZipFile(tool, "r") as zObject:
                zObject.extractall(key)


def convert_bat(pwd):
    postfix = "*.bat"
    all_bats = Path("build_log").rglob(postfix)
    for each in all_bats:
        print(f"Convert:{each}")
        f = open(each, "r")
        lines = f.readlines()
        f.close()
        old_firm_dir = ""
        for i in range(0, len(lines)):
            new = lines[i]
            if new.find("set XLNX_VART_FIRMWARE=") != -1:
                tmp_list = new.split("=")
                old_firm_dir = os.path.dirname(tmp_list[1])
                lines[i] = (
                    tmp_list[0] + "=" + os.path.join(pwd, os.path.basename(tmp_list[1]))
                )
                # print(lines[i])
            elif new.find("set PATH=") != -1:
                old_dir = os.path.join(old_firm_dir, "hi")
                new_dir = os.path.join(pwd, "voe")
                lines[i] = new.replace(old_dir, new_dir)
        f = open(str(each), "w")
        lines = f.writelines(lines)
        f.close()


def convert_json(pwd):
    pdi_elf = os.path.join(pwd, "ctl_pkts")
    if not os.path.exists(pdi_elf):
        return
    postfix = "vaip_config.json"
    all_config = Path("build_log").rglob(postfix)
    for each in all_config:
        print(f"Convert: {each}")
        json_file = open(each, "r")
        json_dict = json.load(json_file)
        json_file.close()
        update_json_dict(
            json_dict,
            "pdi_elf_path",
            "stringValue",
            pdi_elf,
        )
        json_file = open(str(each), "w")
        json.dump(json_dict, json_file, indent=4)
        json_file.close()


def install_vaitrace(pwd):
    try:
        cwd = Path(pwd) / "voe"
        os.chdir(cwd)
        python_dir = str(cwd / "python" / "python.exe")
        cmd = [
            python_dir,
            "-m",
            "pip",
            "install",
            "vaitrace",
            "--find-links",
            ".",
        ]
        print(cmd, flush=True)
        subprocess.check_call(cmd)
    except Exception as e:
        print(e, flush=True)
    finally:
        os.chdir(pwd)


def main():
    try:
        tmp_dir = os.path.dirname(arti_url[:-1])
        work_dir = (
            os.path.basename(os.path.dirname(tmp_dir)) + "_" + os.path.basename(tmp_dir)
        )
        if not os.path.isdir(work_dir):
            os.makedirs(work_dir)

        os.chdir(work_dir)
        pwd = os.getcwd()
        print(f"Setup workspace: {pwd}")

        download()

        install_vaitrace(pwd)

        convert_bat(pwd)

        convert_json(pwd)

        print(f"Success setup workspace: {pwd}")

    except Exception as e:
        tb = traceback.format_exc()
        if not tb is None:
            print(tb)
        print(e)


if __name__ == "__main__":
    main()
