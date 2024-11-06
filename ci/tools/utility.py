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
import argparse
import re
import hashlib
import json
import traceback
from pathlib import Path
from collections import OrderedDict
import datetime
import math
import subprocess
import shutil
import pathlib
import platform
import logging
import pip
from pathlib import Path

from . import constant

CWD = Path.cwd()


def import_or_install(package):
    try:
        __import__(package)
    except ImportError:
        pip.main(["install", package])


def run(run_path, run_args, run_env):
    cwd = os.getcwd()
    try:
        os.chdir(run_path)
        print("Run cmd: %s" % " ".join(run_args))
        output = subprocess.run(
            run_args, shell=False, capture_output=True, check=True, env=run_env
        )
        return output
    except Exception as e:
        print(e)
    finally:
        os.chdir(cwd)


def get_driver_version():
    xrt_tool = str(pathlib.Path("C:\\") / "Windows" / "System32" / "AMD" / "xbutil.exe")
    xbutil_download = (
        pathlib.Path("C:\\") / "ipu_stack_rel_silicon" / "kipudrv" / "xbutil.exe"
    )
    if os.path.exists(xbutil_download):
        xrt_tool = str(xbutil_download)
    xrt_smi = str(pathlib.Path("C:\\") / "Windows" / "System32" / "AMD" / "xrt-smi.exe")
    if os.path.exists(xrt_smi):
        xrt_tool = xrt_smi
    try:
        run_args = [xrt_tool, "--version"]
        output = run(".", run_args, os.environ)
        if not output:
            return

        stdout = str(output.stdout.decode("utf-8"))
        for line in stdout.split("\n"):
            m = re.match(r".*(IPUKMDDRV|NPU\sDRIVER).*?(\d+\.\d+\.\d+\.\d+).*", line)
            if not m:
                continue
            driver_version = m.group(2)
            print(driver_version)
            if driver_version in constant.DRIVER_MAP:
                return driver_version, constant.DRIVER_MAP[driver_version]
            else:
                return driver_version, None
    except Exception as e:
        print("Get NPU driver version failed: %s" % e)


def get_onnx_model_info(modelzoo_list, model_name):
    if not isinstance(modelzoo_list, list):
        return {}
    for onnx_model_info in modelzoo_list:
        if onnx_model_info.get("id", "") == model_name:
            return onnx_model_info
    else:
        return {}


def md5sum(file_path):
    md5 = ""
    if not os.path.exists(file_path):
        return md5
    try:
        with open(file_path, "rb") as fb:
            md5 = hashlib.md5(fb.read()).hexdigest()
    except Exception as e:
        print(e)
    finally:
        return md5


def get_version_info():
    workspace = os.environ.get("WORKSPACE", "")
    version_info = {}
    if os.path.exists(os.path.join(workspace, "hi", "version_info.txt")):
        version_info_txt = os.path.join(workspace, "hi", "version_info.txt")
    elif os.path.exists(os.path.join(workspace, "hi", "latest.txt")):
        version_info_txt = os.path.join(workspace, "hi", "latest.txt")
    else:
        print("Not Found version info txt file!")
        return version_info
    lines = read_lines(version_info_txt)
    for line in lines:
        if len(line.split(":")) == 2:
            key, value = [x.strip(" #\n") for x in line.split(":")]
            version_info[key] = value

    return version_info


def make_target_info(modelzoo_list):
    workspace = os.environ.get("WORKSPACE", "")
    vai_rt_ref = os.environ.get("VAI_RT_BRANCH", "")
    release_file = os.environ.get("RELEASE_FILE", "")
    target_info = ""
    target_info += "<b><font>VAI-RT Branch: %s</font></b>" % vai_rt_ref + "<br>\n"
    target_info += "<b><font>Release File: %s</font></b>" % release_file + "<br>\n"
    version_info = get_version_info()
    if version_info:
        target_info += "<b><font>Version info:</font></b>" + "<br>\n"
        for repo, commit in version_info.items():
            target_info += f"<font>{repo}: {commit}</font><br>\n"

    jenkins_url = os.environ.get("BUILD_URL", "")

    # voe path
    original_package = os.environ.get(
        "GLOBAL_PACKAGE",
        "",
    )
    voe_basename = os.path.basename(
        os.environ.get(
            "GLOBAL_PACKAGE", "voe-win_amd64-with_xcompiler_on-latest_dev.zip"
        )
    )
    voe_package = os.path.join(workspace, voe_basename)
    voe_package_md5 = md5sum(voe_package)
    voe_url = jenkins_url + "artifact/" + voe_basename
    target_info += (
        "<br>"
        + '<b>voe download path: </b>\n<a href="%s">%s</a> %s'
        % (voe_url, voe_url, voe_package_md5)
        + "<br>\n"
    )
    if original_package != "":
        target_info += (
            "<b>original voe path: </b>"
            + '<a href="%s">%s</a>' % (original_package, original_package)
            + "<br><br>\n"
        )

    # get xclbin md5sum
    target_info += "<b>xclbin download path:</b>\n"
    xclbin_file = os.environ.get("XLNX_VART_FIRMWARE", "").strip().replace('"', "")
    xclbin_md5 = md5sum(xclbin_file)

    xclbin_url = jenkins_url + "artifact/" + os.path.basename(xclbin_file)
    target_info += (
        '<a href="%s">%s</a>  %s' % (xclbin_url, xclbin_url, xclbin_md5) + "<br>\n"
    )

    # get xclbin info.txt
    xclbin_info_txt = os.path.join(workspace, "info.txt")
    if os.path.exists(xclbin_info_txt):
        with open(xclbin_info_txt, "r") as f:
            lines = f.readlines()
        for line in lines:
            if "DPU_PHX" in line:
                target_info += line + "<br>\n"
                break
        else:
            target_info += "\n"

    original_firmware = os.environ.get("ORIGINAL_FIRMWARE", "")
    if original_firmware:
        target_info += (
            f'<b>Original Firmware path:</b> <a href="{original_firmware}">{original_firmware}</a>'
            + "<br>\n"
        )
    else:
        target_info += "\n"

    info_name = os.environ.get("XLNX_TARGET_NAME", "") + "_tools_version.json"
    toolsVersion_json = os.path.join(workspace, info_name)
    target_type = (
        "STRIX" if not os.environ.get("TARGET_TYPE", "") else os.environ["TARGET_TYPE"]
    )
    build_id = os.environ.get("BUILD_ID", "999")
    xclbin_json = os.path.join(workspace, f"aie_partition_{build_id}.json")

    if os.path.exists(toolsVersion_json):
        with open(toolsVersion_json, "r") as j:
            data = json.load(j)
            # print(data, flush=True)
        if target_type == "PHOENIX":
            dpu_phx_for_phx_commit = data.get("DPU_PHX_for_PHX", {}).get(
                "commitID", "NONE"
            )
            dpu_phx_kernel_for_phx_commit = data.get("DPU_PHX_KERNEL_for_PHX", {}).get(
                "commitID", "NONE"
            )
            target_info += (
                f"<b>dpu_phx_for_phx commit:</b> {dpu_phx_for_phx_commit}" + "<br>\n"
            )
            target_info += (
                f"<b>dpu_phx_kernel_for_phx commit:</b> {dpu_phx_kernel_for_phx_commit}"
                + "<br><br>\n"
            )
        else:
            dpu_phx_for_stx_commit = data.get("DPU_PHX_for_STX", {}).get(
                "commitID", "NONE"
            )
            dpu_phx_kernel_for_stx_commit = data.get("DPU_PHX_KERNEL_for_STX", {}).get(
                "commitID", "NONE"
            )
            target_info += (
                f"<b>dpu_phx_for_stx commit:</b> {dpu_phx_for_stx_commit}" + "<br>\n"
            )
            target_info += (
                f"<b>dpu_phx_kernel_for_stx commit:</b> {dpu_phx_kernel_for_stx_commit}"
                + "<br>\n"
            )
    elif os.path.exists(xclbin_json):
        with open(xclbin_json, "r") as j:
            data = json.load(j)
            print(data)
            commit = data.get("aie_partition", {}).get("pre_post_fingerprint", "0")
            # print(f"====== commit : {commit}")
            commit_16 = hex(int(commit))
            # print(f"====== commit_16 : {str(commit_16)[2:]}")
            target_info += (
                f"<b>dpu_phx_kernel_for_stx commit:</b> {str(commit_16)[2:]}" + "<br>\n"
            )

    else:
        print(f"xclbin info {toolsVersion_json} not exist!", flush=True)
        target_info += "<br>\n"

    original_pdi_elf = os.environ.get("PDI_ELF", "")
    if original_pdi_elf:
        pdi_elf_basename = os.path.basename(original_pdi_elf)
        pdi_elf_package_md5 = ""
        if pdi_elf_basename.endswith(".zip"):
            pdi_elf_package = os.path.join(workspace, pdi_elf_basename)
            pdi_elf_package_md5 = md5sum(pdi_elf_package)
        pdi_elf_url = jenkins_url + "artifact/" + pdi_elf_basename
        target_info += (
            "<br>"
            + '\n<b>pdi elf download path: </b>\n<a href="%s">%s</a> %s'
            % (pdi_elf_url, pdi_elf_url, pdi_elf_package_md5)
            + "<br>\n"
        )
        target_info += (
            "<b>original pdi elf path: </b>"
            + '<a href="%s">%s</a>' % (original_pdi_elf, original_pdi_elf)
            + "<br><br>\n"
        )

    # dirver path
    driver_version_map = get_driver_version()
    if driver_version_map:
        driver_version, driver_url = driver_version_map
        if driver_url:
            target_info += (
                '<b>NPU Driver version:</b> %s, <b>Download URL:</b> <a href="%s">%s</a>'
                % (driver_version, driver_url, driver_url)
                + "<br><br>\n"
            )
        else:
            target_info += (
                '<b><font color="red">Your NPU driver(%s) is not the one regression used!</font></b><br>\n '
                % driver_version
                + '<b>Recommended NPU Driver version:</b> %s, <b>Download URL:</b> <a href="%s">%s</a>'
                % (
                    list(constant.DRIVER_MAP.keys())[-1],
                    constant.DRIVER_MAP.get(list(constant.DRIVER_MAP.keys())[-1]),
                    constant.DRIVER_MAP.get(list(constant.DRIVER_MAP.keys())[-1]),
                )
                + "<br><br>\n"
            )
            print("Your driver is not our regression driver")
    else:
        target_info += (
            '<b><font color="red">ERROR: Not got the NPU version</font></b><br>\n '
            + '<b>Recommended NPU Driver version:</b> %s, <b>Download URL:</b> <a href="%s">%s</a>'
            % (
                list(constant.DRIVER_MAP.keys())[-1],
                constant.DRIVER_MAP.get(list(constant.DRIVER_MAP.keys())[-1]),
                constant.DRIVER_MAP.get(list(constant.DRIVER_MAP.keys())[-1]),
            )
            + "<br><br>\n"
        )
        print("ERROR: failed to get the driver version")

    # get modelzoo path info
    modelzoo_saved_path_list = []
    if os.environ.get("USER_MODEL_ZOO", ""):
        model_zoo_url = (
            jenkins_url + "artifact/" + "%s.json" % os.environ["USER_MODEL_ZOO"]
        )
        target_info += "<b>modelzoo path:</b> %s<br><br>\n" % model_zoo_url
    elif os.environ.get("MODEL_ZOO", ""):
        modelzoo_url = (
            "https://gitenterprise.xilinx.com/VitisAI/vaip/blob/dev/ci/%s.json"
            % os.environ["MODEL_ZOO"]
        )
        target_info += '<b>modelzoo path:</b> <a href="%s">%s</a> <br><br>\n' % (
            modelzoo_url,
            modelzoo_url,
        )
    if modelzoo_saved_path_list:
        target_info += "\n".join(modelzoo_saved_path_list) + "<br><br>\n"

    # get COMPUTERNAME info
    target_info += (
        ""
        if not os.environ.get("COMPUTERNAME", "")
        else "<b>COMPUTERNAME:</b>" + os.environ["COMPUTERNAME"] + "<br>\n"
    )

    return target_info


def get_test_time(model_result_dict, result_lines):
    # get time consumption
    if re.match("TEST_START_TIME=(.*)", result_lines[-3]):
        model_result_dict["TEST_START_TIME"] = (
            re.match("TEST_START_TIME=(.*)", result_lines[-3]).group(1).strip()
        )
    if re.match("TEST_END_TIME=(.*)", result_lines[-2]):
        model_result_dict["TEST_END_TIME"] = (
            re.match("TEST_END_TIME=(.*)", result_lines[-2]).group(1).strip()
        )

    if re.match("Build elapse seconds:(.*)", result_lines[-1]):
        elapse = (
            re.match("Build elapse seconds:(.*)", result_lines[-1]).group(1).strip()
        )
        # model_result_dict["TEST_ELAPSED"] = int(float(elapse.strip()))
    else:
        elapse = "NA"
    return elapse


def write_file(file_path, content):
    try:
        with open(file_path, "w") as f:
            f.write(content)
    except Exception as e:
        print("Write failed! %s" % e)


def cp_file(src, dst):
    try:
        print("cp %s to %s" % (src, dst))
        shutil.copyfile(src, dst)
    except Exception as e:
        print("Copy file failed! %s" % e)


def remove_dir(dir_path):
    try:
        print("remove dir %s" % dir_path)
        shutil.rmtree(dir_path)
    except Exception as e:
        print("Copy file failed! %s" % e)


def dump_dict_to_json(dict_data, json_path):
    try:
        with open(json_path, "w") as f:
            json.dump(dict_data, f, indent=4)
    except Exception as e:
        print("Dump dict to file failed! %s" % e)


def dump_dict_to_txt(dict_data, file_txt):
    try:
        with open(file_txt, "w") as f:
            for key, val in dict_data.items():
                status = val.get("TEST_STATUS", "NA")
                f.write(f"{status},{key}\n")
    except Exception as e:
        print("Dump dict to file failed! %s" % e)


def read_lines(file_path):
    try:
        if not os.path.exists(file_path):
            print("File not existed! %s" % file_path)
            return []
        # with open(file_path, "r", encoding="utf-8") as f:
        with open(file_path, "r") as f:
            return f.readlines()
    except Exception as e:
        print("Write failed! %s" % e)


def load_json_file(json_file):
    data = {}
    try:
        with open(json_file) as j:
            data = json.load(j)
    except Exception as e:
        print("Load json file failed! %s" % e)
    finally:
        return data


def cal_elapsed(strtime_start="", strtime_end=""):
    # time format must be "%d-%m-%Y %H:%M:%S"
    if (not strtime_start) or (not strtime_end):
        return ""
    time1 = datetime.datetime.strptime(strtime_start, "%d-%m-%Y %H:%M:%S")
    time2 = datetime.datetime.strptime(strtime_end, "%d-%m-%Y %H:%M:%S")
    suite_run_time = time2 - time1
    if suite_run_time:
        return suite_run_time
    else:
        return ""


def key_word_found(key_word, result_lines):
    flag = False
    for line in result_lines:
        if key_word in line:
            flag = True
            break
    return flag


def pattern_match(pattern, result_lines):
    for line in result_lines:
        m = re.match(pattern, line)
        if m:
            return m
    return None


def pattern_match_last(pattern, result_lines):
    last_match = None
    for line in result_lines:
        m = re.match(pattern, line)
        if m:
            last_match = m
    return last_match


def geomean(xs):
    return math.exp(math.fsum(math.log(x) for x in xs) / len(xs))


def get_silicon_time(target_type, build_id):
    from openpyxl import Workbook, load_workbook

    silicon_time_dict = {}
    try:
        pattern = r"vaitrace_profiling_%s_.*?_%s_.*?G\.xlsx" % (target_type, build_id)
        workspace = os.environ.get("WORKSPACE", "")
        if not workspace:
            return {}
        matched_excel_files = [x for x in os.listdir(workspace) if re.match(pattern, x)]
        if len(matched_excel_files) != 1:
            return {}
        profiling_file = matched_excel_files[0]

        print("Profiling data: %s " % profiling_file)

        wb = load_workbook(os.path.join(workspace, profiling_file), data_only=True)
        sheet_name_list = wb.get_sheet_names()

        for sheet_name in sheet_name_list:
            if sheet_name in ("summary", "op summary"):
                continue
            print("Sheet name: %s" % sheet_name)
            sheet = wb[sheet_name]
            data_all = sheet.columns
            data_tuple = tuple(data_all)
            for col in data_tuple:
                if not col[0].value:
                    continue
                if not re.match(r".*?silicon.*?time\(us\).*?", repr(col[0].value)):
                    continue
                silicon_time_sum = 0
                for cell in col[1:]:
                    if cell.value and cell.value != "None":
                        silicon_time_sum += float(cell.value)
                silicon_time_dict[sheet_name] = round(silicon_time_sum / 1000, 2)

    except Exception as e:
        print(e)
    finally:
        return silicon_time_dict


def write_xrt_ini(aietrace=False):
    if aietrace:
        int_content = constant.XRT_INI_AIE_TRACE
    else:
        int_content = constant.XRT_INI_VERBOSITY
    try:
        with open("xrt.ini", "w") as f:
            f.write(int_content)
    except Exception as e:
        print(e)


def scp_dir(host_name, src, dst):
    try:
        cmd_list = [
            "scp",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-r",
            host_name + ":" + src,
            dst,
        ]
        print(f"{(' ').join([str(each) for each in cmd_list])}", flush=True)
        subprocess.check_call(cmd_list)

    except Exception as e:
        print(e)


def pip_install_repos(repo_list):
    # python -m pip install opencv-python lpips natsort scikit-image
    try:
        cmd_list = [
            "python",
            "-m",
            "pip",
            "install",
        ]
        for repo in repo_list:
            cmd_list.append(repo)
        print(f"{(' ').join([str(each) for each in cmd_list])}", flush=True)
        subprocess.check_call(cmd_list)

    except Exception as e:
        print(e)


def pip_install_requirements(requirements_file):
    # python -m pip install -r requirements.txt
    try:
        cmd_list = [
            "python",
            "-m",
            "pip",
            "install",
            "-r",
            requirements_file,
        ]
        print(f"{(' ').join([str(each) for each in cmd_list])}", flush=True)
        subprocess.check_call(cmd_list)

    except Exception as e:
        print(e)


def bin2png(src, dst):
    # src and dst are dir
    try:
        import cv2
        import numpy as np
    except Exception as e:
        print("ERROR: import failed. %s" % e)
        return

    src_files = [x for x in os.listdir(src)]
    if not os.path.exists(dst):
        os.makedirs(dst)
    print("Start to convet bin:%s to png:%s" % (src, dst))
    for src_file_name in src_files:
        if src_file_name == "io.json" or src_file_name.startswith("onnx-dpu_interface"):
            continue
        try:
            src_file = os.path.join(src, src_file_name)
            dst_file = os.path.join(dst, "%s.png" % os.path.splitext(src_file_name)[0])

            sr0 = np.fromfile(src_file, dtype=np.float32).reshape([512, 512, 3])
            sr1 = ((sr0 + 1) / 2.0) * 255
            sr2 = cv2.cvtColor(sr1, cv2.COLOR_RGB2BGR)
            cv2.imwrite(dst_file, sr2)
        except Exception as e:
            print("ERROR: %s with error: %s" % (src_file_name, e))


def get_verbose():
    try:
        system_version = ""
        verbose = ""
        xcom_opt = ""
        bios_version = ""
        for model_name in get_model_list():
            log = os.path.join(get_vaip_regression_dir(), model_name, "build.log")
            if not os.path.isfile(log):
                return
            result_lines = read_lines(log)
            first_half = []
            isFound = True
            for line in result_lines:
                if "OS Version" in line and (not system_version):
                    system_version += f"<br>{line}"
                if "BIOS Version:" in line and (not bios_version):
                    bios_version += f"<br>{line}"
                    system_version += bios_version
                if "Begin to compile" in line:
                    isFound = False
                if isFound:
                    first_half.append(line)
            if not verbose:
                pattern_ver = re.compile(
                    r"\[XLNX_ONNX_EP_VERBOSE\]\s(?!.*EXEC\sVERISON\S*)(.+=.+)$"
                )
                for item in first_half:
                    verb = pattern_ver.search(item)
                    if verb:
                        verbose += "<font> %s </font><br>" % verb.group(1) + "\n"
            if not xcom_opt:
                xcom_opt_ver = re.compile(r".*?(Enable.*optimization.*)")
                for item in first_half:
                    xcom_verb = xcom_opt_ver.search(item)
                    if xcom_verb:
                        xcom_opt += (
                            "<font> %s </font><br>" % xcom_verb.group(1).strip() + "\n"
                        )
            if verbose and xcom_opt:
                break

        verbose = "<b>Xcompiler Attrs: </b><br>" + verbose if verbose else verbose
        xcom_opt = (
            "<b>Xcompiler Optimizations: </b><br>" + xcom_opt if xcom_opt else xcom_opt
        )
        return system_version, verbose, xcom_opt
    except Exception as e:
        print(f"!!! warning : get verbose failed! {e}.)")


def get_vaip_regression_dir():
    if platform.system() == "Linux":
        build_dir = Path("/home") / os.environ["USER"] / Path("build/vaip_regression")
        return build_dir
    workspace_home = (
        pathlib.Path.home()
        if not os.environ.get("JOB_BASE_NAME", "")
        else Path(r"C:\\IPU_workspace") / os.environ["JOB_BASE_NAME"]
    )
    if not os.path.exists(workspace_home):
        os.makedirs(workspace_home)
    return workspace_home / os.environ.get("VAIP_REGRESSION", "vaip_regression")


def get_model_list():
    modelzoo = os.environ.get("MODEL_ZOO", "")
    modelzoo_json = os.path.join(CWD, "ci", modelzoo + ".json")
    env_file = os.environ.get("VAIP_REGRESSION", "")
    json_result = os.environ.get("OUTPUT_JSON", env_file + ".json")
    if (not modelzoo) or (not os.path.exists(modelzoo_json)):
        logging.warning("No MODEL_ZOO got in env!")
        return
    with open(modelzoo_json, "r") as mz:
        modelzoo_list = json.load(mz)
    mz.close()
    run_model_list = []
    if os.path.exists(json_result):
        with open(json_result, "r") as jr:
            json_result_data = json.load(jr)
        jr.close()
        run_model_list = [x.get("id", "") for x in json_result_data]
    print("All model list:%s" % run_model_list, flush=True)

    if os.environ.get("CASE_NAME", ""):
        model_list = os.environ.get("CASE_NAME", "").strip().split(" ")
    else:
        model_list = [x.get("id", "") for x in modelzoo_list if x.get("id", "")]
    if os.environ.get("FORBID_CASE", ""):
        forbid_models = os.environ.get("FORBID_CASE", "").split(" ")
        model_list = [x for x in model_list if x and (x not in forbid_models)]
    if (
        os.environ.get("ONLY_REPORT", "") == "true"
        or os.environ.get("INCREMENTAL_TEST", "") == "true"
    ):
        vaip_regression_dir = get_vaip_regression_dir()
        modelzoo_models = [x.get("id", "") for x in modelzoo_list if x.get("id", "")]
        model_list = [
            x for x in modelzoo_models if x in os.listdir(vaip_regression_dir)
        ]
    else:
        model_list = [x for x in model_list if x in run_model_list]

    print("Run model list:%s" % model_list, flush=True)
    return model_list


def html_to_json(html_string):
    html_string = html_string.replace("\n", "").replace("<br>", "")

    pattern = re.compile(r"<font>\s*(.*?)\s*=\s*(.*?)\s*</font>")
    matches = pattern.findall(html_string)

    verbose_data = {}
    for match in matches:
        key = match[0].strip()
        value = match[1].strip()

        if value.lower() == "true":
            value = True
        elif value.lower() == "false":
            value = False
        elif value.isdigit():
            value = int(value)
        else:
            try:
                value = float(value)
            except ValueError:
                pass

        verbose_data[key] = value
    return verbose_data


if __name__ == "__main__":
    get_driver_version()
