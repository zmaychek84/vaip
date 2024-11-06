#
#   The Xilinx Vitis AI Vaip in this distribution are provided under the following free
#   and permissive binary-only license, but are not provided in source code form.  While the following free
#   and permissive license is similar to the BSD open source license, it is NOT the BSD open source license
#   nor other OSI-approved open source license.
#
#    Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
#
#    Redistribution and use in binary form only, without modification, is permitted provided that the following conditions are met:
#
#    1. Redistributions must reproduce the above copyright notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the distribution.
#
#    2. The name of Xilinx, Inc. may not be used to endorse or promote products redistributed with this software without specific
#    prior written permission.
#
#    THIS SOFTWARE IS PROVIDED BY XILINX, INC. "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
#    THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL XILINX, INC.
#    BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
#    PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
#    OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE
#


import argparse
import os
import platform
import re
import shutil
import subprocess
import sys
import stat
import time
import json
import logging
import traceback
from pathlib import Path
import tarfile
import zipfile
from zipfile import ZipFile

from tools import utility
from tools import constant

IS_CROSS_COMPILATION = False
IS_WINDOWS = False
IS_NATIVE_COMPILATION = False
if "OECORE_TARGET_SYSROOT" in os.environ:
    IS_CROSS_COMPILATION = True
elif platform.system() == "Windows":
    IS_WINDOWS = True
else:
    IS_NATIVE_COMPILATION = True

CWD = Path.cwd()
HOME = Path.home()


def main(args):
    parser = argparse.ArgumentParser(description="Script to run ci build and test.")
    subparsers = parser.add_subparsers()

    parser_build = subparsers.add_parser("config")
    parser_build.set_defaults(func=config)

    parser_build = subparsers.add_parser("build")
    parser_build.set_defaults(func=build)

    parser_build = subparsers.add_parser("download_xclbin")
    parser_build.set_defaults(func=download_xclbin)

    parser_init_test = subparsers.add_parser("init_test")
    parser_init_test.add_argument("-p", help="voe package path")
    parser_init_test.set_defaults(func=init_test)

    parser_test_modelzoo = subparsers.add_parser("test_modelzoo")
    parser_test_modelzoo.add_argument("-c", default="", help="case name")
    parser_test_modelzoo.add_argument("-f", help="package path for test")
    parser_test_modelzoo.add_argument("-j", default="1", help="parallel run number")
    parser_test_modelzoo.set_defaults(func=test_modelzoo)

    parser_get_result = subparsers.add_parser("get_result")
    parser_get_result.add_argument("-f", help="package path for test")
    parser_get_result.set_defaults(func=get_result)

    args = parser.parse_args()
    args.func(args)


def checkoutByCommit(commit):
    match_obj = re.match("pr(\d+)", commit)
    if match_obj:
        subprocess.check_call(
            ["git", "fetch", "-u", "origin", f"pull/{match_obj.group(1)}/head:{commit}"]
        )
        subprocess.check_call(["git", "checkout", "--force", commit])
    else:
        subprocess.check_call(["git", "checkout", "--force", commit])
        # subprocess.check_call(["git", "pull", "--rebase"])
    subprocess.check_call(["git", "rev-parse", "HEAD"])


def update_to_latest(contain_submodule):
    subprocess.check_call(["git", "fetch", "--all"])
    subprocess.check_call(["git", "checkout", "."])
    subprocess.check_call(["git", "clean", "-xfd"])
    subprocess.check_call(["git", "pull", "--rebase"])
    subprocess.check_call(["git", "rev-parse", "HEAD"])
    if contain_submodule:
        subprocess.check_call(["git", "submodule", "sync", "--recursive"])
        subprocess.check_call(["git", "submodule", "update", "--init", "--recursive"])


def set_proxy():
    os.environ["https_proxy"] = "http://xcdl190074:9181"


def set_env():
    if IS_WINDOWS:
        os.environ["HOME"] = os.environ["USERPROFILE"]

    if os.environ.get("DUMP_MC_CODE", "false") == "true":
        os.environ["SKIP_CLEAN_GRAPH"] = "ON"

    os.environ["DEBUG_LOG_LEVEL"] = "info"
    os.environ["WITH_CPURUNNER"] = "ON"
    os.environ["WITH_XCOMPILER"] = "ON"
    os.environ["USE_CPYTHON"] = "OFF"
    os.environ["BUILD_PYTHON_EXT"] = "OFF"
    os.environ["EN_LLM_DOD_OPS"] = "OFF"
    if os.environ.get("CI_WITH_DOD", "true") == "true":
        os.environ["EN_LLM_DOD_OPS"] = "ON"
    os.environ["onnxruntime_DONT_VECTORIZE"] = "ON"
    os.environ["VAI_RT_WORKSPACE"] = str(CWD / "hw")
    if IS_CROSS_COMPILATION:
        os.environ["VAI_RT_BUILD_DIR"] = str(CWD / "cb")
    else:
        os.environ["VAI_RT_PREFIX"] = str(CWD / "hi")
        os.environ["VAI_RT_BUILD_DIR"] = str(CWD / "hb")


def set_test_env():
    os.environ["BUILD_TYPE"] = "Release"
    os.environ["USE_EP_CONTEXT"] = "true"
    os.environ["ENABLE_CACHE_FILE_IO_IN_MEM"] = "0"
    os.environ["XLNX_MINIMUM_NUM_OF_CONV"] = "0"
    os.environ["XLNX_ENABLE_CACHE"] = "1"
    os.environ["DEBUG_RUNNER"] = "1"
    os.environ["XLNX_ONNX_EP_VERBOSE"] = "2"
    os.environ["DEBUG_GRAPH_RUNNER"] = "1"
    if os.environ.get("CACHED_ONNX_CLEAN", "false") == "true":
        os.environ["XLNX_ENABLE_DUMP_XIR_MODEL"] = "0"
        os.environ["XLNX_ONNX_EP_REPORT_FILE"] = ""
        os.environ["XLNX_ONNX_EP_DPU_REPORT_FILE"] = ""
    else:
        os.environ["XLNX_ENABLE_DUMP_XIR_MODEL"] = "1"
        os.environ["XLNX_ONNX_EP_REPORT_FILE"] = "vitisai_ep_report.json"
        os.environ["XLNX_ONNX_EP_DPU_REPORT_FILE"] = "vitisai_ep_dpu_report.json"
    os.environ["PATH"] = str(CWD / "hi" / "bin") + os.pathsep + os.environ["PATH"]
    if IS_WINDOWS:
        if Path.is_dir(CWD / "hi" / "python"):
            os.environ["PATH"] = (
                str(CWD / "hi" / "python") + os.pathsep + os.environ["PATH"]
            )
            os.environ["PATH"] = (
                str(CWD / "hi" / "python" / "Scripts") + os.pathsep + os.environ["PATH"]
            )
            if os.path.exists(CWD / "hi" / "python" / "python310._pth"):
                update_pth(str(CWD / "hi" / "python" / "python310._pth"))
            else:
                update_pth(str(CWD / "hi" / "python" / "python39._pth"))
    else:
        os.environ["PATH"] = (
            str(HOME / ".local" / "bin") + os.pathsep + os.environ["PATH"]
        )
        os.environ["LD_LIBRARY_PATH"] = (
            str(CWD / "hi" / "lib") + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
        )


def remove_readonly(func, path, exc_info):
    # "Clear the readonly bit and reattempt the removal"
    # ERROR_ACCESS_DENIED = 5

    if func not in (os.unlink, os.rmdir) or exc_info[1].winerror != 5:
        raise exc_info[1]
    os.chmod(path, stat.S_IWRITE)
    func(path)


def get_build_cmd(copy_base, build_type):
    base_file = get_release_file()
    print(base_file, flush=True)
    win_list = (
        [
            "demo_multi_model",
            "demo_yolov8",
        ]
        if IS_WINDOWS
        else []
    )
    if IS_WINDOWS and os.environ.get("USE_CPYTHON", "ON") == "ON":
        win_list.append("cpython")
        win_list.append("static_cpython")
        win_list.append("numpy")
        win_list.append("static_python")
        win_list.append("onnx")
        win_list.append("protobuf")
        win_list.append("py_protobuf")

    cmd = ["python", "main.py", "--type", build_type]
    cmd.extend(["--release_file", base_file])
    cmd.extend(["--package"])
    if os.environ.get("PACKAGE_VAITRACE", "true") == "true":
        cmd.append("--vaitrace")
    if os.environ.get("SKIP_BUILD", "false") == "true":
        cmd.append("--skip_build")

    if copy_base:
        cmd.extend(
            [
                "--project",
                "gtest",
                "unilog",
                "xir",
                "target_factory",
                "xcompiler",
            ]
        )
        if IS_WINDOWS and os.environ.get("CI_WITH_DOD", "true") == "true":
            cmd.extend(
                [
                    "transformers_header",
                    "aie_rt",
                    "dod",
                ]
            )
        if os.environ.get("PACKAGE_VAITRACE", "true") == "true":
            cmd.append("vaitrace")
        if os.environ.get("PACKAGE_XCOMPILER", "") != "true":
            cmd.extend(
                [
                    "vart",
                    "trace_logging",
                    "graph-engine",
                    "vairt",
                    *win_list,
                    "onnxruntime",
                    "vaip",
                    "testcases",
                    "test_onnx_runner",
                ]
            )
        if os.environ.get("WIN24_BUILD", "OFF") == "ON":
            cmd.extend(["vaip_xclbin"])

    print(cmd, flush=True)
    return cmd


def update_repo(project, name, submodule, repo_branch=None):
    if not Path.is_dir(CWD / name):
        for i in range(0, 2):
            try:
                ck_cmd = ["git", "clone"]
                if submodule:
                    ck_cmd.append("--recurse-submodules")
                if repo_branch:
                    ck_cmd.append("--branch")
                    ck_cmd.append(repo_branch)
                ck_cmd.append(f"git@gitenterprise.xilinx.com:{project}/{name}")
                print(ck_cmd, flush=True)
                subprocess.check_call(ck_cmd)
            except Exception:
                continue
            break
    else:
        for i in range(0, 2):
            try:
                os.chdir(CWD / name)
                update_to_latest(submodule)
                os.chdir(CWD)
            except Exception:
                continue
            break


def get_submodule_commits(name, branch):
    submodule_commits = {}
    os.chdir(CWD / name)
    subprocess.check_call(["git", "checkout", branch])
    subprocess.check_call(["git", "submodule", "update", "--init", "--recursive"])
    for each in ["xir", "target_factory", "xcompiler", "vart"]:
        os.chdir(CWD / name / "lib" / each)
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("ascii")
            .strip()
        )
        submodule_commits[each] = commit
        print(f"{each}: {commit}", flush=True)
        os.chdir(CWD)
    submodule_commit_file = os.environ.get(
        "DPU_PHX_SUBMODULE_COMMIT", "dpu_phx_submodule_commits.txt"
    )
    print(submodule_commits, flush=True)
    with open(submodule_commit_file, "w") as f:
        for k, v in submodule_commits.items():
            f.write(f"{k}:{v}\n")
    f.close()
    return submodule_commits


def get_commit_ids(release_file):
    commit_ids = {}
    with open(release_file, "r") as f:
        lines = f.readlines()
        lines = [re.sub("#.*", "", line).strip() for line in lines]
        lines = [line for line in lines if len(line) > 0]
        for line in lines:
            words = line.split(":")
            words = [word.strip(" \t") for word in words]
            lib = words[0]
            commit = words[1]
            param = lib.upper() + "_BRANCH"
            param1 = param.replace("-", "_")
            if os.environ.get(param):
                commit = os.environ[param]
                print(f"{param}: {commit}", flush=True)
            elif os.environ.get(param1):
                commit = os.environ[param1]
                print(f"{param1}: {commit}", flush=True)
            commit_ids[lib] = commit
    f.close()
    return commit_ids


def generate_release_file(submodule_commits):
    os.chdir(CWD / "vai-rt")
    tmp_file = os.environ.get("RELEASE_FILE", "release_file/latest_dev.txt")
    base_file = os.path.basename(tmp_file)
    all_recipe_commits = get_commit_ids(tmp_file)
    print(all_recipe_commits, flush=True)

    release_file = os.environ.get("CURRENT_RELEASE_FILE", "release_file_1.txt")
    print(f"Write new release file: {release_file}", flush=True)
    with open(release_file, "w") as f:
        for k, v in all_recipe_commits.items():
            if k in submodule_commits.keys() and v != submodule_commits[k]:
                v = submodule_commits[k]
            print(f"{k}: {v}", flush=True)
            f.write(f"{k}: {v}\n")
    f.close()


def config(args):
    set_env()
    set_env_into_os("USER_ENV")
    set_env_into_os("CI_ENV")
    set_big_buffer_driver_env()
    print(os.environ, flush=True)

    update_repo("VitisAI", "vai-rt", False)
    os.chdir(CWD / "vai-rt")
    if os.environ.get("VAI_RT_BRANCH"):
        checkoutByCommit(os.environ.get("VAI_RT_BRANCH"))
    elif not IS_WINDOWS:
        subprocess.check_call(["git", "pull"])
    submodule_commits = {}
    if os.environ.get("UPDATE_RELEASE_FILE", "") == "true":
        os.chdir(CWD)
        update_repo("IPU", "DPU_PHX", True, repo_branch="strix")
        submodule_commits = get_submodule_commits("DPU_PHX", "strix")

    generate_release_file(submodule_commits)


def build(args):
    set_env()
    set_env_into_os("USER_ENV")
    set_env_into_os("CI_ENV")
    set_big_buffer_driver_env()
    print(os.environ, flush=True)

    # set_proxy()
    # remove existed tar.gz file
    gz_files = []
    if os.path.exists(os.environ["VAI_RT_BUILD_DIR"]):
        gz_files = [
            x
            for x in os.listdir(os.environ["VAI_RT_BUILD_DIR"])
            if x.endswith("tar.gz")
        ]
    if gz_files:
        for gz_file in gz_files:
            gz_file_path = os.path.join(os.environ["VAI_RT_BUILD_DIR"], gz_file)
            logging.info("rm file: %s" % gz_file_path)
            os.remove(gz_file_path)
    tb = None
    try:
        sys.path.insert(0, os.path.abspath("vai-rt"))
        os.chdir(CWD / "vai-rt")

        build_type = "release"
        if os.environ.get("DEBUG_BUILD", "") == "true":
            build_type = "debug"

        workspace = os.environ.get("WORKSPACE", "")
        job_name = os.environ.get("JOB_NAME", "job_name")
        if "/" in job_name:
            jenkins_workspace = Path(workspace[: workspace.find(job_name)])
        else:
            jenkins_workspace = CWD.parent
        print("Jenkins workspace %s" % jenkins_workspace)

        install_base_in_ws = Path("C:\\") / "workspace" / "host_install_base"
        if Path.is_dir(install_base_in_ws):
            base_install = install_base_in_ws
        else:
            base_install = jenkins_workspace / "host_install_base"

        if Path.is_dir(jenkins_workspace / f"host_install_base_{build_type}"):
            base_install = jenkins_workspace / f"host_install_base_{build_type}"
        if Path.is_dir(install_base_in_ws / f"host_install_base_{build_type}"):
            base_install = install_base_in_ws / f"host_install_base_{build_type}"

        if not IS_WINDOWS:
            base_install = jenkins_workspace / "linux_host_install_base"
            if Path.is_dir(jenkins_workspace / f"linux_host_install_base_{build_type}"):
                base_install = (
                    jenkins_workspace / f"linux_host_install_base_{build_type}"
                )

        if os.environ.get("HOST_INSTALL_BASE", ""):
            base_install = Path(os.environ["HOST_INSTALL_BASE"])

        cwd_install = CWD / "hi"
        copy_base = False
        if not Path.is_dir(cwd_install) and Path.is_dir(base_install):
            print(f"==copy {base_install} to {cwd_install}", flush=True)
            shutil.copytree(base_install, cwd_install)
            copy_base = True
        elif Path.is_dir(cwd_install):
            copy_base = True

        if os.environ.get("BASE_BUILD_FULLY", "") == "true":
            copy_base = False

        cmd = get_build_cmd(copy_base, build_type)
        if os.environ.get("ONLY_REPORT", "") != "true":
            subprocess.check_call(cmd)

        if os.environ.get("UPDATE_RELEASE_FILE", "") == "true":
            tmp_file = os.environ.get("RELEASE_FILE", "release_file/latest_dev.txt")
            base_file = os.path.basename(tmp_file)
            release_file = os.environ.get("CURRENT_RELEASE_FILE", "release_file_1.txt")
            org_file = CWD / "vai-rt" / release_file
            det_file = CWD / "vai-rt" / "release_file" / f"{base_file}"
            shutil.copyfile(org_file, det_file)
            print(f"git add release_file/{base_file}", flush=True)
            try:
                subprocess.check_call(["git", "add", f"release_file/{base_file}"])
                print(f"git commit -m 'sync with DPU_PHX'", flush=True)
                subprocess.check_call(["git", "commit", "-m", "sync with DPU_PHX"])
                print(f"git push origin dev", flush=True)
                subprocess.check_call(["git", "push", "origin", "dev"])
            except Exception as e:
                print(e, flush=True)
        if os.environ.get("PACKAGE_XCOMPILER", "") != "true":
            install_voe()
    except Exception:
        tb = traceback.format_exc()
        if not tb is None:
            print(tb)
        raise


def get_package_name(dir_name, postfix):
    if not os.path.exists(dir_name):
        return "NotFound"
    files = os.listdir(dir_name)
    for f in files:
        if f.endswith(postfix):
            return os.path.join(dir_name, f)

    return "NotFound"


def install_voe():
    set_env()
    # set_proxy()
    if IS_NATIVE_COMPILATION:
        vaip_dist_dir = get_package_name(
            f"{os.environ['VAI_RT_BUILD_DIR']}/vaip/onnxruntime_vitisai_ep/python/dist",
            "whl",
        )
        if vaip_dist_dir == "NotFound":
            return
        cmd = ["pip", "install", vaip_dist_dir, "--force-reinstall", "--user"]
        print(cmd, flush=True)
        subprocess.check_call(cmd)


def get_release_file():
    return os.environ.get("CURRENT_RELEASE_FILE", "release_file/latest_dev.txt")


def get_commit(r, latest_file_commits):
    param = r.name().upper() + "_BRANCH"
    if os.environ.get(param):
        return os.environ[param]

    if r.name() in latest_file_commits and r.name() not in (
        "vart",
        "vaip",
        "test_onnx_runner",
    ):
        return latest_file_commits[r.name()]

    commit_id = (
        subprocess.check_output(["git", "ls-remote", r.git_url(), r.git_branch()])
        .decode("ascii")
        .strip()
        .split()[0][:8]
    )
    return commit_id


def get_unit_model():
    unit_model = "null"
    try:
        sample_json = os.path.join(CWD, "ci", "sample.json")
        samples = {}
        with open(sample_json, "r") as mz:
            samples = json.load(mz)
        mz.close()
        print(samples, flush=True)
        default_target = "AMD_AIE2P_4x4_Overlay_CFG0"
        unit_model = samples.get(default_target, unit_model)
        target = os.environ.get("XLNX_TARGET_NAME", default_target)
        print(target, flush=True)
        unit_model = samples.get(target, unit_model)
        print(unit_model, flush=True)
    except Exception as e:
        logging.warning(f"!!! warning : get unit model failed! {e}.)")
    finally:
        return unit_model


def test_sample(extract_dir):
    if os.environ.get("CI_UNIT", "") != "true":
        return
    try:
        if not os.environ.get("BIG_BUFFER_DRIVER", ""):
            set_big_buffer_driver_env()
        if os.environ.get("BIG_BUFFER_DRIVER", "false") != "true":
            return
        if platform.system() == "Linux":
            return

        os.chdir(CWD / extract_dir)
        if Path.is_file(CWD / extract_dir / "python" / "python.exe"):
            python_dir = str(CWD / extract_dir / "python" / "python.exe")
            cmd = []
            if Path.is_file(CWD / extract_dir / "install_deps.py"):
                cmd = [
                    python_dir,
                    "install_deps.py",
                ]
                print(cmd, flush=True)
                subprocess.check_call(cmd)
            if Path.is_file(CWD / extract_dir / "installer.py"):
                cmd = [
                    python_dir,
                    "installer.py",
                ]
                print(cmd, flush=True)
                subprocess.check_call(cmd)

            set_test_env()
            # os.environ["VAIP_COMPILE_RESERVE_CONST_DATA"] = "1"
            os.environ["XLNX_ENABLE_CACHE"] = "0"

            set_env_into_os("USER_ENV")
            set_env_into_os("CI_ENV")
            print(os.environ, flush=True)
            # os.chdir(CWD / extract_dir / "vitis_ai_ep_py_sample")
            # cmd = [
            #     python_dir,
            #     "resnet50_python/test.py",
            # ]

            os.chdir(CWD / extract_dir / "vitis_ai_ep_cxx_sample")
            scp_files("xcdl190074", get_unit_model(), "unit.onnx")
            cmd = [
                "onnxruntime_perf_test.exe",
                "-I",
                "-e",
                "vitisai",
                "-i",
                "config_file|..\\bin\\vaip_config.json",
                "-t",
                "10",
                "-c",
                "1",
                ".\\unit.onnx",
            ]
            print(cmd, flush=True)
            subprocess.check_call(cmd)
        else:
            python_dir = "python.exe"
            if platform.system() == "Linux":
                python_dir = "python"
            print(
                "%s not exist, ignore test sample"
                % str(CWD / extract_dir / "python" / "python"),
                flush=True,
            )
            cmd = [
                python_dir,
                "-m",
                "pip",
                "install",
                "voe",
                "--user",
                "--no-deps",
                "--force-reinstall",
                "--no-index",
                "--find-links",
                ".",
            ]
            print(cmd, flush=True)
            subprocess.check_call(cmd)
            cmd = [
                python_dir,
                "-m",
                "pip",
                "install",
                "onnxruntime_vitisai",
                "--user",
                "--no-deps",
                "--force-reinstall",
                "--no-index",
                "--find-links",
                ".",
            ]
            print(cmd, flush=True)
            subprocess.check_call(cmd)
    except Exception as e:
        logging.warning(f"!!! warning : test_sample failed! {e}.)")


def get_python():
    python_exe = "python"
    if IS_WINDOWS:
        python_exe = "python.exe"
        if Path.is_dir(CWD / "hi" / "python"):
            python_exe = str(CWD / "hi" / "python" / "python.exe")
    return python_exe


def reinstall_vaitrace():
    if os.environ.get("IGNORE_ARCHIVE", "") == "true":
        return
    print("reinstall vaitrace package")
    cwd = Path.cwd()
    try:
        # if os.environ.get("NODE", "").startswith("xcd"):
        #    set_proxy()
        os.chdir(cwd / "hi")
        python_exe = get_python()
        cmd = [
            python_exe,
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
        os.chdir(cwd)


def update_pth(pth_file):
    lines = []
    ci_path = "..\\..\\ci"
    ci_in_syspath = False
    with open(pth_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            if line.find(ci_path) != -1:
                ci_in_syspath = True
                print(f"Found: {line} in {pth_file}\n", flush=True)
    f.close()

    if not ci_in_syspath:
        with open(pth_file, "a+") as f:
            f.write(f"{ci_path}\n")
        f.close()


def get_cache_dir():
    if os.environ.get("XLNX_CACHE_DIR", ""):
        cache_dir = Path(os.environ.get("XLNX_CACHE_DIR"))
        return cache_dir

    cache_dir = (
        Path("C:\\") / "temp" / os.environ["USERNAME"]
        if IS_WINDOWS
        else Path("/tmp") / os.environ["USER"] / Path("vaip/.cache")
    )
    if IS_WINDOWS:
        if os.environ.get("JOB_BASE_NAME", "") and (
            not os.environ.get("XLNX_CACHE_DIR", "")
        ):
            cache_dir = (
                Path(r"C:\\IPU_workspace") / os.environ["JOB_BASE_NAME"] / "cache"
            )
            os.environ["XLNX_CACHE_DIR"] = str(cache_dir)

    return cache_dir


def clean_cache(cache_dir):
    if Path.is_dir(cache_dir):
        print("Clean cache: %s" % cache_dir, flush=True)
        shutil.rmtree(cache_dir)


def zip_cache(build_id):
    time.sleep(2)
    print("begin ----------------zip cache", flush=True)
    try:
        cache_dir = get_cache_dir()
        # zip cache
        cache_zip = os.environ.get("CACHE_LOG_ZIP", "cache_%s.zip" % build_id)
        with zipfile.ZipFile(cache_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(cache_dir):
                for file in files:
                    zipf.write(
                        os.path.join(root, file),
                        os.path.relpath(
                            os.path.join(root, file), os.path.join(cache_dir, "..")
                        ),
                    )
    except Exception as e:
        logging.warning(f"!!! warning : zip cache failed! {e}.)")


def copy_files(model_log_dir, vaip_regression_dir, model, postfix):
    tmp_files = [
        x
        for x in os.listdir(os.path.join(vaip_regression_dir, model))
        if x.endswith(postfix)
    ]
    for tmp_file in tmp_files:
        org_file = os.path.join(vaip_regression_dir, model, tmp_file)
        if os.path.isfile(org_file):
            try:
                det_file = os.path.join(model_log_dir, tmp_file)
                shutil.copyfile(org_file, det_file)
            except Exception as e:
                print("copy_files ERROR: %s" % e)


def get_scp_cmd_list(host_name, file_path, dest):
    cmd_list = [
        "scp",
        "-r",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        host_name + ":" + file_path,
        dest,
    ]
    return cmd_list


def scp_files(host_name, file_path, dest):
    cmd_list = get_scp_cmd_list(host_name, file_path, dest)
    try:
        print(f"{(' ').join([str(each) for each in cmd_list])}", flush=True)
        subprocess.check_call(cmd_list)
    except Exception as e:
        if host_name == "xsjncuph07":
            host_name = "xcdl190074"
        else:
            host_name = "xsjncuph07"
        cmd_list = get_scp_cmd_list(host_name, file_path, dest)
    try:
        print(f"{(' ').join([str(each) for each in cmd_list])}", flush=True)
        subprocess.check_call(cmd_list)
    except Exception as e:
        print(e, flush=True)


def zip_build_log(model_list, log_save_dir, compare_report, compare_perf_report):
    print("begin ----------------zip build log", flush=True)
    vaip_regression_dir = utility.get_vaip_regression_dir()
    # backup log
    for model in model_list:
        try:
            model_log_dir = os.path.join(log_save_dir, model)
            if not os.path.exists(model_log_dir):
                os.makedirs(model_log_dir)
            copy_files(model_log_dir, vaip_regression_dir, model, "log")
            copy_files(model_log_dir, vaip_regression_dir, model, "txt")
            copy_files(model_log_dir, vaip_regression_dir, model, "json")
            copy_files(model_log_dir, vaip_regression_dir, model, "csv")
            copy_files(model_log_dir, vaip_regression_dir, model, "xat")
            copy_files(model_log_dir, vaip_regression_dir, model, "bat")
            copy_files(model_log_dir, vaip_regression_dir, model, "onnx")
            copy_files(model_log_dir, vaip_regression_dir, model, "dll")
            copy_files(model_log_dir, vaip_regression_dir, model, "xmodel")
            copy_files(model_log_dir, vaip_regression_dir, model, "ini")
            copy_files(model_log_dir, vaip_regression_dir, model, "run_summary")
            layer_dir = os.path.join(vaip_regression_dir, model, "layer_result")
            if os.path.exists(layer_dir):
                ori_layer_json = os.path.join(
                    vaip_regression_dir, model, "layer_result/result.json"
                )
                det_layer_json = os.path.join(model_log_dir, "diff_result.json")
                shutil.copyfile(ori_layer_json, det_layer_json)
            org_trace = os.path.join(vaip_regression_dir, model, "NPU_Tracing")
            if os.path.exists(org_trace):
                det_trace = os.path.join(model_log_dir, "NPU_Tracing")
                shutil.copytree(org_trace, det_trace)
            org_aie_trace = os.path.join(
                vaip_regression_dir, model, "aietrace_timeline_output"
            )
            if os.path.exists(org_aie_trace):
                det_aie_trace = os.path.join(model_log_dir, "aietrace_timeline_output")
                shutil.copytree(org_aie_trace, det_aie_trace)
                # copy aietrace_visible.py to aietrace_timeline_output
                org_aietrace_visible = os.path.join(
                    CWD, "aietrace-enable/aietrace_visible.py"
                )
                if os.path.exists(org_aietrace_visible):
                    det_aietrace_visible = os.path.join(
                        det_aie_trace, "aietrace_visible.py"
                    )
                    shutil.copyfile(org_aietrace_visible, det_aietrace_visible)
        except Exception as e:
            logging.warning(f"!!! warning : build log {model} failed! {e}.)")

    # cp html report
    if os.path.exists(compare_report):
        det_html = os.path.join(log_save_dir, "compare_report.html")
        shutil.copyfile(compare_report, det_html)
    if os.path.exists(compare_perf_report):
        det_perf_html = os.path.join(log_save_dir, "compare_perf_report.html")
        shutil.copyfile(compare_perf_report, det_perf_html)

    # cp silicon time csv
    build_id2 = os.environ.get("BUILD_ID", "999")
    test_case = os.environ.get("TEST_CASE", "")
    csv_file = f"{test_case}_silicon_time_{build_id2}.csv"
    if os.path.exists(csv_file):
        csv_report = os.path.join(log_save_dir, csv_file)
        shutil.copyfile(csv_file, csv_report)

    # zip backup build log dir
    try:
        build_log_zip = f"{log_save_dir}.zip"
        with zipfile.ZipFile(build_log_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(log_save_dir):
                for file in files:
                    zipf.write(
                        os.path.join(root, file),
                        os.path.relpath(
                            os.path.join(root, file),
                            os.path.join(log_save_dir, ".."),
                        ),
                    )
    except Exception as e:
        print("zipfileERROR %s" % e)


def get_env_file(args):
    return os.environ.get("VAIP_REGRESSION", args.f.replace(",", "_"))


def get_new_package(package_name, download):
    new_name = package_name
    try:
        print(f"Current test using {new_name}.")
    except Exception as e:
        encode_name = new_name.encode("utf-8")
        print(f'Current test using "{encode_name}".', flush=True)

    if package_name.startswith("http"):
        base_name = os.path.basename(package_name)
        cwd_package = CWD / base_name
        new_name = str(cwd_package)
        if os.environ.get("IGNORE_ARCHIVE", "") == "true" or not download:
            print(f"Ignore download {package_name}...")
            return new_name
        print(f"Download {package_name}...")
        cwd_package_bak = CWD / (os.path.basename(package_name) + ".bak")
        try:
            if Path.is_file(cwd_package):
                shutil.move(cwd_package, cwd_package_bak)
            cmd = ["wget", "--quiet", "--no-check-certificate", package_name]
            print(" ".join(cmd))
            subprocess.check_call(cmd)
        except Exception as e:
            if Path.is_file(cwd_package_bak):
                shutil.copy(cwd_package_bak, cwd_package)
            logging.warning(f"!!! warning : get {package_name} failed! {e}.)")
    return new_name


def get_firmware_patten1(firmware, bin_name):
    bin_dir = bin_name.split(".")[0]
    if Path.is_dir(CWD / bin_dir):
        print("clean %s" % str(CWD / bin_dir))
        shutil.rmtree(CWD / bin_dir)

    print(f"Extract {bin_name} ...", flush=True)
    tar = tarfile.open(bin_name, "r")
    tar.extractall()
    tar.close()
    xclbin1 = CWD / bin_dir / "1x4.xclbin"
    xclbin2 = CWD / bin_dir / "5x4.xclbin"
    xclbin3 = CWD / bin_dir / "4x4.xclbin"
    if Path.is_file(xclbin1):
        os.environ["XLNX_VART_FIRMWARE"] = str(xclbin1)
        shutil.copy(xclbin1, CWD / "AMD_AIE2_Nx4_Overlay.xclbin")
    elif Path.is_file(xclbin2):
        os.environ["XLNX_VART_FIRMWARE"] = str(xclbin2)
        shutil.copy(xclbin2, CWD / "5x4.xclbin")
    elif Path.is_file(xclbin3):
        os.environ["XLNX_VART_FIRMWARE"] = str(xclbin3)
        shutil.copy(xclbin3, CWD / "AMD_AIE2_4x4_Overlay.xclbin")
    else:
        logging.warning(f"!!! warning : get {str(xclbin)} failed!")

    info_txt = firmware.replace(bin_name, "info.txt")
    info_txt = get_new_package(info_txt, True)


def get_firmware_patten2(rt_url, bin_name):
    voe_url = os.path.dirname(rt_url) + "/" + bin_name
    firmware = get_new_package(voe_url, True)

    bin_dir = bin_name.split(".zip")[0]
    if Path.is_dir(CWD / bin_dir):
        print("clean %s" % str(CWD / bin_dir))
        shutil.rmtree(CWD / bin_dir)

    print(f"Extract {bin_name} ...", flush=True)
    with ZipFile(bin_name, "r") as zObject:
        zObject.extractall()

    for x in os.listdir(CWD / bin_dir):
        if Path.is_file(CWD / bin_dir / x):
            shutil.copyfile(CWD / bin_dir / x, CWD / "hi" / x)
        elif Path.is_dir(CWD / bin_dir / x):
            shutil.copytree(CWD / bin_dir / x, CWD / "hi" / x)

    xclbin1 = CWD / bin_dir / "AMD_AIE2_Nx4_Overlay.xclbin"
    xclbin2 = CWD / bin_dir / "5x4.xclbin"
    if Path.is_file(xclbin1):
        shutil.copy(xclbin1, CWD / "AMD_AIE2_Nx4_Overlay.xclbin")
        os.environ["XLNX_VART_FIRMWARE"] = str(CWD / "AMD_AIE2_Nx4_Overlay.xclbin")
    if Path.is_file(xclbin2):
        shutil.copy(xclbin2, CWD / "5x4.xclbin")
        os.environ["XLNX_VART_FIRMWARE"] = str(CWD / "5x4.xclbin")

    info_name = "info_1x4.txt"
    if Path.is_file(CWD / info_name):
        os.remove(info_name)
    info_txt = os.path.dirname(rt_url) + "/" + info_name
    info_txt = get_new_package(info_txt, True)
    shutil.copyfile(CWD / info_name, CWD / "info.txt")


def get_node_and_path(new_firmware):
    host_name = "xcdl190074"
    # node = os.environ.get("NODE", "")
    # if node.startswith("xsj") or node.startswith("xco"):
    #    host_name = "xsjncuph07"
    str_list = new_firmware.split(":")
    if len(str_list) == 2:
        host_name = str_list[0]
        new_firmware = str_list[1]
    return host_name, new_firmware


def get_xclbin_cmd_list(host_name, build_id, new_firmware):
    cmd_list = [
        "ssh",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        host_name,
        "/proj/xbuilds/9999.0_plus_daily_latest/installs/lin64/Vitis/HEAD/bin/xclbinutil --force "
        + f"--dump-section AIE_PARTITION:JSON:/tmp/aie_partition_{build_id}.json -i {new_firmware}",
    ]
    return cmd_list


def check_xclbin_fingerprint(host_name, new_firmware):
    if new_firmware.endswith("xclbin"):
        build_id = os.environ.get("BUILD_ID", "999")
        cmd_list = get_xclbin_cmd_list(host_name, build_id, new_firmware)
        print(f"{(' ').join([str(each) for each in cmd_list])}", flush=True)
        try:
            subprocess.check_call(cmd_list)
        except Exception as e:
            if host_name == "xsjncuph07":
                host_name = "xcdl190074"
            else:
                host_name = "xsjncuph07"
            cmd_list = get_xclbin_cmd_list(host_name, build_id, new_firmware)
            print(f"{(' ').join([str(each) for each in cmd_list])}", flush=True)
            subprocess.check_call(cmd_list)
        json_file = f"/tmp/aie_partition_{build_id}.json"
        scp_files(host_name, json_file, ".")


def copy_control_file(host_name, file_name):
    try:
        base_name = os.path.basename(file_name)
        if os.path.exists(CWD / base_name):
            print(f"remove {str(CWD / base_name)}", flush=True)
            os.remove(CWD / base_name)
        scp_files(host_name, file_name, ".")
        os.environ["USER_CONTROL_FILE"] = str(CWD / base_name)
    except Exception as e:
        print(e)


def copy_xclbin_info(host_name, new_firmware):
    try:
        xclbin_info = new_firmware.replace(".xclbin", "_tools_version.json")
        base_name = os.path.basename(xclbin_info)
        if os.path.exists(CWD / base_name):
            print(f"remove {str(CWD / base_name)}", flush=True)
            os.remove(CWD / base_name)
        scp_files(host_name, xclbin_info, ".")
    except Exception as e:
        print(e)
        check_xclbin_fingerprint(host_name, new_firmware)


def get_firmware(download):
    firmware = os.environ.get("XLNX_VART_FIRMWARE", "")
    new_firmware = get_new_package(firmware, download)
    if new_firmware == "":
        print(
            f"Error: XLNX_VART_FIRMWARE is empty, please set a NFS path or xcoartifactory path !",
            flush=True,
        )
        return

    os.environ["XLNX_VART_FIRMWARE"] = new_firmware

    if firmware.startswith("http") and firmware.endswith("xclbin"):
        info_txt = firmware.replace(".xclbin", "_tools_version.json")
        base_name = os.path.basename(info_txt)
        if download and os.path.exists(CWD / base_name):
            print(f"remove {str(CWD / base_name)}", flush=True)
            os.remove(CWD / base_name)
        info_txt = get_new_package(info_txt, download)
    else:
        base_name = os.path.basename(new_firmware)
        if (
            Path.is_file(Path(new_firmware))
            and (
                not Path.is_file(CWD / base_name)
                or not Path(new_firmware).samefile(CWD / base_name)
            )
            and download
        ):
            shutil.copyfile(Path(new_firmware), CWD / base_name)
        else:
            host_name, new_firmware = get_node_and_path(new_firmware)
            base_name = os.path.basename(new_firmware)
            dest = str(CWD / base_name)
            if download:
                print(f"scp {new_firmware} to board ...", flush=True)
                scp_files(host_name, new_firmware, dest)
                copy_xclbin_info(host_name, new_firmware)
            os.environ["XLNX_VART_FIRMWARE"] = str(CWD / base_name)

    target_name = os.environ.get("XLNX_TARGET_NAME", "")
    if not target_name:
        target_name_sp = firmware.split("/")[-1].split(".")[0]
        os.environ["XLNX_TARGET_NAME"] = target_name_sp


def get_pdi_elf(download):
    pdi_elf = os.environ.get("PDI_ELF", "")
    if pdi_elf != "":
        if pdi_elf.startswith("http"):
            elf_zip_path = get_new_package(pdi_elf, download)
            file_name = os.path.basename(pdi_elf).replace(".zip", "")
            elf_dir_path = str(CWD / file_name)
            if download:
                if Path.is_dir(CWD / file_name):
                    print(f"Clean {file_name} ...", flush=True)
                    shutil.rmtree(CWD / file_name)
                with ZipFile(elf_zip_path, "r") as zObject:
                    zObject.extractall(elf_dir_path)
            os.environ["PDI_ELF_PATH"] = elf_dir_path
        elif Path.is_dir(Path(pdi_elf)):
            os.environ["PDI_ELF_PATH"] = pdi_elf
        else:
            host_name, pdi_elf = get_node_and_path(pdi_elf)
            base_name = os.path.basename(pdi_elf)
            dest = str(CWD / base_name)
            if download:
                if Path.is_dir(CWD / base_name):
                    print(f"Clean {base_name} ...", flush=True)
                    shutil.rmtree(CWD / base_name)
                print(f"scp {pdi_elf} to board ...", flush=True)
                scp_files(host_name, pdi_elf, dest)
            os.environ["PDI_ELF_PATH"] = str(CWD / base_name)


def get_custom_op_dll(download):
    dll = os.environ.get("CUSTOM_OP_DLL", "")
    if dll != "":
        if dll.startswith("http"):
            dll_path = get_new_package(dll, download)
            os.environ["CUSTOM_OP_DLL_PATH"] = dll_path
        else:
            base_name = os.path.basename(dll)
            if (
                Path.is_file(Path(dll))
                and (
                    not Path.is_file(CWD / base_name)
                    or not Path(dll).samefile(CWD / base_name)
                )
                and download
            ):
                shutil.copyfile(Path(dll), CWD / base_name)
            else:
                host_name, dll = get_node_and_path(dll)
                base_name = os.path.basename(dll)
                dest = str(CWD / base_name)
                if download:
                    print(f"scp {dll} to board ...", flush=True)
                    scp_files(host_name, dll, dest)
            os.environ["CUSTOM_OP_DLL_PATH"] = str(CWD / base_name)


def sed_file(file_r, file_w, sed_str):
    lines = []
    with open(file_r, "r") as r:
        lines = r.readlines()
    r.close()
    with open(file_w, "w") as w:
        for l in lines:
            if not l.startswith(sed_str):
                w.write(l)
    w.close()


def copy_test_package():
    try:
        if not IS_WINDOWS:
            return
        test_package = CWD / "fps_trace" / "test_package"
        if not Path.is_dir(test_package):
            print("test_package not exist, ignore copy.")
            return
        else:
            print("copy test_package ...")

        voe_bin = CWD / "hi" / "bin"
        for root, dirs, files in os.walk(test_package):
            relative_path = Path(root).relative_to(test_package)
            destination_dir = voe_bin / relative_path

            if not destination_dir.exists():
                destination_dir.mkdir(parents=True)

            for file in files:
                src_file = Path(root) / file
                dest_file = destination_dir / file
                shutil.copyfile(src_file, dest_file)
                print(f"Copied {src_file} to {dest_file}")
    except Exception as e:
        print(f"Error during copying test_package: {e}")


def copy_test_package_ori():
    try:
        if not IS_WINDOWS:
            return
        test_package = CWD / "fps_trace" / "test_package"
        if not Path.is_dir(test_package):
            print("test_package not exist, ignore copy.")
            return
        else:
            print("copy test_package ...")

        voe_bin = CWD / "hi" / "bin"
        for x in os.listdir(test_package):
            if not Path.is_file(test_package / x):
                continue
            # if x == "iputrace.cmd":
            #    print("delete start line from %s" % x)
            #    sed_file(str(test_package / x), str(voe_bin / x), "start")
            else:
                shutil.copyfile(test_package / x, voe_bin / x)
    except Exception as e:
        logging.warning(f"!!! warning : copy test_package failed! {e}.)")


def set_big_buffer_driver_env():
    driver_version_map = utility.get_driver_version()
    if not driver_version_map:
        return
    driver_version, driver_url = driver_version_map
    if driver_version in ("10.1.0.1", "32.0.202.206"):
        print("This should be big buffer driver, set BIG_BUFFER_DRIVER=true")
        os.environ["BIG_BUFFER_DRIVER"] = "true"


def set_env_into_os(user_key):
    if os.environ.get(user_key):
        user_env_list = os.environ.get(user_key).strip().split(" ")
        for each in user_env_list:
            each_env = each.strip().split("=")
            if len(each_env) != 2:
                print(f"environment '{each}' not valid!")
            else:
                os.environ[each_env[0].strip()] = each_env[1].strip()


def set_user_control():
    if os.environ.get("USER_CONTROL_FILE", "") != "":
        control_file = os.environ.get("USER_CONTROL_FILE", "")
        host_name, control_file = get_node_and_path(control_file)
        copy_control_file(host_name, control_file)


def read_control_file():
    run_list = []
    skip_list = []
    control_file = os.environ.get("USER_CONTROL_FILE", "")
    if control_file != "":
        base_name = os.path.basename(control_file)
        with open(CWD / base_name, "r") as f:
            tmp_list = f.readline().strip().split(",")
            if tmp_list[0] == "SKIP":
                skip_list.append(tmp_list[1])
            else:
                run_list.append(tmp_list[1])
        f.close()
    return run_list, skip_list


def set_dpm_clk():
    dpm_level = os.environ.get("DPM_LEVEL", "")
    if dpm_level == "1":
        os.environ["VAI_AIE_OVERCLOCK_MHZ"] = "1056"
    else:
        os.environ["VAI_AIE_OVERCLOCK_MHZ"] = "1810"
    print(
        "VAI_AIE_OVERCLOCK_MHZ has been set to",
        os.environ.get("VAI_AIE_OVERCLOCK_MHZ", ""),
        flush=True,
    )


def download_xclbin(args):
    get_firmware(True)


def init_test(args):
    extract_dir = "hi"
    if os.environ.get("IGNORE_ARCHIVE", "") != "true":
        info_name = "info.txt"
        if Path.is_file(CWD / info_name):
            os.remove(info_name)
        if Path.is_dir(CWD / extract_dir):
            print(f"Clean {extract_dir} ...", flush=True)
            shutil.rmtree(CWD / extract_dir)
        new_name = get_new_package(args.p, True)
        onnx_rt = "onnx-rt"
        if new_name.find(".zip") != -1:
            with ZipFile(new_name, "r") as zObject:
                print(f"Extract {args.p} to {extract_dir} ...", flush=True)
                if onnx_rt in new_name:
                    if Path.is_dir(CWD / onnx_rt):
                        shutil.rmtree(CWD / onnx_rt)
                    zObject.extractall()
                    shutil.copytree(CWD / onnx_rt, CWD / extract_dir)
                    for x in os.listdir(CWD / extract_dir / "lib"):
                        if Path.is_file(CWD / extract_dir / "lib" / x):
                            shutil.copyfile(
                                CWD / extract_dir / "lib" / x,
                                CWD / extract_dir / "bin" / x,
                            )
                else:
                    zObject.extractall(extract_dir)
        elif new_name.find(".tgz") != -1:
            tar = tarfile.open(new_name, "r")
            tar.extractall(extract_dir)
            tar.close()

        reinstall_vaitrace()
        get_firmware(True)
        get_pdi_elf(True)
        get_custom_op_dll(True)

    ipu_tools_branch = os.environ.get("IPUTRACE_TOOL_BRANCH", "24.06.11_RC4_174")
    update_repo("VitisAI-Edge", "fps_trace", False, repo_branch=ipu_tools_branch)

    if os.environ.get("FPS_TRACE_BRANCH"):
        os.chdir("fps_trace")
        checkoutByCommit(os.environ.get("FPS_TRACE_BRANCH"))
        os.chdir(CWD)

    update_repo("VitisAI", "tracer_analyze", True)
    if len(os.environ.get("OUTPUT_CHECKING", "cpu_runner,onnx_ep").split(",")) == 2:
        update_repo("yanjunz", "win24_drop", False)

    if os.environ.get("AIE_TRACE", "") == "true":
        update_repo("tianfang", "aietrace-enable", False)

    agm_files = os.path.join(CWD, "tracer_analyze", "tracer_analysis", "agm")
    agm_dir = os.path.join(CWD, "ci", "tools", "agm")
    if not Path.is_dir(CWD / "ci" / "tools" / "agm"):
        shutil.copytree(agm_files, agm_dir)

    os.chdir(CWD)
    workspace = os.environ.get("WORKSPACE", "")
    job_name = os.environ.get("JOB_NAME", "job_name")
    if "/" in job_name:
        jenkins_workspace = workspace[: workspace.find(job_name)]
    else:
        jenkins_workspace = CWD.parent
    print("Jenkins workspace %s" % jenkins_workspace)
    base_jpg = Path(jenkins_workspace) / "dog.jpg"

    test_jpg = CWD / "hi" / "bin" / "dog.jpg"
    if not Path.is_file(test_jpg) and Path.is_file(base_jpg):
        shutil.copy(base_jpg, test_jpg)

    cache_dir = get_cache_dir()
    vaip_regression_dir = utility.get_vaip_regression_dir()
    if (
        os.environ.get("ONLY_REPORT", "") != "true"
        and os.environ.get("INCREMENTAL_TEST", "") != "true"
    ):
        clean_cache(cache_dir)
        if Path.is_dir(vaip_regression_dir):
            print("Clean build dir: %s" % vaip_regression_dir, flush=True)
            shutil.rmtree(vaip_regression_dir)
    test_sample(extract_dir)


def get_vaip_config():
    print("get vaip config--->")
    try:
        vaip_config_path = os.environ.get("VAIP_CONFIG_PATH", "")
        if vaip_config_path != "":
            host_name, vaip_config_file = get_node_and_path(vaip_config_path)
            base_name = os.path.basename(vaip_config_file)
            dest = str(CWD / base_name)
            print(f"scp {vaip_config_file} to board ...", flush=True)
            scp_files(host_name, vaip_config_file, dest)
            os.environ["CI_VAIP_CONFIG"] = dest
    except Exception as e:
        print("get_vaip_config error", e)


def test_modelzoo(args):
    if args.c:
        if os.environ.get("FORBID_CASE", ""):
            forbid_models = os.environ.get("FORBID_CASE", "").split(" ")
            if args.c in forbid_models:
                print("skip forbid case ", args.c, flush=True)
                return
        if os.environ.get("INCREMENTAL_TEST", "") == "true":
            vaip_regression_dir = utility.get_vaip_regression_dir()
            if args.c in os.listdir(vaip_regression_dir):
                return

    set_env()

    set_test_env()

    copy_test_package()

    original_firmware = os.environ.get("XLNX_VART_FIRMWARE", "")
    if original_firmware:
        os.environ["ORIGINAL_FIRMWARE"] = original_firmware
        package = os.environ.get("GLOBAL_PACKAGE", "")
        if package != "" and package.endswith("onnx-rt.zip"):
            os.environ["ORIGINAL_FIRMWARE"] = (
                os.path.dirname(package) + "/voe-4.0-win_amd64.zip"
            )
    # set XLNX_VART_FIRMWARE and XLNX_TARGET_NAME and PDI_ELF
    get_firmware(False)
    get_pdi_elf(False)
    get_custom_op_dll(False)
    cache_dir = get_cache_dir()
    get_vaip_config()
    vaip_config = (
        CWD / "hi" / "bin" / os.environ.get("CI_VAIP_CONFIG", "vaip_config.json")
    )
    if not Path.is_file(vaip_config):
        print(
            f"Error: {str(vaip_config)} not exist, please check your voe package.",
            flush=True,
        )
        return
    os.environ["VITISAI_EP_JSON_CONFIG"] = str(vaip_config)

    modelzoo = os.environ.get("MODEL_ZOO", "")
    new_modelzoo = os.path.basename(get_new_package(modelzoo, True))
    if modelzoo != new_modelzoo and Path.is_file(CWD / new_modelzoo):
        shutil.copy(CWD / new_modelzoo, CWD / "ci" / new_modelzoo)
        os.environ["MODEL_ZOO"] = new_modelzoo.replace(".json", "")
    modelzoo_ref_json = modelzoo + ".ref.json"
    modelzoo_ref = CWD / "ci" / modelzoo_ref_json
    if Path.is_file(modelzoo_ref):
        os.remove(modelzoo_ref)

    env_file = get_env_file(args)
    json_result = os.environ.get("OUTPUT_JSON", env_file + ".json")
    os.environ["DUMP_RESULT"] = json_result

    set_env_into_os("USER_ENV")
    set_env_into_os("CI_ENV")
    if not os.environ.get("BIG_BUFFER_DRIVER", ""):
        set_big_buffer_driver_env()
    set_user_control()
    # set_dpm_clk()
    if os.environ.get("AIE_TRACE", "") == "true":
        try:
            aie_trace_local_dll = os.path.join(
                os.path.split(utility.get_vaip_regression_dir())[0], "aie_trace_dll"
            )
            if os.path.exists(aie_trace_local_dll):
                utility.remove_dir(aie_trace_local_dll)
            if os.environ.get("BIG_BUFFER_DRIVER", "") == "true":
                utility.scp_dir(
                    "xcdl190074.xilinx.com",
                    constant.BIG_BUFFER_AIE_TRACE_DLL,
                    aie_trace_local_dll,
                )
            else:
                utility.scp_dir(
                    "xcdl190074.xilinx.com",
                    constant.RC_AIE_TRACE_DLL,
                    aie_trace_local_dll,
                )
            os.environ["XILINX_XRT"] = aie_trace_local_dll.replace("/", "\\")
        except Exception as e:
            print(e)

    python_exe = get_python()
    cmd = [
        python_exe,
        "ci/main.py",
        "test",
    ]
    if args.c:
        cmd.extend(
            [
                args.c,
            ]
        )
    elif os.environ.get("CASE_NAME"):
        if os.environ.get("INCREMENTAL_TEST", "") == "true":
            vaip_regression_dir = utility.get_vaip_regression_dir()
            cases = os.environ.get("CASE_NAME").split(" ")
            for c in cases:
                if c not in os.listdir(vaip_regression_dir):
                    cmd.extend(c)
            if cmd[-1] == "test":
                return
        else:
            cmd.extend(os.environ.get("CASE_NAME").split(" "))
    cmd.extend(
        [
            "-j",
            os.environ.get("PARALLEL_NUM", args.j),
            "--env",
            *[f"ci/envs/{x}" for x in args.f.split(",")],
        ]
    )
    print(cmd, flush=True)
    if os.environ.get("ONLY_REPORT", "") != "true":
        subprocess.check_call(cmd)


def get_result(args):
    os.environ["PYTHONPATH"] = (
        str(CWD / "hi" / "lib" / "xir") + os.pathsep + os.environ.get("PYTHONPATH", "")
    )
    set_env_into_os("USER_ENV")
    set_env_into_os("CI_ENV")
    set_big_buffer_driver_env()
    set_user_control()
    # set_dpm_clk()
    get_firmware(False)
    get_pdi_elf(False)
    get_custom_op_dll(False)

    suite_start = time.strftime("%d-%m-%Y %H:%M:%S", time.localtime())
    env_file = get_env_file(args)
    vaip_regression_dir = utility.get_vaip_regression_dir()
    json_result = os.environ.get("OUTPUT_JSON", env_file + ".json")
    html_result = os.environ.get("OUTPUT_HTML", env_file + ".html")
    print(os.environ, flush=True)

    write_setup_script()

    baseline_benchmark_data = None
    if os.environ.get("BASELINE_BENCHMARK_DATA", ""):
        print(
            "BASELINE_BENCHMARK_DATA : %s"
            % os.environ.get("BASELINE_BENCHMARK_DATA", "")
        )
        baseline_benchmark_data = os.path.basename(
            get_new_package(os.environ["BASELINE_BENCHMARK_DATA"], True)
        )
    print(f"baseline_benchmark_data {baseline_benchmark_data}")

    collect_simple_result(json_result, html_result, baseline_benchmark_data)

    collect_modelzoo_result(args, json_result, suite_start, baseline_benchmark_data)

    ops_str = "ONNX_Ops_Coverage"
    for filename in os.listdir(CWD):
        if ops_str in filename:
            print(f"find ops file: {filename}", flush=True)
            os.remove(os.path.join(CWD, filename))
    file_path = str(
        CWD / "tracer_analyze" / "onnx-ops-coverage" / "operators_parser" / "main.py"
    )
    build_dir = utility.get_vaip_regression_dir()
    cache_dir = get_cache_dir()
    cmd = ["python", file_path, "-p", build_dir, "-cp", cache_dir]
    print(cmd, flush=True)
    try:
        subprocess.check_call(cmd)
    except Exception as e:
        logging.warning(f"!!! warning : fetch ops_offload failed! {e}.)")

    tracer_service_client_dir = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "tools", "tracer_service_client"
    )
    try:
        # if os.environ.get("NODE", "").startswith("xcd"):
        #    set_proxy()
        subprocess.check_call(["python", "-m", "pip", "install", "requests"])
        if not os.path.isdir(tracer_service_client_dir):
            subprocess.check_call(
                [
                    "git",
                    "clone",
                    "https://gitenterprise.xilinx.com/guoqiaxu/tracer_service_client.git",
                    tracer_service_client_dir,
                ]
            )
        else:
            os.chdir(tracer_service_client_dir)
            update_to_latest(False)
            os.chdir(CWD)
        import tools.tracer_service_client.tracer_service_client.tracer_info as tracer_info

        context = {}
        context["json_result"] = json_result
        context["vaip_regression_dir"] = vaip_regression_dir
        succ, rsp = tracer_info.upload_regression_data(
            CWD, os.environ.get("BUILD_URL", ""), context
        )
        print(f"regession data upload: succ:{succ} rsp:{rsp}")
    except Exception as e:
        os.chdir(CWD)
        logging.warning(f"!!! warning : upload regression data failed! {e}.)")


def write_setup_script():
    try:
        artifact = os.environ.get("BUILD_URL", "") + "artifact/"
        artifact = artifact.replace("xsjvitisaijenkins", "xsjvitisaijenkins.xilinx.com")
        xclbin = os.path.basename(
            os.environ.get("XLNX_VART_FIRMWARE", "").strip().replace('"', "")
        )
        voe = os.path.basename(
            os.environ.get(
                "GLOBAL_PACKAGE", "voe-win_amd64-with_xcompiler_on-latest_dev.zip"
            )
        )
        build_log = os.environ.get("BUILD_LOG_ZIP", "")
        pdi_elf = os.path.basename(os.environ.get("PDI_ELF", ""))
        if not pdi_elf.endswith(".zip"):
            pdi_elf = f"{pdi_elf}.zip"
        if (
            os.environ.get("ENABLE_FAST_PM", "") != "true"
            or os.environ.get("PDI_ELF", "") == ""
        ):
            pdi_elf = ""

        build_id = os.environ.get("BUILD_ID", "999")
        file_r = CWD / "ci" / "tools" / "setup_workspace.py"
        file_w = CWD / os.environ.get("SETUP_WORKSPACE_SCRIPT", "setup_workspace.py")
        lines = []
        with open(file_r, "r") as r:
            lines = r.readlines()
        r.close()
        with open(file_w, "w") as w:
            for l in lines:
                if l.find("ARTIFACT_URL") != -1:
                    w.write(l.replace("ARTIFACT_URL", artifact))
                elif l.find("VOE_NAME") != -1:
                    w.write(l.replace("VOE_NAME", voe))
                elif l.find("XCLBIN_NAME") != -1:
                    w.write(l.replace("XCLBIN_NAME", xclbin))
                elif l.find("CTL_PKTS") != -1:
                    w.write(l.replace("CTL_PKTS", pdi_elf))
                elif l.find("BUILD_LOG") != -1:
                    w.write(l.replace("BUILD_LOG", build_log))
                else:
                    w.write(l)
        w.close()
    except Exception as e:
        print(e, flush=True)


def collect_simple_result(json_result, html_result, baseline_benchmark_data=None):
    try:
        from tools import json_to_html

        print(f"baseline_benchmark_data {baseline_benchmark_data} ")
        json_to_html.main(
            json_result,
            html_result,
            utility.get_vaip_regression_dir(),
            baseline_benchmark_data,
        )
    except Exception as e:
        logging.warning(f"!!! warning : json_to_html convert failed! {e}.)")


def collect_modelzoo_result(
    args, json_result, suite_start, baseline_benchmark_data=None
):
    modelzoo_results = {}
    modelzoo_perf_results = {}
    build_id = os.environ.get("BUILD_ID", "999")
    modelzoo = os.environ.get("MODEL_ZOO", "")
    job_name = os.environ.get("JOB_BASE_NAME", "dw")
    opt_level = os.environ.get("OPT_LEVEL", "0")

    suite_end = time.strftime("%d-%m-%Y %H:%M:%S", time.localtime())
    run_date = time.strftime("%Y-%m-%d", time.localtime())
    modelzoo_json = os.path.join(CWD, "ci", modelzoo + ".json")
    log_save_dir = os.environ.get(
        "BUILD_LOG_ZIP", f"build_log_{job_name}_{build_id}"
    ).replace(".zip", "")
    try:
        if (not modelzoo) or (not os.path.exists(modelzoo_json)):
            logging.warning("No MODEL_ZOO got in env!")
            return
        with open(modelzoo_json, "r") as mz:
            modelzoo_list = json.load(mz)

        try:
            from tools import utility

            suite_run_elapsed = utility.cal_elapsed(suite_start, suite_end)
            print("Suite Run elapsed:%s" % suite_run_elapsed, flush=True)
        except Exception as e:
            logging.warning(f"!!! warning : utility from faile! {e}.)")
        from tools import compare
        from tools import compare_perf

        compare_report = os.environ.get(
            "COMPARE_HTML", str(get_env_file(args)) + "_compare_%s.html" % build_id
        )
        if os.path.exists(CWD / compare_report):
            os.remove(CWD / compare_report)
        compare_perf_report = os.environ.get(
            "PERF_HTML",
            str(get_env_file(args)) + "_compare_performance_%s.html" % build_id,
        )
        if os.path.exists(CWD / compare_perf_report):
            os.remove(CWD / compare_perf_report)
        try:
            if os.environ.get("TEST_MODE", "performance") == "mismatch":
                modelzoo_results = compare.compare(
                    utility.get_model_list(),
                    utility.get_vaip_regression_dir(),
                    compare_report,
                    modelzoo_list,
                    suite_run_elapsed,
                    run_date,
                )
                if not modelzoo_results:
                    logging.warning(f"!!! warning : no modelzoo results.)")
                    return
        except Exception as e:
            logging.warning(f"!!! warning : test model is mismatch failed! {e}.)")
        try:
            if os.environ.get("TEST_MODE", "performance").find("perf") != -1:
                print("start to make performance report")
                modelzoo_perf_results = compare_perf.compare_perf(
                    utility.get_model_list(),
                    utility.get_vaip_regression_dir(),
                    compare_perf_report,
                    modelzoo_list,
                    suite_run_elapsed,
                    baseline_benchmark_data,
                    run_date,
                )
                if not modelzoo_perf_results:
                    logging.warning(f"!!! warning : no modelzoo performance results.)")
                    return

                if os.environ.get("AGM_VISUALIZER", ""):
                    import pandas as pd

                    all_data = pd.DataFrame()
                    vaip_regression_dir = utility.get_vaip_regression_dir()
                    for model in utility.get_model_list():
                        bw_csv = f"{vaip_regression_dir}\\{model}\\bandwidth.csv"
                        if not os.path.exists(bw_csv):
                            logging.warning(f"!!! warning : no bandwidth results.)")
                        df = pd.read_csv(bw_csv)
                        column_name = "model name"
                        df[column_name] = model
                        all_data = all_data._append(df, ignore_index=True)
                    all_data.to_csv("BandWidth.csv", index=False)
                if (
                    os.environ.get("MODEL_TYPE", "onnx") == "xmodel"
                    or os.environ.get("TEST_MODE", "performance") == "vart_perf"
                ):
                    from tools import extract_silicon

                    txt_dir = CWD / "ci/model_name"
                    extract_silicon.get_silicon(modelzoo_perf_results, txt_dir)
        except Exception as e:
            print(e, flush=True)
            logging.warning(f"!!! warning : generate performance data failed! {e}.)")

        zip_cache(build_id)
        zip_build_log(
            utility.get_model_list(), log_save_dir, compare_report, compare_perf_report
        )

        try:
            if os.environ.get("TEST_MODE", "performance").find("perf") != -1:
                if os.environ.get("VAITRACE_PROFILING", ""):
                    from tools import integrate_trace_csv

                    output_csv = os.environ.get(
                        "PROFILING_EXCEL",
                        "vaitrace_profiling_%s_%s" % (build_id, run_date),
                    )
                    integrate_trace_csv.integrate(
                        utility.get_model_list(),
                        utility.get_vaip_regression_dir(),
                        get_cache_dir(),
                        output_csv,
                    )
        except Exception as e:
            tb = traceback.format_exc()
            if not tb is None:
                print(tb)
            logging.warning(f"!!! warning generate profiling excel: failed! {e}.)")
    except Exception as e:
        tb = traceback.format_exc()
        if not tb is None:
            print(tb)
        logging.warning(f"!!! warning : generate compare data failed! {e}.)")

    try:
        from tools import save2mongo

        benchmark_data_json = os.environ.get(
            "BENCHMARK_RESULT_JSON",
            f"benchmark_result_{job_name}_{build_id}_{run_date}.json",
        )
        save2mongo.save2mongo(benchmark_data_json)
    except Exception as e:
        print("ERROR: save mongo DB failed!!! %s" % e)

    modelzoo_name = os.environ.get("MODEL_ZOO", "")
    try:
        from tools import xoah_tool

        dpu_type = os.environ.get("XLNX_TARGET_NAME", "")
        if "nightly" in modelzoo_name:
            run_type = "nightly"
        elif "weekly" in modelzoo_name:
            run_type = "weekly"
        else:
            run_type = "ipu"
        run_type = run_type.upper()
        run_type = run_type.upper()
        vai_rt_branch = os.environ.get("VAI_RT_BRANCH", "dev").upper()
        suite_run_name = f"IPU_XBJ_REGRESSION_{run_type}_{vai_rt_branch}_{run_date}"
        if os.environ.get("SUITE_RUN_NAME", ""):
            suite_run_name = os.environ["SUITE_RUN_NAME"]

        m = re.match(r"(.*)_p(\d)", modelzoo_name)
        priority = ""
        if m:
            priority = m.group(2)

        target_type = (
            "STRIX"
            if not os.environ.get("TARGET_TYPE", "")
            else os.environ["TARGET_TYPE"]
        )
        rel_branch = (
            "1.0.0-dev"
            if not os.environ.get("REL_BRANCH", "")
            else os.environ["REL_BRANCH"]
        )
        xoah_paras = [
            suite_start,
            suite_end,
            suite_run_name,
            modelzoo_results,
            log_save_dir,
        ]
        if (
            os.environ.get("TEST_MODE", "performance") == "mismatch"
        ) and modelzoo_results:
            suite_name = (
                f"{dpu_type}_p{priority}_opt{opt_level}"
                if priority
                else f"{dpu_type}_opt{opt_level}"
            )
            xoah_json = os.environ.get(
                "XOAH_JSON",
                f"xoah_{target_type}_{modelzoo_name}_{dpu_type}_opt{opt_level}_{build_id}.json",
            )
            xoah_paras.append(suite_name)
            xoah_paras.append(xoah_json)
            xoah_paras.append(rel_branch)
            xoah_paras.append(target_type)
            xoah_tool.make_yoda_json(*xoah_paras)

        xoah_perf_paras = [
            suite_start,
            suite_end,
            suite_run_name,
            modelzoo_perf_results,
            log_save_dir,
        ]
        if (
            os.environ.get("TEST_MODE", "performance").find("perf") != -1
        ) and modelzoo_perf_results:
            suite_name = (
                f"{dpu_type}_p{priority}_perf_opt{opt_level}"
                if priority
                else f"{dpu_type}_perf_opt{opt_level}"
            )
            xoah_json = os.environ.get(
                "XOAH_JSON_PERF",
                f"xoah_{target_type}_{modelzoo_name}_{dpu_type}_perf_opt{opt_level}_{build_id}.json",
            )
            xoah_perf_paras.append(suite_name)
            xoah_perf_paras.append(xoah_json)
            xoah_paras.append(target_type)
            xoah_paras.append(target_type)
            xoah_tool.make_yoda_json(*xoah_perf_paras)

    except Exception as e:
        logging.warning(f"!!! warning : generate yoda data failed! {e}.)")


if __name__ == "__main__":
    main(sys.argv[1:])
