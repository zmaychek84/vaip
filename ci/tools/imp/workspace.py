#
#   The Xilinx Vitis AI Vaip in this distribution are provided under the following free
#   and permissive binary-only license, but are not provided in source code form.  While the following free
#   and permissive license is similar to the BSD open source license, it is NOT the BSD open source license
#   nor other OSI-approved open source license.
#
#    Copyright (C) 2022 Xilinx, Inc. All rights reserved.
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


import json
import logging
import os
import pathlib
import platform
import shlex
import subprocess
import sys
import traceback
from pathlib import Path

from .. import shell

HOME = Path.home()

if platform.system() == "Windows":
    SYSTEM = platform.system()
    VERSION_ID = platform.version()
elif os.path.isfile("/etc/lsb-release"):
    with open("/etc/lsb-release", encoding="utf-8") as f:
        lsb_info = {
            k: v.rstrip() for s in f.readlines() for [k, v] in [s.split("=", 2)]
        }
        SYSTEM = lsb_info["DISTRIB_ID"]
        VERSION_ID = lsb_info["DISTRIB_RELEASE"]
else:
    SYSTEM = platform.system()
    VERSION_ID = platform.version()


def _msvc_setup_envs(exists=True):
    import setuptools.msvc

    msvc_env = setuptools.msvc.EnvironmentInfo("x64")
    env = dict(
        include=msvc_env._build_paths(
            "include",
            [
                msvc_env.VCIncludes,
                msvc_env.OSIncludes,
                msvc_env.UCRTIncludes,
                msvc_env.NetFxSDKIncludes,
            ],
            exists,
        ),
        lib=msvc_env._build_paths(
            "lib",
            [
                msvc_env.VCLibraries,
                msvc_env.OSLibraries,
                msvc_env.FxTools,
                msvc_env.UCRTLibraries,
                msvc_env.NetFxSDKLibraries,
            ],
            exists,
        ),
        libpath=msvc_env._build_paths(
            "libpath",
            [
                msvc_env.VCLibraries,
                msvc_env.FxTools,
                msvc_env.VCStoreRefs,
                msvc_env.OSLibpath,
                msvc_env.MSBuild,
            ],
            exists,
        ),
        path=msvc_env._build_paths(
            "path",
            [
                msvc_env.VCTools,
                msvc_env.VSTools,
                msvc_env.VsTDb,
                msvc_env.SdkTools,
                msvc_env.SdkSetup,
                msvc_env.FxTools,
                msvc_env.MSBuild,
                msvc_env.HTMLHelpWorkshop,
                msvc_env.FSharp,
            ],
            exists,
        ),
    )
    return env


def _update_enviroment_variables(filename):
    ret = {}
    if not os.path.isfile(filename):
        logging.warn(f"cannot update environ file: {filename}")
        return ret
    with open(filename, encoding="utf-8") as f:
        ret = {
            k: v.rstrip()
            for s in f.readlines()
            if len(s) > 1 and not s.startswith("#")
            for [k, v] in [s.split("=", 2)]
        }
        return ret
    return ret


def is_crosss_compilation():
    return "OECORE_TARGET_SYSROOT" in os.environ


def is_windows():
    return platform.system() == "Windows"


# DEFAULT_MSVC_ENV = _msvc_setup_envs(exists=True) if is_windows() else {}
DEFAULT_MSVC_ENV = {}
INIT_ENVIRONMENT = {}


def initialize_windows_environment():
    if "init" in INIT_ENVIRONMENT:
        return
    os.environ.update(DEFAULT_MSVC_ENV)
    ## scp etc
    os.environ["PATH"] += ";C:\\Program Files\\Git\\usr\\bin"
    INIT_ENVIRONMENT["init"] = True


def init_environ(workspace):
    ret = os.environ.copy()
    ## setup enviroment for MSVC if any
    if is_windows():
        initialize_windows_environment()
        ret.update(DEFAULT_MSVC_ENV)
    ## setup enviroment for search path
    if is_windows():
        ret["PATH"] += os.pathsep + str(workspace.install_prefix() / "bin")
        ret["PATH"] += os.pathsep + str(workspace.install_prefix() / "xrt")
    else:
        if not "LD_LIBRARY_PATH" in ret:
            ret["LD_LIBRARY_PATH"] = ""
        ret["LD_LIBRARY_PATH"] += os.pathsep + str(workspace.install_prefix() / "lib")

    log_msg = "\t\n".join([f"{k}={v}" for (k, v) in ret.items()])
    if False:
        logging.info(f"environ:\n{log_msg}")
    return ret


def _clean_log_file(workspace):
    if os.path.exists(workspace.log_file()):
        os.remove(workspace.log_file())
    print(str(workspace.workspace()))
    os.makedirs(workspace.workspace(), exist_ok=True)
    os.makedirs(workspace.build_dir(), exist_ok=True)
    with open(workspace.log_file(), "w") as file:
        file.write("start to build " + workspace.name())


def _show_log_file(workspace):
    with open(workspace.log_file(), "r") as file:
        all_of_it = file.read()
        print(all_of_it)


def _arch():
    if is_crosss_compilation():
        return os.environ["OECORE_TARGET_ARCH"]
    else:
        return platform.machine()


def _system():
    if is_crosss_compilation():
        return os.environ["OECORE_TARGET_OS"]
    else:
        return SYSTEM


def _version_id():
    if is_crosss_compilation():
        return os.environ["OECORE_SDK_VERSION"]
    else:
        return VERSION_ID


def target_info(workspace):
    return ".".join(
        [_system(), _version_id(), _arch(), workspace.build_type().capitalize()]
    )


def install_prefix(workspace):
    if is_crosss_compilation():
        return (
            Path(os.environ["OECORE_TARGET_SYSROOT"])
            / "install"
            / workspace.build_type()
        )
    elif os.environ.get("VAI_RT_PREFIX"):
        return Path(os.environ["VAI_RT_PREFIX"])
    else:
        return workspace.home() / ".local" / target_info(workspace)


def update_environment_from_dict(workspace, var_dict):
    new_dict = dict(
        (k, v) for (k, v) in var_dict.items() if k not in workspace._environ
    )
    log_msg = "\t\n".join([f"{k}={v}" for (k, v) in new_dict.items()])
    logging.info(f"update environ with:\n{log_msg}")
    workspace._environ.update(new_dict)


def update_environment_from_file(workspace, file_list):
    for file in file_list:
        update_environment_from_dict(workspace, _update_enviroment_variables(file))


def _load_json(file):
    if not os.path.isfile(file):
        print(f"{file} not exist")
        return []
    else:
        print(file)
        return json.load(open(file))


def _get_model_zoo_name(workspace):
    return (
        workspace._environ.get("MODEL_ZOO")
        if workspace._environ.get("MODEL_ZOO")
        else "model_zoo"
    )


def load_recipe_json(workspace):
    return _load_json(
        pathlib.Path(__file__).parent.parent.parent
        / (os.environ.get("MODEL_ZOO", _get_model_zoo_name(workspace)) + ".json")
    )


def load_ref_json(workspace):
    return {
        r["id"]: r
        for r in _load_json(
            pathlib.Path(__file__).parent.parent.parent
            / (_get_model_zoo_name(workspace) + ".ref.json")
        )
    }


def save_workspace(w):
    import json

    output_file_path = pathlib.Path(__file__).parent.parent.parent / (
        _get_model_zoo_name(w) + ".ref.json"
    )
    open(output_file_path, "w").write(
        json.dumps(
            [w._ref_json[k] for k in sorted(w._ref_json.keys())],
            sort_keys=True,
            indent=4,
        )
    )

    output_file_path = pathlib.Path(__file__).parent.parent.parent / (
        _get_model_zoo_name(w) + ".ref.failure.json"
    )
    open(output_file_path, "w").write(
        json.dumps(
            [
                w._ref_json[k]
                for k in sorted(w._ref_json.keys())
                if w._ref_json[k].get("result", "FAILED") != "OK"
            ],
            sort_keys=True,
            indent=4,
        )
    )
