# -*- coding: utf-8 -*-
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

import glob
import os
import platform
from pathlib import Path

import setuptools.command.build_ext
from setuptools import Extension, find_packages, setup

TOP_DIR = os.path.realpath(os.path.dirname(__file__))
is_static_python = (
    len(glob.glob(os.path.join(TOP_DIR, "voe", "voe_cpp2py_export*.*"))) == 0
)
HOME = Path.home()

with open("README.rst") as f:
    readme = f.read()

with open("LICENSE") as f:
    license = f.read()

ext_modules = [Extension(name="voe.voe_cpp2py_export", sources=[])]
data_files = []

if platform.system() == "Windows":
    vitisai = glob.glob("../*/onnxruntime_vitisai_ep.dll")[0]
    other = []
    gemm_asr = []
    if "VAIP_WORKSPACE_PATH" in os.environ:
        other += glob.glob(str(p / "bin" / "onnxruntime.dll"))
        other += glob.glob(str(p / "bin" / "zlib.dll"))
        other += glob.glob(str(p / "bin" / "zstd.dll"))
        workspace_path = Path(os.environ["VAIP_WORKSPACE_PATH"])
        other += glob.glob(
            str(
                workspace_path
                / "vaip"
                / "vaip_pass_gemm_asr"
                / "python"
                / "_ld_preload.py"
            )
        )
        gemm_asr += glob.glob(
            str(workspace_path / "vaip" / "vaip_pass_gemm_asr" / "python" / "ort.py")
        )
        gemm_asr += glob.glob(
            str(
                workspace_path
                / "vaip"
                / "vaip_pass_gemm_asr"
                / "python"
                / "__init__.py"
            )
        )
    capi = [vitisai, *other]
    data_files.append(("lib/site-packages/onnxruntime/capi", capi))
    # data_files.append(("lib/site-packages/onnxruntime/providers/tvm", [*gemm_asr]))


class build_ext(setuptools.command.build_ext.build_ext):
    def build_extensions(self):
        if is_static_python:
            return
        src = glob.glob(os.path.join(TOP_DIR, "voe", "voe_cpp2py_export*.*"))[0]
        filename = "voe_cpp2py_export" + src.split("voe_cpp2py_export")[-1]
        dst = os.path.join(os.path.realpath(self.build_lib), "voe", filename)
        self.copy_file(src, dst)


setup(
    name="voe",
    version="1.2.0",
    description="some common util for vaip dev",
    long_description=readme,
    author="Wang Chunye",
    install_requires=["glog==0.3.1"]
    if not is_static_python
    else [],  # change vai-rt static_cpython's install_pkg as well
    extras_require={
        "tools": ["graphviz", "pandas", "xlsxwriter", "onnx==1.15.0", "tabulate"]
        if not is_static_python
        else [],
        "vaiq": ["vai-q-onnx", "olive-ai[cpu]"] if not is_static_python else [],
    },
    entry_points={"console_scripts": ["voe_vis_pass = voe.vis_pass:main"]},
    license=license,
    packages=find_packages(where=".", exclude=("tests", "docs")),
    data_files=data_files,
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
