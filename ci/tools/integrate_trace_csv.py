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
import re
import subprocess
import shutil
import logging
import traceback

from . import utility


def rename_csv(model_name, src_csv):
    src_dir = os.path.split(src_csv)[0]
    vaitrace_type = os.environ.get("VAITRACE_TYPE", "xat")
    dst_csv = os.path.join(src_dir, "vaitrace-%s.%s" % (model_name, vaitrace_type))
    try:
        shutil.move(str(src_csv), str(dst_csv))
        return dst_csv
    except Exception as e:
        logging.error("ERROR: %s" % e)


def integrate(model_list, log_path, cache_path, output_csv):
    model_csv_list = []
    xmodel_list = []
    vaitrace_type = os.environ.get("VAITRACE_TYPE", "xat")
    for model_name in model_list:
        try:
            model_log_path = os.path.join(log_path, model_name)
            logging.info("Integrating csv: %s " % model_name)
            csv_files = [
                x
                for x in os.listdir(str(model_log_path))
                if x.endswith(vaitrace_type)
                and x not in ("summary.csv", "vaitrace_debug.xat")
            ]
            if len(csv_files) != 1:
                logging.error("ERROR: not only one csv or xat file!")
                continue
            model_csv = csv_files[0]
            model_csv_path = os.path.join(log_path, model_name, model_csv)

            renamed_csv_path = ""
            if not os.environ.get("USER_ONNX_PATH", ""):
                renamed_csv_path = rename_csv(model_name, model_csv_path)
            model_csv_path = (
                renamed_csv_path
                if renamed_csv_path and os.path.exists(renamed_csv_path)
                else model_csv_path
            )

            model_cache_path = os.path.join(cache_path, model_name)
            compiled_xmodel = [
                x for x in os.listdir(str(model_cache_path)) if x.startswith("compiled")
            ]
            compiled_xmodel = "" if len(compiled_xmodel) != 1 else compiled_xmodel[0]
            if compiled_xmodel:
                xmodel_list.append(
                    os.path.join(cache_path, model_name, compiled_xmodel)
                )
            model_csv_list.append(model_csv_path)

        except Exception as e:
            logging.error("ERROR: %s" % e)

    workspace = os.environ.get("WORKSPACE", "")
    tracer_analysis = os.path.join(
        workspace, "tracer_analyze", "tracer_analysis", "vai_profiler"
    )
    if not os.path.exists(tracer_analysis):
        logging.error("ERROR: not found tracer_analysis scripts")
        return
    try:
        requirements_file = os.path.join(tracer_analysis, "requirements.txt")
        if os.path.join(requirements_file):
            utility.pip_install_requirements(requirements_file)
        sys.path.append(tracer_analysis)
        import run

        if model_csv_list:
            logging.info("Tracer analysis: %s" % model_csv_list)
            if len(model_csv_list) == len(xmodel_list):
                run.main(
                    model_csv_list,
                    output_csv,
                    summary=True,
                    ext=vaitrace_type,
                    xmodels=xmodel_list,
                )
            else:
                run.main(model_csv_list, output_csv, summary=True, ext=vaitrace_type)
    except Exception as e:
        tb = traceback.format_exc()
        if not tb is None:
            print(tb)
        logging.error("ERROR: %s" % e)


if __name__ == "__main__":
    pass
