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
import logging
import json
import sys
import time


def make_yoda_json(
    suite_start_time,
    suite_end_time,
    suite_run_name,
    modelzoo_results,
    log_save_dir,
    suite_name,
    xoah_json,
    rel_branch="1.0.0-dev",
    suite_category1="STRIX",
):
    yoda_suite_data = {
        "SUPER_SUITE_NAME": "IPU_TEST_PIPELINE",  # IPU_TEST_PIPELINE  VAI_REGRESSIONS
        "SUITE_NAME": suite_name,  # 1x4
        "SUITE_RUN_NAME": suite_run_name,
        "REL_BRANCH": rel_branch,  # 1.0.0
        "SUITE_CATEGORY1": suite_category1,
        "SUITE_CATEGORY2": "ONNX",
        "USER": "huizhang,yanjunz",
        "SITE": "xbj",
        "SUITE_START_TIME": suite_start_time,
        "SUITE_END_TIME": suite_end_time,
    }

    xoah_results = []
    log_base = "/wrk/fg1/buildsD/vitis-ai/vitis-ai-edge-XOAH/Log"
    # log_save_dir = 'build_log_dw131'
    for model_name, results in modelzoo_results.items():
        if not isinstance(results, dict):
            continue
        fake_time = time.strftime("%d-%m-%Y %H:%M:%S", time.localtime())
        output_file = f"{log_base}/{log_save_dir}/{model_name}/build.log"
        test_work_dir = f"{log_base}/{log_save_dir}/{model_name}"
        test_result = {
            "TEST_CASE_NAME": model_name,
            "TEST_NAME": model_name,
            "TEST_PLATFORM": suite_category1,
            "TEST_OWNER": "huizhang,yanjunz",
            "TEST_START_TIME": results.get("TEST_START_TIME", fake_time),
            "TEST_END_TIME": results.get("TEST_END_TIME", fake_time),
            "TEST_STATUS": results.get("TEST_STATUS", "FAIL"),
            "EXIT_CODE": results.get("EXIT_CODE", "1"),
            "JOB_HOST": "xbj",
            "TEST_FAMILY": "classification",
            "JOB_OUTPUT_FILE": output_file,
            "TEST_WORK_DIR": test_work_dir,
        }
        if "performance" in results.get("Result", {}).keys():
            test_result[
                "TEST_PATH"
            ] = f"{log_base}/{log_save_dir}/compare_perf_report.html"
        else:
            test_result["TEST_PATH"] = f"{log_base}/{log_save_dir}/compare_report.html"
        if results.get("TEST_UNIQUE_ERROR", ""):
            test_result["TEST_UNIQUE_ERROR"] = results["TEST_UNIQUE_ERROR"]
        if results.get("TEST_FIRST_ERROR_STRING", ""):
            test_result["TEST_FIRST_ERROR_STRING"] = results["TEST_FIRST_ERROR_STRING"]
        if results.get("TEST_ELAPSED", ""):
            test_result["TEST_ELAPSED"] = results["TEST_ELAPSED"]
        xoah_results.append(test_result)

    yoda_suite_data["PLATFORMS"] = {"WIN64": xoah_results}

    with open(xoah_json, "w") as f:
        json.dump(yoda_suite_data, f, indent=4)


if __name__ == "__main__":
    pass
