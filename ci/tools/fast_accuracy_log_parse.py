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
import re
import json
from collections import defaultdict


def extract_log_segment(input_file, output_file, start_marker, end_marker):
    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        copying = False
        for line in infile:
            if start_marker in line:
                copying = True
                continue
            if end_marker in line:
                copying = False
                continue
            if copying:
                outfile.write(line)


def parse_log(file_path):
    with open(file_path, "r") as file:
        data = file.read()

    # Regex patterns to match the required data, allowing for negative numbers
    output_pattern = re.compile(
        r"cos:\s*(-?\d+(?:\.\d+)?)\s*psnr:\s*(-?\d+(?:\.\d+)?)\s*snr:\s*(-?\d+(?:\.\d+)?)\s*l2norm:\s*(-?\d+(?:\.\d+)?)\s*File name:.*?output_(\d+)(?:[_\.].*)?"
    )

    psnr_values = defaultdict(list)
    snr_values = defaultdict(list)
    l2norm_values = defaultdict(list)

    for match in output_pattern.finditer(data):
        psnr = match.group(2)
        snr = match.group(3)
        l2norm = match.group(4)
        index = int(match.group(5))

        psnr_values[index].append(float(psnr))
        snr_values[index].append(float(snr))
        l2norm_values[index].append(float(l2norm))

    l2norm_detail = {
        str(key): list(map(str, l2norm_values[key])) for key in l2norm_values
    }

    psnr_avg = "/".join(
        [
            str(round(sum(psnr_values[i]) / len(psnr_values[i]), 4))
            for i in sorted(psnr_values.keys())
        ]
    )
    snr_avg = "/".join(
        [
            str(round(sum(snr_values[i]) / len(snr_values[i]), 4))
            for i in sorted(snr_values.keys())
        ]
    )
    l2norm_avg = "/".join(
        [
            str(round(sum(l2norm_values[i]) / len(l2norm_values[i]), 4))
            for i in sorted(l2norm_values.keys())
        ]
    )

    result = {
        "snr": snr_avg,
        "psnr": psnr_avg,
        "l2norm": l2norm_avg,
        "l2norm_detail": l2norm_detail,
    }

    return result


def main():
    input_log_path = os.path.join(os.getcwd(), "build.log")
    output_log_path = os.path.join(os.getcwd(), "group_run_log.txt")
    start_marker = "###Fast_accuracy start###"
    end_marker = "###Fast_accuracy finish###"

    extract_log_segment(input_log_path, output_log_path, start_marker, end_marker)

    print(
        f"[CI][fast_accuracy]Extracted log segment has been saved to {output_log_path}"
    )

    log_file = "group_run_log.txt"
    result = parse_log(log_file)
    with open("fast_accuracy_summary.json", "w") as outfile:
        json.dump(result, outfile, indent=4)


if __name__ == "__main__":
    main()
