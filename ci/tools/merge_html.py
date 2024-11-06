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


def parse_file(file_name, ignore_str):
    all_lines = ""
    f = open(file_name, "r")
    for each_line in f:
        if each_line.strip() != ignore_str:
            all_lines += each_line

    f.close()
    return all_lines


if __name__ == "__main__":
    all_lines = ""
    for i in range(1, len(sys.argv)):
        file_name = sys.argv[i]
        if i == 1:
            all_lines = parse_file(file_name, "</html>")
        else:
            all_lines += parse_file(file_name, "<html>")

    output_file = os.environ.get("MERGED_HTML", "total.html")
    print(f"Total html: {output_file}")
    f = open(output_file, "w")
    f.write(all_lines)
    f.close()
