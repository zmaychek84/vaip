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

ipu_remote = {
    "hostname": "xcoengvm229033.xilinx.com",
    "directory": "/group/dsv_vai/models/hugging_face/timm/quantized_out/",
}


def main():
    # find /group/dsv_vai/models/hugging_face/timm/quantized_out/ -iname "*onnx" | xargs md5sum >> /tmp/ipu.md5sum
    # run in xcoengvm229033 and copy ipu.md5sum to /workspace/vaip/ci/

    with open("/workspace/vaip/ci/ipu.md5sum", "r") as file:
        lines = file.readlines()
        file_and_md5sums = [line.strip() for line in lines]

        res = [
            {
                "id": fm.split()[1].split("/")[-2],
                "onnx_model": fm.split()[1],
                "hostname": ipu_remote["hostname"],
                "md5sum": fm.split()[0],
            }
            for fm in file_and_md5sums
        ]
        with open("/workspace/vaip/ci/ipu.json", "w") as file:
            file.write(json.dumps(res, indent=4))


if __name__ == "__main__":
    main()
