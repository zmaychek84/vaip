##
##  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
##  Licensed under the MIT License.
##

import subprocess
import sys

print(sys.argv)
output = sys.argv[1]
inputs = sys.argv[2:]
with open(output, "w") as f:
    sys.stdout = f
    print("/*")
    print(" *  Copyright (C) 2022 Xilinx, Inc. All rights reserved.")
    print(
        " *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved."
    )
    print(" *")
    print(' *  Licensed under the Apache License, Version 2.0 (the "License");')
    print(" *  you may not use this file except in compliance with the License.")
    print(" *  You may obtain a copy of the License at")
    print(" *")
    print(" *  http://www.apache.org/licenses/LICENSE-2.0")
    print(" *")
    print(" *  Unless required by applicable law or agreed to in writing, software")
    print(' *  distributed under the License is distributed on an "AS IS" BASIS,')
    print(
        " *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied."
    )
    print(" *  See the License for the specific language governing permissions and")
    print(" *  limitations under the License.")
    print(" **/")
    print("typedef void* void_ptr_t;")
    xcompiler = []
    for i in inputs:
        for sym in open(i, "r").readlines():
            if sym.startswith("xcompiler_"):
                xcompiler.append(sym)
            else:
                print(f'extern "C" void* {sym};')

    if len(xcompiler):
        print("#if WITH_XCOMPILER")
    for sym in xcompiler:
        print(f'extern "C" void* {sym};')
    if len(xcompiler):
        print("#endif")

    print("void_ptr_t reserved_symbols[] = {")

    for i in inputs:
        for sym in open(i, "r").readlines():
            if not sym.startswith("xcompiler_"):
                print(f"{sym},")

    if len(xcompiler):
        print("#if WITH_XCOMPILER")
    for sym in xcompiler:
        print(f"{sym},")
    if len(xcompiler):
        print("#endif")

    print("};")

try:
    subprocess.run(["clang-format", "-i", output])
except Exception as e:
    print(f"An error occurred: {e}", file=sys.stderr)
