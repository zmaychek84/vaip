##
##  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
##  Licensed under the MIT License.
##

import json
import sys

input = sys.argv

with open(input[1]) as json_file:
    data = json.load(json_file)
for i in data["subgraphs"]:
    print(i["subgrpahName"], "\t", i["status"])
