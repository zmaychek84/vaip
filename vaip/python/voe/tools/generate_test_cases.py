##
##  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
##  Licensed under the MIT License.
##
import onnxruntime as ort
import sys


def main():
    if len(sys.argv) < 3:
        print(
            "usage: python -m voe.tools.generate_test_cases <onnx model> <json_config>"
        )
        return
    ort.InferenceSession(
        sys.argv[1],
        providers=["VitisAIExecutionProvider"],
        provider_options=[{"config_file": sys.argv[2]}],
    )


if __name__ == "__main__":
    main()
