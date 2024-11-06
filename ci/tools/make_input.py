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
import sys
from onnx import numpy_helper
import numpy as np
import onnx

import onnxruntime as rt
import shutil


def main():
    # run this script in vai3.5 docker image on xsj server
    # pip install onnxruntime in docker container
    modelzoo_path = "/proj/ipu_models/hug_fc/onnx_req_gpu/"
    saved_input_path = "/proj/xsjhdstaff4/yanjunz/hugging_face_models"
    model_list = [x for x in os.listdir(modelzoo_path)]
    for model_name in model_list:
        onnx_models = [
            y
            for y in os.listdir(os.path.join(modelzoo_path, model_name))
            if y.endswith(".onnx")
        ]
        if len(onnx_models) != 1:
            print("ERROR: not HAS or ONLY HAS ONE onnx model")
            continue
        onnx_model = os.path.join(modelzoo_path, model_name, onnx_models[0])

        print("make %s input" % model_name)
        session = rt.InferenceSession(
            onnx_model, providers=rt.get_available_providers()
        )
        # if len(session.get_inputs()) > 1:
        #     print("ERROR: not HAS or ONLY HAS ONE input.")
        #     continue

        for index, number in enumerate(session.get_inputs()):
            input_name = session.get_inputs()[index].name
            input_shape = session.get_inputs()[index].shape
            input_type = session.get_inputs()[index].type

            print("input %s name" % index, input_name)
            print("input %s shape" % index, input_shape)

            numpy_array = np.random.randn(*input_shape).astype(np.float32)
            tensor = numpy_helper.from_array(numpy_array)

            input_data_dir = os.path.join(
                saved_input_path, model_name, "test_data_set_0"
            )
            if not os.path.exists(input_data_dir):
                os.makedirs(input_data_dir)

            with open(
                os.path.join(input_data_dir, "input_%s.pb" % index),
                "wb",
            ) as f:
                f.write(tensor.SerializeToString())


if __name__ == "__main__":
    main()
