##
##  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
##  Licensed under the MIT License.
##
def quantize_static(input_model_path, output_model_path, input_shape, quantize_mode):
    if input_shape[0] == -1:
        input_shape[0] = 1

    if quantize_mode == "vaiq":
        from vai_q_onnx.quantize import quantize_static

        quantize_static(input_model_path, output_model_path, None, input_shape)
    else:
        from olive.model import ONNXModel
        from olive.passes.onnx import OnnxStaticQuantization
        from vai_q_onnx.calibrate import RandomDataReader

        dataloader = lambda a, b: RandomDataReader(input_model_path, input_shape)
        quantization = OnnxStaticQuantization(
            {
                "dataloader_func": dataloader,
                "ActivationSymmetric": True,
            },
            True,
        )
        model = ONNXModel(input_model_path, "resnet18")
        quantization.run(model, output_model_path)
