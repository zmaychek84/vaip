#
#  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
#  Licensed under the MIT License.
#

set -e
for i in  \
             ./vaip/custom_op.hpp \
             ./vaip/dll_safe.hpp \
             ./vaip/export.hpp \
             ./vaip/my_ort.hpp \
             ./vaip/vaip_gsl.hpp \
             ./vaip/vaip_ort_api.hpp; do
    cp -v include/$i /workspace/onnxruntime/onnxruntime/core/providers/vitisai/include/vaip;
done

cp /workspace/vaip/onnxruntime_vitisai_ep/include/onnxruntime_vitisai_ep/onnxruntime_vitisai_ep.hpp /workspace/onnxruntime/onnxruntime/core/providers/vitisai/include/onnxruntime_vitisai_ep/onnxruntime_vitisai_ep.hpp
