##
##  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
##  Licensed under the MIT License.
##
import voe.anchor_point as ap

print(ap.NCHW2NHWC)

ret = ap.transpose([0, 3, 1, 2])

print(ret)

print(ap.is_anchor_point(ret))
