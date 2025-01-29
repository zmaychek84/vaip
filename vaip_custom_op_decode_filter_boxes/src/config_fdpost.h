/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */
#ifndef __CONFIG_FDPOST_H_
#define __CONFIG_FDPOST_H_

#include <cmath>

#define TOTAL_ELEMENTS 1351
#define OUT_ELEMENTS 10

#define INPUT_SCORE_THRESHOLD 0.5
#define F_BITS 7

#define SCORE_THRESH ((int)(INPUT_SCORE_THRESHOLD * std::pow(2, F_BITS)))

#define IOU_THRESH 2457
#define MAX_DET 10
#define HEIGHT 311 // 431
#define WIDTH 450

#define DATA_TYPE_IN uint8_t
#define DATA_TYPE_OUT uint16_t

#define IN_BOX_ELEM (TOTAL_ELEMENTS * 12)
#define IN_ANCHOR_ELEM (TOTAL_ELEMENTS * 4)
#define IN_SCORE_ELEM (TOTAL_ELEMENTS * 4)
#define OUT_ELEM (OUT_ELEMENTS * 4 + 2)

// 32 byte alignment for aligned loads in kernel
#define IN_BOX_SIZE IN_BOX_ELEM
#define IN_ANCHOR_SIZE IN_ANCHOR_ELEM
#define IN_SCORE_SIZE IN_SCORE_ELEM
#define OUT_SIZE (OUT_ELEM * sizeof(DATA_TYPE_OUT))

#endif //__CONFIG_FDPOST_H_
