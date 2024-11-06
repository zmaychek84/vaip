/*
 *     The Xilinx Vitis AI Vaip in this distribution are provided under the
 * following free and permissive binary-only license, but are not provided in
 * source code form.  While the following free and permissive license is similar
 * to the BSD open source license, it is NOT the BSD open source license nor
 * other OSI-approved open source license.
 *
 *      Copyright (C) 2022 Xilinx, Inc. All rights reserved.
 *      Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights
 * reserved.
 *
 *      Redistribution and use in binary form only, without modification, is
 * permitted provided that the following conditions are met:
 *
 *      1. Redistributions must reproduce the above copyright notice, this list
 * of conditions and the following disclaimer in the documentation and/or other
 * materials provided with the distribution.
 *
 *      2. The name of Xilinx, Inc. may not be used to endorse or promote
 * products redistributed with this software without specific prior written
 * permission.
 *
 *      THIS SOFTWARE IS PROVIDED BY XILINX, INC. "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL XILINX, INC. BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 *      PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE
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
