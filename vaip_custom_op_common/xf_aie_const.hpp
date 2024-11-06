/*
 *     The Xilinx Vitis AI Vaip in this distribution are provided under the
 * following free and permissive binary-only license, but are not provided in
 * source code form.  While the following free and permissive license is similar
 * to the BSD open source license, it is NOT the BSD open source license nor
 * other OSI-approved open source license.
 *
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

#ifndef _XF_AIE_CONST_H_
#define _XF_AIE_CONST_H_
#include <stdint.h>
#include <string.h>

#define RTP_ELEMENTS 64
#define RTP_SIZE (RTP_ELEMENTS * sizeof(int16_t))
#define RTP_OFFSETS_SIZE (RTP_SIZE + 32 * 3) // + 3 offset arrays of 32 bytes

enum PP_AIE_OPCODES {
  PP_AIE_FD_PRE,
  PP_AIE_SSIM_PRE,
  PP_AIE_SSIM_IND,
  PP_AIE_EGC_PRE,
  PP_AIE_FD_POST,
  PP_AIE_EGC_POST,
  PP_AIE_PIXELWISE_SELECT,
  PP_AIE_ROW_FILTER,
  PP_AIE_RESIZE_DOWN,
  PP_AIE_RESIZE_UP,
  PP_AIE_TRANSPOSE,
  PP_AIE_BLENDING,
  PP_AIE_MASK_GEN,
  PP_AIE_MASK_GEN_TRACK_PARAM_COMP,
  PP_AIE_ROW_FILTER_1CH,
  PP_AIE_MIN_MAX,
  PP_AIE_NORM,
  PP_AIE_CLAMP,
  PP_AIE_TOPK,
  PP_AIE_SOFTMAX,
};

/*  [7-0] : Operator identifier
 * [15:8] : HOST arguements (excluding instruction args)
 */
#define OPCODE_CALCULATOR(CODE, ARGC)                                          \
  (((ARGC << 8) & 0X0000FF00) + (CODE & 0X000000FF))
enum PP_OPCODES {
  PP_FD_PRE = OPCODE_CALCULATOR(PP_AIE_FD_PRE, 3),
  PP_SSIM_PRE = OPCODE_CALCULATOR(PP_AIE_SSIM_PRE, 5),
  PP_SSIM_IND = OPCODE_CALCULATOR(PP_AIE_SSIM_IND, 5),
  PP_EGC_PRE = OPCODE_CALCULATOR(PP_AIE_EGC_PRE, 3),
  PP_FD_POST = OPCODE_CALCULATOR(PP_AIE_FD_POST, 4),
  PP_EGC_POST = OPCODE_CALCULATOR(PP_AIE_EGC_POST, 4),
  PP_PIXELWISE_SELECT = OPCODE_CALCULATOR(PP_AIE_PIXELWISE_SELECT, 3),
  PP_ROW_FILTER = OPCODE_CALCULATOR(PP_AIE_ROW_FILTER, 3),
  PP_RESIZE_DOWN = OPCODE_CALCULATOR(PP_AIE_RESIZE_DOWN, 2),
  PP_RESIZE_UP = OPCODE_CALCULATOR(PP_AIE_RESIZE_UP, 3),
  PP_TRANSPOSE = OPCODE_CALCULATOR(PP_AIE_TRANSPOSE, 2),
  PP_BLENDING = OPCODE_CALCULATOR(PP_AIE_BLENDING, 4),
  PP_MASK_GEN = OPCODE_CALCULATOR(PP_AIE_MASK_GEN, 2),
  PP_MASK_GEN_TRACK_PARAM_COMP =
      OPCODE_CALCULATOR(PP_AIE_MASK_GEN_TRACK_PARAM_COMP, 3),
  PP_ROW_FILTER_1CH = OPCODE_CALCULATOR(PP_AIE_ROW_FILTER_1CH, 3),
  PP_PWS_RD = OPCODE_CALCULATOR(PP_AIE_PIXELWISE_SELECT, 3),
  PP_MIN_MAX = OPCODE_CALCULATOR(PP_AIE_MIN_MAX, 2),
  PP_RESIZEUP_MINMAX_FUSION = OPCODE_CALCULATOR(PP_AIE_RESIZE_UP, 4),
  PP_NORM = OPCODE_CALCULATOR(PP_AIE_NORM, 2),
  PP_CLAMP = OPCODE_CALCULATOR(PP_AIE_CLAMP, 2),
  PP_TOPK = OPCODE_CALCULATOR(PP_AIE_TOPK, 3),
  PP_SOFTMAX = OPCODE_CALCULATOR(PP_AIE_SOFTMAX, 2),
};

// Common RTPs accross all PP nodes
enum PP_RTPSPOS {
  PP_OPCODE = 0,
  PP_RTP_IMG_WIDTH_IN,
  PP_RTP_IMG_HEIGHT_IN,
  PP_RTP_IMG_WIDTH_OUT,
  PP_RTP_IMG_HEIGHT_OUT,
  PP_RTP_NUM_TILES_PER_CORE =
      6, // Controller RTP, index has to be even-numbered for the address to be
         // 32-bit aligned
  PP_RTP_NEXT_OPCODE_POS,
  PP_RTP_COMMON_MAX,
};

// Start from index 'PP_RTP_COMMON_MAX' to add other RTPs
enum PP_FDPRE_RTPSPOS {
  PP_FDPRE_RTP_SCALE_X_LO = PP_RTP_COMMON_MAX,
  PP_FDPRE_RTP_SCALE_X_HI,
  PP_FDPRE_RTP_SCALE_Y_LO,
  PP_FDPRE_RTP_SCALE_Y_HI,
  PP_FDPRE_RTP_FBITS_ALPHA,
  PP_FDPRE_RTP_FBITS_BETA,
  PP_FDPRE_RTP_FBITS_OUT,
  PP_FDPRE_RTP_ALPHA_0,
  PP_FDPRE_RTP_ALPHA_1,
  PP_FDPRE_RTP_ALPHA_2,
  PP_FDPRE_RTP_ALPHA_3,
  PP_FDPRE_RTP_BETA_0,
  PP_FDPRE_RTP_BETA_1,
  PP_FDPRE_RTP_BETA_2,
  PP_FDPRE_RTP_BETA_3,
  PP_FDPRE_RTP_IN_ROW_0 = 32, // Controller RTP, index has to be even-numbered
                              // for the address to be 32-bit aligned
  PP_FDPRE_RTP_OUT_ROW_0,
  PP_FDPRE_RTP_IN_ROW_1 = 34, // Controller RTP, index has to be even-numbered
                              // for the address to be 32-bit aligned
  PP_FDPRE_RTP_OUT_ROW_1,
  PP_FDPRE_RTP_IN_ROW_2 = 36, // Controller RTP, index has to be even-numbered
                              // for the address to be 32-bit aligned
  PP_FDPRE_RTP_OUT_ROW_2,
};

enum PP_SSIMPRE_RTPSPOS {
  PP_SSIMPRE_RTP_SCALE_X_LO = PP_RTP_COMMON_MAX,
  PP_SSIMPRE_RTP_SCALE_X_HI,
  PP_SSIMPRE_RTP_SCALE_Y_LO,
  PP_SSIMPRE_RTP_SCALE_Y_HI,
  PP_SSIMPRE_RTP_IN_ROW_0 = 32, // Controller RTP, index has to be even-numbered
                                // for the address to be 32-bit aligned
  PP_SSIMPRE_RTP_OUT_ROW_0,
  PP_SSIMPRE_RTP_IN_ROW_1 = 34, // Controller RTP, index has to be even-numbered
                                // for the address to be 32-bit aligned
  PP_SSIMPRE_RTP_OUT_ROW_1,
  PP_SSIMPRE_RTP_IN_ROW_2 = 36, // Controller RTP, index has to be even-numbered
                                // for the address to be 32-bit aligned
  PP_SSIMPRE_RTP_OUT_ROW_2,
};

enum PP_EGCPRE_RTPSPOS {
  PP_EGCPRE_RTP_SCALE_X_LO = PP_RTP_COMMON_MAX,
  PP_EGCPRE_RTP_SCALE_X_HI,
  PP_EGCPRE_RTP_SCALE_Y_LO,
  PP_EGCPRE_RTP_SCALE_Y_HI,
  PP_EGCPRE_RTP_FBITS_ALPHA,
  PP_EGCPRE_RTP_FBITS_BETA,
  PP_EGCPRE_RTP_FBITS_OUT,
  PP_EGCPRE_RTP_ALPHA_0,
  PP_EGCPRE_RTP_ALPHA_1,
  PP_EGCPRE_RTP_ALPHA_2,
  PP_EGCPRE_RTP_ALPHA_3,
  PP_EGCPRE_RTP_BETA_0,
  PP_EGCPRE_RTP_BETA_1,
  PP_EGCPRE_RTP_BETA_2,
  PP_EGCPRE_RTP_BETA_3,
  PP_EGCPRE_RTP_CROP_X_IN,
  PP_EGCPRE_RTP_CROP_Y_IN,
  PP_EGCPRE_RTP_CROP_WIDTH_IN,
  PP_EGCPRE_RTP_CROP_HEIGHT_IN,
  PP_EGCPRE_RTP_IN_ROW_0 = 32, // Controller RTP, index has to be even-numbered
                               // for the address to be 32-bit aligned
  PP_EGCPRE_RTP_OUT_ROW_0,
  PP_EGCPRE_RTP_IN_ROW_1 = 34, // Controller RTP, index has to be even-numbered
                               // for the address to be 32-bit aligned
  PP_EGCPRE_RTP_OUT_ROW_1,
  PP_EGCPRE_RTP_IN_ROW_2 = 36, // Controller RTP, index has to be even-numbered
                               // for the address to be 32-bit aligned
  PP_EGCPRE_RTP_OUT_ROW_2,
  PP_EGCPRE_EN_PAD
};

enum PP_EGCPOST_RTPSPOS {
  PP_EGCPOST_RTP_SCALE_X_LO = PP_RTP_COMMON_MAX,
  PP_EGCPOST_RTP_SCALE_X_HI,
  PP_EGCPOST_RTP_SCALE_Y_LO,
  PP_EGCPOST_RTP_SCALE_Y_HI,
  PP_EGCPOST_RTP_FBITS_IN,
  PP_EGCPOST_RTP_FBITS_ALPHA,
  PP_EGCPOST_RTP_FBITS_BETA,
  PP_EGCPOST_RTP_FBITS_OUT,
  PP_EGCPOST_RTP_ALPHA_0,
  PP_EGCPOST_RTP_ALPHA_1,
  PP_EGCPOST_RTP_ALPHA_2,
  PP_EGCPOST_RTP_ALPHA_3,
  PP_EGCPOST_RTP_BETA_0,
  PP_EGCPOST_RTP_BETA_1,
  PP_EGCPOST_RTP_BETA_2,
  PP_EGCPOST_RTP_BETA_3,
  PP_EGCPOST_RTP_PATCH_X_OUT,
  PP_EGCPOST_RTP_PATCH_Y_OUT,
  PP_EGCPOST_RTP_PATCH_WIDTH_OUT,
  PP_EGCPOST_RTP_PATCH_HEIGHT_OUT,
  PP_EGCPOST_RTP_IN_ROW_0 = 32, // Controller RTP, index has to be even-numbered
                                // for the address to be 32-bit aligned
  PP_EGCPOST_RTP_OUT_ROW_0,
  PP_EGCPOST_RTP_IN_ROW_1 = 34, // Controller RTP, index has to be even-numbered
                                // for the address to be 32-bit aligned
  PP_EGCPOST_RTP_OUT_ROW_1,
  PP_EGCPOST_RTP_IN_ROW_2 = 36, // Controller RTP, index has to be even-numbered
                                // for the address to be 32-bit aligned
  PP_EGCPOST_RTP_OUT_ROW_2,
};

enum PP_FDPOST_RTPSPOS {
  PP_FDPOST_RTP_SCORE_THRESH = PP_RTP_COMMON_MAX,
  PP_FDPOST_RTP_IOU_THRESH,
  PP_FDPOST_RTP_MAX_DET
};

enum PP_ROWFILTER_RTPSPOS {
  PP_ROWFILTER_RTP_PASS = PP_RTP_COMMON_MAX,
  PP_ROWFILTER_RTP_SRS_SHIFT
};

enum PP_PIXELWISE_SELECT_RTPSPOS {
  PP_PIXELWISE_SELECT_RTP_TILE_ROWS = 2 * ((PP_RTP_COMMON_MAX + 1) / 2),
  PP_PIXELWISE_SELECT_RTP_DUMMY
};

enum PP_RESIZE_DOWN_RTPSPOS {
  PP_RESIZE_DOWN_RTP_SCALE_X_LO = PP_RTP_COMMON_MAX,
  PP_RESIZE_DOWN_RTP_SCALE_X_HI,
  PP_RESIZE_DOWN_RTP_SCALE_Y_LO,
  PP_RESIZE_DOWN_RTP_SCALE_Y_HI,
  PP_RESIZE_DOWN_RTP_CHANNELS,
  PP_RESIZE_DOWN_RTP_IN_ROW_0 =
      32, // Controller RTP, index has to be even-numbered for the address to be
          // 32-bit aligned
  PP_RESIZE_DOWN_RTP_OUT_ROW_0,
  PP_RESIZE_DOWN_RTP_IN_ROW_1 =
      34, // Controller RTP, index has to be even-numbered for the address to be
          // 32-bit aligned
  PP_RESIZE_DOWN_RTP_OUT_ROW_1,
  PP_RESIZE_DOWN_RTP_IN_ROW_2 =
      36, // Controller RTP, index has to be even-numbered for the address to be
          // 32-bit aligned
  PP_RESIZE_DOWN_RTP_OUT_ROW_2,
};

enum PP_MASK_GEN_RTPSPOS {
  PP_MASK_GEN_RTP_DEPTH_MIN = PP_RTP_COMMON_MAX,
  PP_MASK_GEN_RTP_DEPTH_MAX,
  PP_MASK_GEN_RTP_TH_F_NEW,
  PP_MASK_GEN_RTP_TH_B_NEW,
  PP_MASK_GEN_RTP_PRED_SEG_THRESH,
  PP_MASK_GEN_RTP_SUM_NONZERO = 14,
};

enum PP_MIN_MAX_RTPSPOS {
  PP_MIN_MAX_RTP_TILE_WIDTH = PP_RTP_COMMON_MAX,
  PP_MIN_MAX_RTP_TILE_HEIGHT,
  PP_MIN_MAX_RTP_MINMAX_VAL = 12
};

enum PP_RESIZE_UP_RTPSPOS {
  PP_RESIZE_UP_RTP_CHANNELS = PP_RTP_COMMON_MAX,
  PP_RESIZE_UP_RTP_VEC_FACTOR,
  PP_RESIZE_UP_RTP_TILE_HEIGHT_OUT,
  PP_RESIZE_UP_RTP_SCALE_X_LO,
  PP_RESIZE_UP_RTP_SCALE_X_HI,
  PP_RESIZE_UP_RTP_SCALE_Y_LO,
  PP_RESIZE_UP_RTP_SCALE_Y_HI,
  PP_RESIZE_UP_RTP_PASS,
  PP_RESIZE_UP_RTP_IN_ROW_0 =
      32, // Controller RTP, index has to be even-numbered for the address to be
          // 32-bit aligned
  PP_RESIZE_UP_RTP_OUT_ROW_0,
  PP_RESIZE_UP_RTP_IN_ROW_1 =
      34, // Controller RTP, index has to be even-numbered for the address to be
          // 32-bit aligned
  PP_RESIZE_UP_RTP_OUT_ROW_1,
  PP_RESIZE_UP_RTP_IN_ROW_2 =
      36, // Controller RTP, index has to be even-numbered for the address to be
          // 32-bit aligned
  PP_RESIZE_UP_RTP_OUT_ROW_2,
};

// Start from index 'PP_RTP_COMMON_MAX' to add other RTPs
enum PP_NORM_RTPSPOS {
  PP_NORM_RTP_FBITS_ALPHA = PP_RTP_COMMON_MAX,
  PP_NORM_RTP_FBITS_BETA,
  PP_NORM_RTP_FBITS_OUT,
  PP_NORM_RTP_ALPHA_0,
  PP_NORM_RTP_ALPHA_1,
  PP_NORM_RTP_ALPHA_2,
  PP_NORM_RTP_ALPHA_3,
  PP_NORM_RTP_BETA_0,
  PP_NORM_RTP_BETA_1,
  PP_NORM_RTP_BETA_2,
  PP_NORM_RTP_BETA_3,
};

// Start from index 'PP_RTP_COMMON_MAX' to add other RTPs
enum PP_CLAMP_RTPSPOS {
  PP_CLAMP_RTP_LOWER_BOUND = PP_RTP_COMMON_MAX,
  PP_CLAMP_RTP_UPPER_BOUND,
};

enum PP_TOPK_RTPSPOS {
  PP_TOPK_RTP_NUM_ELEM = PP_RTP_COMMON_MAX,
  PP_TOPK_RTP_K,
  PP_TOPK_RTP_START_IDX,
};

enum PP_SOFTMAX_RTPSPOS {
  PP_SOFTMAX_RTP_IN_ELEM = PP_RTP_COMMON_MAX,
  PP_SOFTMAX_RTP_OUT_ELEM,
};

#endif
