/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */
#pragma once
#include "vaip/vaip.hpp"
#include "vitis/ai/env_config.hpp"

using namespace vaip_core;
namespace vaip::dd {
enum DD_OPS {
  QMHAGRPB,
  DQAdd,
  QMatMul,
  QMatMulAdd,
  QMatMulAddGelu,
  QLayerNorm,
  QEltWiseAdd,
  QGlobalAvgPool,
  QMHA,
  QMHACHANNEL,
  QMHAWINDOW,
  IConv,
  QReshapeTranspose,
  QConv,
  QConcateOPs,
  QLstm,
  QGroupNorm,
  QConv2MatMul,
  QELWEMUL_qdq,
  QSlice,
  QConcat,
  Mladfelwmul,
  MLADFMATMULA16A16,
  Mladfsoftmax
};
}