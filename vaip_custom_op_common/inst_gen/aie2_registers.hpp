/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

/**
 * @file  aie2_registers.hpp
 * @brief File containing register representations of different tiles in AIE2.
 */
#ifdef __GNUC__
#  pragma GCC diagnostic ignored "-Wpedantic"
#endif
#pragma once
#include <cstdint>

namespace AIE2 {

/**
 * @enum    MTLockInfo
 * @brief   Mem Tile lock base address and offset.
 */
enum MTLockInfo : uint32_t {
  MT_LOCK_BASE_ADDR = 0xc0000,
  MT_LOCK_OFFSET = 0x10
};

/**
 * @enum    ATLockInfo
 * @brief   AIE Tile lock base address and offset.
 */
enum ATLockInfo : uint32_t {
  AT_LOCK_BASE_ADDR = 0x1f000,
  AT_LOCK_OFFSET = 0x10
};

/**
 * @enum    NTLockInfo
 * @brief   NoC Tile lock base address and offset.
 */
enum NTLockInfo : uint32_t {
  NT_LOCK_BASE_ADDR = 0x14000,
  NT_LOCK_OFFSET = 0x10
};

/**
 * @enum    ATChannelInfo
 * @brief   AIE DMA Channel and Queue base address and offset.
 */
enum ATChannelInfo : uint32_t {
  ATChCtrlBase = 0x1de00,
  ATChQueueBase = 0x1de04,
  ATNumChannels = 0x2,
  ATChIdxOffset = 0x8,
};

/**
 * @enum    MTChannelInfo
 * @brief   Mem DMA Channel and Queue base address and offset.
 */
enum MTChannelInfo : uint32_t {
  MTChCtrlBase = 0xa0600,
  MTChQueueBase = 0xa0604,
  MTNumChannels = 0x6,
  MTChIdxOffset = 0x8,
};

/**
 * @enum    ShimChannelInfo
 * @brief   NoC DMA Channel and Queue base address and offset.
 */
enum ShimChannelInfo : uint32_t {
  ShimChCtrlBase = 0x1d200,
  ShimChQueueBase = 0x1d204,
  ShimNumChannels = 0x2,
  ShimChIdxOffset = 0x8,
};

/**
 * @enum  ChannelRegInfo
 * @brief Shifts and Masks for each bit-field in DMA Channel.
 */
enum class ChannelRegInfo : uint32_t {
  RESET_LSB = 1,
  RESET_MASK = 0x00000002,
  EN_OUT_OF_ORDER_LSB = 3,
  EN_OUT_OF_ORDER_MASK = 0x00000008,
  DECOMPRESSION_ENABLE_LSB = 4,
  DECOMPRESSION_ENABLE_MASK = 0x00000010,
  CONTROLLER_ID_LSB = 8,
  CONTROLLER_ID_MASK = 0x0000FF00,
  FOT_MODE_LSB = 16,
  FOT_MODE_MASK = 0x00030000,
};

/**
 * @enum  ChannelStartQueueRegInfo
 * @brief Shifts and Masks for each bit-field in DMA Channel Start Queue.
 */
enum class ChannelStartQueueRegInfo : uint32_t {
  STARTBD_LSB = 0,
  STARTBD_MASK = 0x0000003f,
  REPEAT_LSB = 16,
  REPEAT_MASK = 0x00ff0000,
  ENTOKEN_LSB = 31,
  ENTOKEN_MASK = 0x80000000
};

/**
 * @enum  MTBdRegInfo
 * @brief Shifts and Masks for each bit-field in Mem-tile BD registers.
 */
enum class MTBdRegInfo : uint32_t {
  // BD0
  MT_ENABLE_PACKET_LSB = 31,
  MT_ENABLE_PACKET_MASK = 0x80000000,
  MT_PACKET_TYPE_LSB = 28,
  MT_PACKET_TYPE_MASK = 0x70000000,
  MT_PACKET_ID_LSB = 23,
  MT_PACKET_ID_MASK = 0x0F800000,
  MT_OUT_OF_ORDER_BD_ID_LSB = 17,
  MT_OUT_OF_ORDER_BD_ID_MASK = 0x007E0000,
  MT_BUFFER_LENGTH_LSB = 0,
  MT_BUFFER_LENGTH_MASK = 0x0001FFFF,
  // BD1
  MT_D0_ZERO_BEFORE_LSB = 26,
  MT_D0_ZERO_BEFORE_MASK = 0xFC000000,
  MT_NEXT_BD_LSB = 20,
  MT_NEXT_BD_MASK = 0x03F00000,
  MT_USE_NEXT_BD_LSB = 19,
  MT_USE_NEXT_BD_MASK = 0x00080000,
  MT_BASE_ADDRESS_LSB = 0,
  MT_BASE_ADDRESS_MASK = 0x0007FFFF,
  // BD2
  MT_TLAST_SUPPRESS_LSB = 31,
  MT_TLAST_SUPPRESS_MASK = 0x80000000,
  MT_D0_WRAP_LSB = 17,
  MT_D0_WRAP_MASK = 0x07FE0000,
  MT_D0_STEPSIZE_LSB = 0,
  MT_D0_STEPSIZE_MASK = 0x0001FFFF,
  MT_D1_ZERO_BEFORE_LSB = 27,
  MT_D1_ZERO_BEFORE_MASK = 0xF8000000,
  // BD3
  MT_D1_WRAP_LSB = 17,
  MT_D1_WRAP_MASK = 0x07FE0000,
  MT_D1_STEPSIZE_LSB = 0,
  MT_D1_STEPSIZE_MASK = 0x0001FFFF,
  // BD4
  MT_ENABLE_COMPRESSION_LSB = 31,
  MT_ENABLE_COMPRESSION_MASK = 0x80000000,
  MT_D2_ZERO_BEFORE_LSB = 27,
  MT_D2_ZERO_BEFORE_MASK = 0x78000000,
  MT_D2_WRAP_LSB = 17,
  MT_D2_WRAP_MASK = 0x07FE0000,
  MT_D2_STEPSIZE_LSB = 0,
  MT_D2_STEPSIZE_MASK = 0x0001FFFF,
  // BD5
  MT_D2_ZERO_AFTER_LSB = 28,
  MT_D2_ZERO_AFTER_MASK = 0xF0000000,
  MT_D1_ZERO_AFTER_LSB = 23,
  MT_D1_ZERO_AFTER_MASK = 0x0F800000,
  MT_D0_ZERO_AFTER_LSB = 17,
  MT_D0_ZERO_AFTER_MASK = 0x007E0000,
  MT_D3_STEPSIZE_LSB = 0,
  MT_D3_STEPSIZE_MASK = 0x0001FFFF,
  // BD6
  MT_ITERATION_CURRENT_LSB = 23,
  MT_ITERATION_CURRENT_MASK = 0x1F800000,
  MT_ITERATION_WRAP_LSB = 17,
  MT_ITERATION_WRAP_MASK = 0x007E0000,
  MT_ITERATION_STEPSIZE_LSB = 0,
  MT_ITERATION_STEPSIZE_MASK = 0x0001FFFF,
  // BD7
  MT_VALID_BD_LSB = 31,
  MT_VALID_BD_MASK = 0x80000000,
  MT_LOCK_REL_VALUE_LSB = 24,
  MT_LOCK_REL_VALUE_MASK = 0x7F000000,
  MT_LOCK_REL_ID_LSB = 16,
  MT_LOCK_REL_ID_MASK = 0x00FF0000,
  MT_LOCK_ACQ_ENABLE_LSB = 15,
  MT_LOCK_ACQ_ENABLE_MASK = 0x00008000,
  MT_LOCK_ACQ_VALUE_LSB = 8,
  MT_LOCK_ACQ_VALUE_MASK = 0x00007F00,
  MT_LOCK_ACQ_ID_LSB = 0,
  MT_LOCK_ACQ_ID_MASK = 0x000000FF
};

/**
 * @enum  ATBdRegInfo
 * @brief Shifts and Masks for each bit-field in AIE-tile BD registers.
 */
enum class ATBdRegInfo : uint32_t {
  // BD0
  AT_BASE_ADDRESS_LSB = 14,
  AT_BASE_ADDRESS_MASK = 0x0FFFC000,
  AT_BUFFER_LENGTH_LSB = 0,
  AT_BUFFER_LENGTH_MASK = 0x00003FFF,
  // BD1
  AT_ENABLE_COMPRESSION_LSB = 31,
  AT_ENABLE_COMPRESSION_MASK = 0x80000000,
  AT_ENABLE_PACKET_LSB = 30,
  AT_ENABLE_PACKET_MASK = 0x40000000,
  AT_OUT_OF_ORDER_BD_ID_LSB = 24,
  AT_OUT_OF_ORDER_BD_ID_MASK = 0x3F000000,
  AT_PACKET_ID_LSB = 19,
  AT_PACKET_ID_MASK = 0x00F80000,
  AT_PACKET_TYPE_LSB = 16,
  AT_PACKET_TYPE_MASK = 0x00070000,
  // BD2
  AT_D1_STEPSIZE_LSB = 13,
  AT_D1_STEPSIZE_MASK = 0x03FFE000,
  AT_D0_STEPSIZE_LSB = 0,
  AT_D0_STEPSIZE_MASK = 0x00001FFF,
  // BD3
  AT_D1_WRAP_LSB = 21,
  AT_D1_WRAP_MASK = 0x1FE00000,
  AT_D0_WRAP_LSB = 13,
  AT_D0_WRAP_MASK = 0x001FE000,
  AT_D2_STEPSIZE_LSB = 0,
  AT_D2_STEPSIZE_MASK = 0x00001FFF,
  // BD4
  AT_ITERATION_CURRENT_LSB = 19,
  AT_ITERATION_CURRENT_MASK = 0x01F80000,
  AT_ITERATION_WRAP_LSB = 13,
  AT_ITERATION_WRAP_MASK = 0x0007E000,
  AT_ITERATION_STEPSIZE_LSB = 0,
  AT_ITERATION_STEPSIZE_MASK = 0x00001FFF,
  // BD5
  AT_TLAST_SUPPRESS_LSB = 31,
  AT_TLAST_SUPPRESS_MASK = 0x80000000,
  AT_NEXT_BD_LSB = 27,
  AT_NEXT_BD_MASK = 0x78000000,
  AT_USE_NEXT_BD_LSB = 26,
  AT_USE_NEXT_BD_MASK = 0x04000000,
  AT_VALID_BD_LSB = 25,
  AT_VALID_BD_MASK = 0x02000000,
  AT_LOCK_REL_VALUE_LSB = 18,
  AT_LOCK_REL_VALUE_MASK = 0x01FC0000,
  AT_LOCK_REL_ID_LSB = 13,
  AT_LOCK_REL_ID_MASK = 0x0001E000,
  AT_LOCK_ACQ_ENABLE_LSB = 12,
  AT_LOCK_ACQ_ENABLE_MASK = 0x00001000,
  AT_LOCK_ACQ_VALUE_LSB = 5,
  AT_LOCK_ACQ_VALUE_MASK = 0x00000FE0,
  AT_LOCK_ACQ_ID_LSB = 0,
  AT_LOCK_ACQ_ID_MASK = 0x0000000F
};

/**
 * @enum  ShimBdRegInfo
 * @brief Shifts and Masks for each bit-field in NoC-tile BD registers.
 */
enum class ShimBdRegInfo : uint32_t {
  // BD0
  SHIM_BUFFER_LENGTH_LSB = 0,
  SHIM_BUFFER_LENGTH_MASK = 0xFFFFFFFF,
  // BD1
  SHIM_BASE_ADDRESS_LOW_LSB = 0,
  SHIM_BASE_ADDRESS_LOW_MASK = 0xFFFFFFFC,
  // BD2
  SHIM_ENABLE_PACKET_LSB = 30,
  SHIM_ENABLE_PACKET_MASK = 0x40000000,
  SHIM_OUT_OF_ORDER_BD_ID_LSB = 24,
  SHIM_OUT_OF_ORDER_BD_ID_MASK = 0x3F000000,
  SHIM_PACKET_ID_LSB = 19,
  SHIM_PACKET_ID_MASK = 0x00F80000,
  SHIM_PACKET_TYPE_LSB = 16,
  SHIM_PACKET_TYPE_MASK = 0x00070000,
  SHIM_BASE_ADDRESS_HIGH_LSB = 0,
  SHIM_BASE_ADDRESS_HIGH_MASK = 0x0000FFFF,
  // BD3
  SHIM_SECURE_ACCESS_LSB = 30,
  SHIM_SECURE_ACCESS_MASK = 0x40000000,
  SHIM_D0_WRAP_LSB = 20,
  SHIM_D0_WRAP_MASK = 0x3FF00000,
  SHIM_D0_STEPSIZE_LSB = 0,
  SHIM_D0_STEPSIZE_MASK = 0x000FFFFF,
  // BD4
  SHIM_BURST_LENGTH_LSB = 30,
  SHIM_BURST_LENGTH_MASK = 0xC0000000,
  SHIM_D1_WRAP_LSB = 20,
  SHIM_D1_WRAP_MASK = 0x3FF00000,
  SHIM_D1_STEPSIZE_LSB = 0,
  SHIM_D1_STEPSIZE_MASK = 0x000FFFFF,
  // BD5
  SHIM_SMID_LSB = 28,
  SHIM_SMID_MASK = 0xF0000000,
  SHIM_AXCACHE_LSB = 24,
  SHIM_AXCACHE_MASK = 0x0F000000,
  SHIM_AXQOS_LSB = 20,
  SHIM_AXQOS_MASK = 0x00F00000,
  SHIM_D2_STEPSIZE_LSB = 0,
  SHIM_D2_STEPSIZE_MASK = 0x000FFFFF,
  // BD6
  SHIM_ITERATION_CURRENT_LSB = 26,
  SHIM_ITERATION_CURRENT_MASK = 0xFC000000,
  SHIM_ITERATION_WRAP_LSB = 20,
  SHIM_ITERATION_WRAP_MASK = 0x03F00000,
  SHIM_ITERATION_STEPSIZE_LSB = 0,
  SHIM_ITERATION_STEPSIZE_MASK = 0x000FFFFF,
  // BD7
  SHIM_TLAST_SUPPRESS_LSB = 31,
  SHIM_TLAST_SUPPRESS_MASK = 0x80000000,
  SHIM_NEXT_BD_LSB = 27,
  SHIM_NEXT_BD_MASK = 0x78000000,
  SHIM_USE_NEXT_BD_LSB = 26,
  SHIM_USE_NEXT_BD_MASK = 0x04000000,
  SHIM_VALID_BD_LSB = 25,
  SHIM_VALID_BD_MASK = 0x02000000,
  SHIM_LOCK_REL_VALUE_LSB = 18,
  SHIM_LOCK_REL_VALUE_MASK = 0x01FC0000,
  SHIM_LOCK_REL_ID_LSB = 13,
  SHIM_LOCK_REL_ID_MASK = 0x0001E000,
  SHIM_LOCK_ACQ_ENABLE_LSB = 12,
  SHIM_LOCK_ACQ_ENABLE_MASK = 0x00001000,
  SHIM_LOCK_ACQ_VALUE_LSB = 5,
  SHIM_LOCK_ACQ_VALUE_MASK = 0x00000FE0,
  SHIM_LOCK_ACQ_ID_LSB = 0,
  SHIM_LOCK_ACQ_ID_MASK = 0x0000000F
};

/**
 * @union AieBDReg0
 * @brief AIE-tile BD register 0.
 */
union AieBDReg0 {
  std::uint32_t reg;
  struct {
    std::uint32_t buffer_length : 14;
    std::uint32_t base_addr : 14;
    std::uint32_t unused : 4;
  };
};

/**
 * @union AieBDReg1
 * @brief AIE-tile BD register 1.
 */
union AieBDReg1 {
  std::uint32_t reg;
  struct {
    std::uint32_t unused : 16;
    std::uint32_t packet_type : 3;
    std::uint32_t packet_id : 5;
    std::uint32_t out_of_order_bd_id : 6;
    std::uint32_t packet_en : 1;
    std::uint32_t compression_en : 1;
  };
};

/**
 * @union AieBDReg2
 * @brief AIE-tile BD register 2.
 */
union AieBDReg2 {
  std::uint32_t reg;
  struct {
    std::uint32_t d0_step_size : 13;
    std::uint32_t d1_step_size : 13;
    std::uint32_t unused : 6;
  };
};

/**
 * @union AieBDReg3
 * @brief AIE-tile BD register 3.
 */
union AieBDReg3 {
  std::uint32_t reg;
  struct {
    std::uint32_t d2_step_size : 13;
    std::uint32_t d0_wrap : 8;
    std::uint32_t d1_wrap : 8;
    std::uint32_t unused : 3;
  };
};

/**
 * @union AieBDReg4
 * @brief AIE-tile BD register 4.
 */
union AieBDReg4 {
  std::uint32_t reg;
  struct {
    std::uint32_t iter_step : 13;
    std::uint32_t iter_wrap : 6;
    std::uint32_t iter_curr : 6;
    std::uint32_t unused : 7;
  };
};

/**
 * @union AieBDReg5
 * @brief AIE-tile BD register 5.
 */
union AieBDReg5 {
  std::uint32_t reg;
  struct {
    std::uint32_t lock_acq_id : 4;
    std::uint32_t unused1 : 1;
    std::int32_t lock_acq_val : 7;
    std::uint32_t lock_acq_en : 1;
    std::uint32_t lock_rel_id : 4;
    std::uint32_t unused2 : 1;
    std::int32_t lock_rel_val : 7;
    std::uint32_t valid_bd : 1;
    std::uint32_t use_next_bd : 1;
    std::uint32_t next_bd : 4;
    std::uint32_t tlast_suppress : 1;
  };
};

/**
 * @union MemBDReg0
 * @brief Mem-tile BD register 0.
 */
union MemBDReg0 {
  std::uint32_t reg;
  struct {
    std::uint32_t buffer_length : 17;
    std::uint32_t out_of_order_bd_id : 6;
    std::uint32_t packet_id : 5;
    std::uint32_t packet_type : 3;
    std::uint32_t packet_en : 1;
  };
};

/**
 * @union MemBDReg1
 * @brief Mem-tile BD register 1.
 */
union MemBDReg1 {
  std::uint32_t reg;
  struct {
    std::uint32_t base_addr : 19;
    std::uint32_t use_next_bd : 1;
    std::uint32_t next_bd : 6;
    std::uint32_t d0_zero_before : 6;
  };
};

/**
 * @union MemBDReg2
 * @brief Mem-tile BD register 2.
 */
union MemBDReg2 {
  std::uint32_t reg;
  struct {
    std::uint32_t d0_step_size : 17;
    std::uint32_t d0_wrap : 10;
    std::uint32_t unused : 4;
    std::uint32_t tlast_suppress : 1;
  };
};

/**
 * @union MemBDReg3
 * @brief Mem-tile BD register 3.
 */
union MemBDReg3 {
  std::uint32_t reg;
  struct {
    std::uint32_t d1_step_size : 17;
    std::uint32_t d1_wrap : 10;
    std::uint32_t d1_zero_before : 5;
  };
};

/**
 * @union MemBDReg4
 * @brief Mem-tile BD register 4.
 */
union MemBDReg4 {
  std::uint32_t reg;
  struct {
    std::uint32_t d2_step_size : 17;
    std::uint32_t d2_wrap : 10;
    std::uint32_t d2_zero_before : 4;
    std::uint32_t compression_en : 1;
  };
};

/**
 * @union MemBDReg5
 * @brief Mem-tile BD register 5.
 */
union MemBDReg5 {
  std::uint32_t reg;
  struct {
    std::uint32_t d3_step_size : 17;
    std::uint32_t d0_zero_after : 6;
    std::uint32_t d1_zero_after : 5;
    std::uint32_t d2_zero_after : 4;
  };
};

/**
 * @union MemBDReg6
 * @brief Mem-tile BD register 6.
 */
union MemBDReg6 {
  std::uint32_t reg;
  struct {
    std::uint32_t iter_step : 17;
    std::uint32_t iter_wrap : 6;
    std::uint32_t iter_curr : 6;
    std::uint32_t unused : 3;
  };
};

/**
 * @union MemBDReg7
 * @brief Mem-tile BD register 7.
 */
union MemBDReg7 {
  std::uint32_t reg;
  struct {
    std::uint32_t lock_acq_id : 8;
    std::int32_t lock_acq_val : 7;
    std::uint32_t lock_acq_en : 1;
    std::uint32_t lock_rel_id : 8;
    std::int32_t lock_rel_val : 7;
    std::uint32_t valid_bd : 1;
  };
};

/**
 * @union NoCBDReg0
 * @brief NoC-tile BD register 0.
 */
union NoCBDReg0 {
  std::uint32_t reg;
  struct {
    std::uint32_t buffer_length : 32;
  };
};

/**
 * @union NoCBDReg1
 * @brief NoC-tile BD register 1.
 */
union NoCBDReg1 {
  std::uint32_t reg;
  struct {
    std::uint32_t reserved : 2;
    std::uint32_t base_addr_low : 30;
  };
};

/**
 * @union NoCBDReg2
 * @brief NoC-tile BD register 2.
 */
union NoCBDReg2 {
  std::uint32_t reg;
  struct {
    std::uint32_t base_addr_high : 16;
    std::uint32_t packet_type : 3;
    std::uint32_t packet_id : 5;
    std::uint32_t out_of_order_bd_id : 6;
    std::uint32_t packet_en : 1;
    std::uint32_t unused : 1;
  };
};

/**
 * @union NoCBDReg3
 * @brief NoC-tile BD register 3.
 */
union NoCBDReg3 {
  std::uint32_t reg;
  struct {
    std::uint32_t d0_step_size : 20;
    std::uint32_t d0_wrap : 10;
    std::uint32_t secure_access : 1;
    std::uint32_t unused : 1;
  };
};

/**
 * @union NoCBDReg4
 * @brief NoC-tile BD register 4.
 */
union NoCBDReg4 {
  std::uint32_t reg;
  struct {
    std::uint32_t d1_step_size : 20;
    std::uint32_t d1_wrap : 10;
    std::uint32_t burst_length : 2;
  };
};

/**
 * @union NoCBDReg5
 * @brief NoC-tile BD register 5.
 */
union NoCBDReg5 {
  std::uint32_t reg;
  struct {
    std::uint32_t d2_step_size : 20;
    std::uint32_t ax_qos : 4;
    std::uint32_t ax_cache : 4;
    std::uint32_t smid : 4;
  };
};

/**
 * @union NoCBDReg6
 * @brief NoC-tile BD register 6.
 */
union NoCBDReg6 {
  std::uint32_t reg;
  struct {
    std::uint32_t iter_step : 20;
    std::uint32_t iter_wrap : 6;
    std::uint32_t iter_curr : 6;
  };
};

/**
 * @union NoCBDReg7
 * @brief NoC-tile BD register 7.
 */
union NoCBDReg7 {
  std::uint32_t reg;
  struct {
    std::uint32_t lock_acq_id : 4;
    std::uint32_t unused1 : 1;
    std::int32_t lock_acq_val : 7;
    std::uint32_t lock_acq_en : 1;
    std::uint32_t lock_rel_id : 4;
    std::uint32_t unused2 : 1;
    std::int32_t lock_rel_val : 7;
    std::uint32_t valid_bd : 1;
    std::uint32_t use_next_bd : 1;
    std::uint32_t next_bd : 4;
    std::uint32_t tlast_suppress : 1;
  };
};

/**
 * @union AieDmaChCtrlS2MM
 * @brief AIE-tile DMA channel S2MM control.
 */
union AieDmaChCtrlS2MM {
  std::uint32_t reg;
  struct {
    std::uint32_t reserved : 1;
    std::uint32_t reset : 1;
    std::uint32_t unused1 : 1;
    std::uint32_t out_of_order_en : 1;
    std::uint32_t decompression_en : 1;
    std::uint32_t unused2 : 3;
    std::uint32_t controller_id : 8;
    std::uint32_t fot_mode : 2;
    std::uint32_t unused3 : 14;
  };
};

/**
 * @union AieDmaChQueueS2MM
 * @brief AIE-tile DMA channel S2MM start queue.
 */
union AieDmaChQueueS2MM {
  std::uint32_t reg;
  struct {
    std::uint32_t start_bd_id : 4;
    std::uint32_t unused1 : 12;
    std::uint32_t repeat : 8;
    std::uint32_t unused2 : 7;
    std::uint32_t token_issue_en : 1;
  };
};

/**
 * @union AieDmaChCtrlMM2S
 * @brief AIE-tile DMA channel MM2S control.
 */
union AieDmaChCtrlMM2S {
  std::uint32_t reg;
  struct {
    std::uint32_t reserved : 1;
    std::uint32_t reset : 1;
    std::uint32_t unused1 : 2;
    std::uint32_t compression_en : 1;
    std::uint32_t unused2 : 3;
    std::uint32_t controller_id : 8;
    std::uint32_t unused3 : 16;
  };
};

/**
 * @union AieDmaChQueueMM2S
 * @brief AIE-tile DMA channel MM2S start queue.
 */
union AieDmaChQueueMM2S {
  std::uint32_t reg;
  struct {
    std::uint32_t start_bd_id : 4;
    std::uint32_t unused1 : 12;
    std::uint32_t repeat : 8;
    std::uint32_t unused2 : 7;
    std::uint32_t token_issue_en : 1;
  };
};

/**
 * @union MemDmaChCtrlS2MM
 * @brief Mem-tile DMA channel S2MM control.
 */
union MemDmaChCtrlS2MM {
  std::uint32_t reg;
  struct {
    std::uint32_t reserved : 1;
    std::uint32_t reset : 1;
    std::uint32_t unused1 : 1;
    std::uint32_t out_of_order_en : 1;
    std::uint32_t decompression_en : 1;
    std::uint32_t unused2 : 3;
    std::uint32_t controller_id : 8;
    std::uint32_t fot_mode : 2;
    std::uint32_t unused3 : 14;
  };
};

/**
 * @union MemDmaChQueueS2MM
 * @brief Mem-tile DMA channel S2MM start queue.
 */
union MemDmaChQueueS2MM {
  std::uint32_t reg;
  struct {
    std::uint32_t start_bd_id : 6;
    std::uint32_t unused1 : 10;
    std::uint32_t repeat : 8;
    std::uint32_t unused2 : 7;
    std::uint32_t token_issue_en : 1;
  };
};

/**
 * @union MemDmaChCtrlMM2S
 * @brief Mem-tile DMA channel MM2S control.
 */
union MemDmaChCtrlMM2S {
  std::uint32_t reg;
  struct {
    std::uint32_t reserved : 1;
    std::uint32_t reset : 1;
    std::uint32_t unused1 : 2;
    std::uint32_t compression_en : 1;
    std::uint32_t unused2 : 3;
    std::uint32_t controller_id : 8;
    std::uint32_t unused3 : 16;
  };
};

/**
 * @union MemDmaChQueueMM2S
 * @brief Mem-tile DMA channel MM2S start queue.
 */
union MemDmaChQueueMM2S {
  std::uint32_t reg;
  struct {
    std::uint32_t start_bd_id : 6;
    std::uint32_t unused1 : 10;
    std::uint32_t repeat : 8;
    std::uint32_t unused2 : 7;
    std::uint32_t token_issue_en : 1;
  };
};

/**
 * @union NoCDmaChCtrlS2MM
 * @brief NoC-tile DMA channel S2MM control.
 */
union NoCDmaChCtrlS2MM {
  std::uint32_t reg;
  struct {
    std::uint32_t unused1 : 1;
    std::uint32_t pause_mem : 1;
    std::uint32_t pause_stream : 1;
    std::uint32_t out_of_order_en : 1;
    std::uint32_t unused2 : 4;
    std::uint32_t controller_id : 8;
    std::uint32_t fot_mode : 2;
    std::uint32_t unused3 : 14;
  };
};

/**
 * @union NoCDmaChQueueS2MM
 * @brief NoC-tile DMA channel S2MM start queue.
 */
union NoCDmaChQueueS2MM {
  std::uint32_t reg;
  struct {
    std::uint32_t start_bd_id : 4;
    std::uint32_t unused1 : 12;
    std::uint32_t repeat : 8;
    std::uint32_t unused2 : 7;
    std::uint32_t token_issue_en : 1;
  };
};

/**
 * @union NoCDmaChCtrlMM2S
 * @brief NoC-tile DMA channel MM2S control.
 */
union NoCDmaChCtrlMM2S {
  std::uint32_t reg;
  struct {
    std::uint32_t unused1 : 1;
    std::uint32_t pause_mem : 1;
    std::uint32_t pause_stream : 1;
    std::uint32_t unused2 : 5;
    std::uint32_t controller_id : 8;
    std::uint32_t unused3 : 16;
  };
};

/**
 * @union NoCDmaChQueueMM2S
 * @brief NoC-tile DMA channel MM2S start queue.
 */
union NoCDmaChQueueMM2S {
  std::uint32_t reg;
  struct {
    std::uint32_t start_bd_id : 4;
    std::uint32_t unused1 : 12;
    std::uint32_t repeat : 8;
    std::uint32_t unused2 : 7;
    std::uint32_t token_issue_en : 1;
  };
};

/**
 * @union LockValue
 * @brief Lock value for all tiles.
 */
union LockValue {
  std::uint32_t reg;
  struct {
    std::uint32_t value : 6;
    std::uint32_t unused : 26;
  };
};

} // namespace AIE2
