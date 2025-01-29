/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */
/**
 * @file  aie2_ipu_isa.hpp
 * @brief File containing ISA Op-codes for IPU.
 */
#pragma once

#include <cstdint>

/**
 * @enum  InstOpCode
 * @brief ISA Opcodes for IPU
 */
enum class InstOpCode : uint32_t {
  NOOP = 0,
  WRITEBD = 1,
  WRITE32 = 2,
  SYNC = 3,
  WRITEBD_EXTEND_AIETILE = 4,
  WRITE32_EXTEND_GENERAL = 5,
  WRITEBD_EXTEND_SHIMTILE = 6,
  WRITEBD_EXTEND_MEMTILE = 7,
  WRITE32_EXTEND_DIFFBD = 8,
  WRITEBD_EXTEND_SAMEBD_MEMTILE = 9,
  DUMPDDR = 10,
  WRITESHIMBD = 11,
  WRITEMEMBD = 12,
  WRITE32_RTP = 13,
  READ32_CMP = 14,
  READ32_POLL = 15
};

/**
 * @brief Returns encoded opcode word.
 *
 * The Op-codes are encoded as the 8 Most significant bits in a 32b word.
 */
template <InstOpCode otype> constexpr uint32_t genOpCode(void) {
  return static_cast<uint32_t>(otype) << 24;
}
