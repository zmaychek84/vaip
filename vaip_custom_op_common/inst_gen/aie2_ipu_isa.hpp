/*
 *     The Xilinx Vitis AI Vaip in this distribution are provided under the
 * following free and permissive binary-only license, but are not provided in
 * source code form.  While the following free and permissive license is similar
 * to the BSD open source license, it is NOT the BSD open source license nor
 * other OSI-approved open source license.
 *
 *      Copyright (C) 2022 Xilinx, Inc.
 *      Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc.
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
