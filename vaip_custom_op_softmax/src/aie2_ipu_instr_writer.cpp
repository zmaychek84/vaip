/*
 *     The Xilinx Vitis AI Vaip in this distribution are provided under the
 * following free and permissive binary-only license, but are not provided in
 * source code form.  While the following free and permissive license is similar
 * to the BSD open source license, it is NOT the BSD open source license nor
 * other OSI-approved open source license.
 *
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
#include "aie2_ipu_instr_writer.hpp"
#include "aie2_dma_types.hpp"

// DMA BD writers
int IPUInstructionWriter::write(AIE2::AieBD* elem, uint32_t* instrs) {
  // Get Opcode
  constexpr uint32_t opcode = genOpCode<InstOpCode::WRITEBD_EXTEND_AIETILE>();
  // Generate word
  uint32_t word = (opcode) | (elem->m_location.m_col << 16) |
                  (elem->m_location.m_row << 8) | (elem->m_id);
  // Number of cols/rows
  uint32_t n_col_row = (elem->m_ncols << 24) | (elem->m_nrows << 16);

  // Write to instr buffer
  instrs[0] = word;
  instrs[1] = n_col_row;
  instrs[2] = elem->m_0.reg;
  instrs[3] = elem->m_1.reg;
  instrs[4] = elem->m_2.reg;
  instrs[5] = elem->m_3.reg;
  instrs[6] = elem->m_4.reg;
  instrs[7] = elem->m_5.reg;

  // Return number of instructions written
  return 8;
}

int IPUInstructionWriter::write(AIE2::MemBD* elem, uint32_t* instrs) {
  // Get Opcode
  constexpr uint32_t opcode = genOpCode<InstOpCode::WRITEBD_EXTEND_MEMTILE>();
  // Generate word
  uint32_t word = (opcode) | (elem->m_location.m_col << 16) |
                  (elem->m_location.m_row << 8) | (elem->m_ncols);
  uint32_t startBD = elem->m_id;
  uint32_t nextBD = elem->m_1.next_bd;
  uint32_t* start_bd = &startBD;
  uint32_t* next_bd = &nextBD;
  uint32_t bd_list = 0, next_bd_list = 0;
  for (uint8_t c = 0u; c < elem->m_ncols; ++c) {
    uint32_t tmp = start_bd[c];
    tmp <<= (c * 8);
    // update bd_list
    bd_list = (bd_list | tmp);
    tmp = next_bd[c];
    tmp <<= (c * 8);
    // update next_bd_list
    next_bd_list = (next_bd_list | tmp);
  }

  // Write to instr buffer
  instrs[0] = word;
  instrs[1] = bd_list;
  instrs[2] = next_bd_list;
  instrs[3] = elem->m_0.reg;
  instrs[4] = elem->m_1.reg;
  instrs[5] = elem->m_2.reg;
  instrs[6] = elem->m_3.reg;
  instrs[7] = elem->m_4.reg;
  instrs[8] = elem->m_5.reg;
  instrs[9] = elem->m_6.reg;
  instrs[10] = elem->m_7.reg;

  // Return number of instructions written
  return 11;
}

int IPUInstructionWriter::write(AIE2::NoCBD* elem, uint32_t* instrs) {
  // Get Opcode
  constexpr uint32_t opcode = genOpCode<InstOpCode::WRITEBD_EXTEND_SHIMTILE>();
  // Generate word
  uint32_t word = (opcode) | (elem->m_location.m_col << 16) |
                  (elem->m_ncols << 8) | (elem->m_ddrtype << 4) | (elem->m_id);
  // Address increment
  uint32_t addr_incr = 0;
  // Write to instr buffer
  instrs[0] = word;
  instrs[1] = addr_incr;
  instrs[2] = elem->m_0.reg;
  instrs[3] = elem->m_1.reg;
  instrs[4] = elem->m_2.reg;
  instrs[5] = elem->m_3.reg;
  instrs[6] = elem->m_4.reg;
  instrs[7] = elem->m_5.reg;
  instrs[8] = elem->m_6.reg;
  instrs[9] = elem->m_7.reg;

  // Return number of instructions written
  return 10;
}

// DMA Channel writers
//// Control
int IPUInstructionWriter::write(AIE2::AieDmaS2MMCtrl* elem, uint32_t* instrs) {
  // Get Opcode
  constexpr uint32_t opcode = genOpCode<InstOpCode::WRITE32>();
  uint32_t word =
      opcode | (elem->m_location.m_col << 16) | (elem->m_location.m_row << 8);
  uint32_t addr = elem->m_base + elem->m_offset * elem->m_id;
  uint32_t data = elem->m_regval.reg;

  // Write to instr buffer
  instrs[0] = word;
  instrs[1] = addr;
  instrs[2] = data;

  // Return number of instructions written
  return 3;
}

int IPUInstructionWriter::write(AIE2::AieDmaMM2SCtrl* elem, uint32_t* instrs) {
  // Get Opcode
  constexpr uint32_t opcode = genOpCode<InstOpCode::WRITE32>();
  uint32_t word =
      opcode | (elem->m_location.m_col << 16) | (elem->m_location.m_row << 8);
  uint32_t addr = elem->m_base + elem->m_offset * elem->m_id;
  uint32_t data = elem->m_regval.reg;

  // Write to instr buffer
  instrs[0] = word;
  instrs[1] = addr;
  instrs[2] = data;

  // Return number of instructions written
  return 3;
}

int IPUInstructionWriter::write(AIE2::MemDmaS2MMCtrl* elem, uint32_t* instrs) {
  // Get Opcode
  constexpr uint32_t opcode = genOpCode<InstOpCode::WRITE32>();
  uint32_t word =
      opcode | (elem->m_location.m_col << 16) | (elem->m_location.m_row << 8);
  uint32_t addr = elem->m_base + elem->m_offset * elem->m_id;
  uint32_t data = elem->m_regval.reg;

  // Write to instr buffer
  instrs[0] = word;
  instrs[1] = addr;
  instrs[2] = data;

  // Return number of instructions written
  return 3;
}

int IPUInstructionWriter::write(AIE2::MemDmaMM2SCtrl* elem, uint32_t* instrs) {
  // Get Opcode
  constexpr uint32_t opcode = genOpCode<InstOpCode::WRITE32>();
  uint32_t word =
      opcode | (elem->m_location.m_col << 16) | (elem->m_location.m_row << 8);
  uint32_t addr = elem->m_base + elem->m_offset * elem->m_id;
  uint32_t data = elem->m_regval.reg;

  // Write to instr buffer
  instrs[0] = word;
  instrs[1] = addr;
  instrs[2] = data;

  // Return number of instructions written
  return 3;
}

int IPUInstructionWriter::write(AIE2::NoCDmaS2MMCtrl* elem, uint32_t* instrs) {
  // Get Opcode
  constexpr uint32_t opcode = genOpCode<InstOpCode::WRITE32>();
  uint32_t word =
      opcode | (elem->m_location.m_col << 16) | (elem->m_location.m_row << 8);
  uint32_t addr = elem->m_base + elem->m_offset * elem->m_id;
  uint32_t data = elem->m_regval.reg;

  // Write to instr buffer
  instrs[0] = word;
  instrs[1] = addr;
  instrs[2] = data;

  // Return number of instructions written
  return 3;
}

int IPUInstructionWriter::write(AIE2::NoCDmaMM2SCtrl* elem, uint32_t* instrs) {
  // Get Opcode
  constexpr uint32_t opcode = genOpCode<InstOpCode::WRITE32>();
  uint32_t word =
      opcode | (elem->m_location.m_col << 16) | (elem->m_location.m_row << 8);
  uint32_t addr = elem->m_base + elem->m_offset * elem->m_id;
  uint32_t data = elem->m_regval.reg;

  // Write to instr buffer
  instrs[0] = word;
  instrs[1] = addr;
  instrs[2] = data;

  // Return number of instructions written
  return 3;
}

//// Queue
int IPUInstructionWriter::write(AIE2::AieDmaS2MMQueue* elem, uint32_t* instrs) {
  // Get Opcode
  constexpr uint32_t opcode = genOpCode<InstOpCode::WRITE32>();
  uint32_t word =
      opcode | (elem->m_location.m_col << 16) | (elem->m_location.m_row << 8);
  uint32_t addr = elem->m_base + elem->m_offset * elem->m_id;
  uint32_t data = elem->m_regval.reg;

  // Write to instr buffer
  instrs[0] = word;
  instrs[1] = addr;
  instrs[2] = data;

  // Return number of instructions written
  return 3;
}

int IPUInstructionWriter::write(AIE2::AieDmaMM2SQueue* elem, uint32_t* instrs) {
  // Get Opcode
  constexpr uint32_t opcode = genOpCode<InstOpCode::WRITE32>();
  uint32_t word =
      opcode | (elem->m_location.m_col << 16) | (elem->m_location.m_row << 8);
  uint32_t addr = elem->m_base + elem->m_offset * elem->m_id;
  uint32_t data = elem->m_regval.reg;

  // Write to instr buffer
  instrs[0] = word;
  instrs[1] = addr;
  instrs[2] = data;

  // Return number of instructions written
  return 3;
}

int IPUInstructionWriter::write(AIE2::MemDmaS2MMQueue* elem, uint32_t* instrs) {
  // Get Opcode
  constexpr uint32_t opcode = genOpCode<InstOpCode::WRITE32>();
  uint32_t word =
      opcode | (elem->m_location.m_col << 16) | (elem->m_location.m_row << 8);
  uint32_t addr = elem->m_base + elem->m_offset * elem->m_id;
  uint32_t data = elem->m_regval.reg;

  // Write to instr buffer
  instrs[0] = word;
  instrs[1] = addr;
  instrs[2] = data;

  // Return number of instructions written
  return 3;
}

int IPUInstructionWriter::write(AIE2::MemDmaMM2SQueue* elem, uint32_t* instrs) {
  // Get Opcode
  constexpr uint32_t opcode = genOpCode<InstOpCode::WRITE32>();
  uint32_t word =
      opcode | (elem->m_location.m_col << 16) | (elem->m_location.m_row << 8);
  uint32_t addr = elem->m_base + elem->m_offset * elem->m_id;
  uint32_t data = elem->m_regval.reg;

  // Write to instr buffer
  instrs[0] = word;
  instrs[1] = addr;
  instrs[2] = data;

  // Return number of instructions written
  return 3;
}

int IPUInstructionWriter::write(AIE2::NoCDmaS2MMQueue* elem, uint32_t* instrs) {
  // Get Opcode
  constexpr uint32_t opcode = genOpCode<InstOpCode::WRITE32>();
  uint32_t word =
      opcode | (elem->m_location.m_col << 16) | (elem->m_location.m_row << 8);
  uint32_t addr = elem->m_base + elem->m_offset * elem->m_id;
  uint32_t data = elem->m_regval.reg;

  // Write to instr buffer
  instrs[0] = word;
  instrs[1] = addr;
  instrs[2] = data;

  // Return number of instructions written
  return 3;
}

int IPUInstructionWriter::write(AIE2::NoCDmaMM2SQueue* elem, uint32_t* instrs) {
  // Get Opcode
  constexpr uint32_t opcode = genOpCode<InstOpCode::WRITE32>();
  uint32_t word =
      opcode | (elem->m_location.m_col << 16) | (elem->m_location.m_row << 8);
  uint32_t addr = elem->m_base + elem->m_offset * elem->m_id;
  uint32_t data = elem->m_regval.reg;

  // Write to instr buffer
  instrs[0] = word;
  instrs[1] = addr;
  instrs[2] = data;

  // Return number of instructions written
  return 3;
}

// Lock Writers
int IPUInstructionWriter::write(AIE2::ATLock* elem, uint32_t* instrs) {
  // Get Opcode
  constexpr uint32_t opcode = genOpCode<InstOpCode::WRITE32>();
  uint32_t word =
      opcode | (elem->m_location.m_col << 16) | (elem->m_location.m_row << 8);
  uint32_t addr = elem->m_base + elem->m_offset * elem->m_id;
  uint32_t data = elem->m_value.reg;

  // Write to instr buffer
  instrs[0] = word;
  instrs[1] = addr;
  instrs[2] = data;

  // Return number of instructions written
  return 3;
}

int IPUInstructionWriter::write(AIE2::MTLock* elem, uint32_t* instrs) {
  // Get Opcode
  constexpr uint32_t opcode = genOpCode<InstOpCode::WRITE32>();
  uint32_t word =
      opcode | (elem->m_location.m_col << 16) | (elem->m_location.m_row << 8);
  uint32_t addr = elem->m_base + elem->m_offset * elem->m_id;
  uint32_t data = elem->m_value.reg;

  // Write to instr buffer
  instrs[0] = word;
  instrs[1] = addr;
  instrs[2] = data;

  // Return number of instructions written
  return 3;
}

int IPUInstructionWriter::write(AIE2::NTLock* elem, uint32_t* instrs) {
  // Get Opcode
  constexpr uint32_t opcode = genOpCode<InstOpCode::WRITE32>();
  uint32_t word =
      opcode | (elem->m_location.m_col << 16) | (elem->m_location.m_row << 8);
  uint32_t addr = elem->m_base + elem->m_offset * elem->m_id;
  uint32_t data = elem->m_value.reg;

  // Write to instr buffer
  instrs[0] = word;
  instrs[1] = addr;
  instrs[2] = data;

  // Return number of instructions written
  return 3;
}

// Word writers
int IPUInstructionWriter::write(AIE2::Word* elem, uint32_t* instrs) {
  // Get Opcode
  constexpr uint32_t opcode = genOpCode<InstOpCode::WRITE32>();
  uint32_t word =
      opcode | (elem->m_location.m_col << 16) | (elem->m_location.m_row << 8);

  // Write to instr buffer
  instrs[0] = word;
  instrs[1] = elem->m_addr;
  instrs[2] = elem->m_data;

  // Return number of instructions written
  return 3;
}

int IPUInstructionWriter::write(AIE2::IndexWord* elem, uint32_t* instrs) {
  // Get Opcode
  constexpr uint32_t opcode = genOpCode<InstOpCode::WRITE32_RTP>();
  uint32_t word =
      opcode | (elem->m_location.m_col << 16) | (elem->m_location.m_row << 8);

  // Write to instr buffer
  instrs[0] = word;
  instrs[1] = elem->m_addr;
  instrs[2] = elem->m_data;

  // Return number of instructions written
  return 3;
}

int IPUInstructionWriter::write(AIE2::SyncWord* elem, uint32_t* instrs) {
  // Get Opcode
  constexpr uint32_t opcode = genOpCode<InstOpCode::SYNC>();
  // Generate word
  uint8_t direction = static_cast<uint8_t>(elem->m_direction);
  uint32_t word = (opcode) | (elem->m_location.m_col << 16) |
                  (elem->m_location.m_row << 8) | (direction);
  uint32_t data =
      (elem->m_channel << 24) | (elem->m_ncols << 16) | (elem->m_nrows << 8);

  // Write to instr buffer
  instrs[0] = word;
  instrs[1] = data;

  // Return number of instructions written
  return 2;
}
