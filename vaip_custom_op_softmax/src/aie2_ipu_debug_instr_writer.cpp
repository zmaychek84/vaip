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
#include "aie2_ipu_debug_instr_writer.hpp"
#include "aie2_dma_types.hpp"

// DMA BD writers
int IPUDebugInstrWriter::write(AIE2::AieBD* elem, uint32_t* instrs) {
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

  // std::cout << "-- Element: " << elem->name() << " Index: " <<
  // (elem->getUniqueId() & 0x0000FFFF) << "\n";

  m_ir_stream << std::setw(sizeof(uint32_t) * 2) << std::setfill('0')
              << std::hex << word << ": opcode "
              << static_cast<uint32_t>(InstOpCode::WRITEBD_EXTEND_AIETILE)
              << " Col " << elem->m_location.m_col << " Row "
              << elem->m_location.m_row << " BdId " << (elem->m_id) << "\n";
  m_ir_stream << std::setw(sizeof(uint32_t) * 2) << std::setfill('0')
              << std::hex << n_col_row << ": NCols " << elem->m_ncols
              << " NRows " << elem->m_nrows << "\n";
  m_ir_stream << std::setw(sizeof(uint32_t) * 2) << std::setfill('0')
              << std::hex << elem->m_0.reg << ": BufferLength "
              << elem->m_0.buffer_length << " BaseAddr " << elem->m_0.base_addr
              << "\n";
  m_ir_stream << std::setw(sizeof(uint32_t) * 2) << std::setfill('0')
              << std::hex << elem->m_1.reg << ": PacketType "
              << elem->m_1.packet_type << " PacketID " << elem->m_1.packet_id
              << " EnablePacket " << elem->m_1.packet_en << " OOOBd "
              << elem->m_1.out_of_order_bd_id << " EnableCompression "
              << elem->m_1.compression_en << "\n";
  m_ir_stream << std::setw(sizeof(uint32_t) * 2) << std::setfill('0')
              << std::hex << elem->m_2.reg << ": D0step "
              << elem->m_2.d0_step_size << " D1step " << elem->m_2.d1_step_size
              << "\n";
  m_ir_stream << std::setw(sizeof(uint32_t) * 2) << std::setfill('0')
              << std::hex << elem->m_3.reg << ": D2step "
              << elem->m_3.d2_step_size << " D0wrap " << elem->m_3.d0_wrap
              << " D1wrap " << elem->m_3.d1_wrap << "\n";
  m_ir_stream << std::setw(sizeof(uint32_t) * 2) << std::setfill('0')
              << std::hex << elem->m_4.reg << ": IterStep "
              << elem->m_4.iter_step << " IterWrap " << elem->m_4.iter_wrap
              << " IterCurrent " << elem->m_4.iter_curr << "\n";
  m_ir_stream << std::setw(sizeof(uint32_t) * 2) << std::setfill('0')
              << std::hex << elem->m_5.reg << ": LockAcqID "
              << elem->m_5.lock_acq_id << " LockAcqVal "
              << elem->m_5.lock_acq_val << " LockAcqEn "
              << elem->m_5.lock_acq_en << " LockRelID " << elem->m_5.lock_rel_id
              << " LockRelVal " << elem->m_5.lock_rel_val << " ValidBD "
              << elem->m_5.valid_bd << " UseNextBD " << elem->m_5.use_next_bd
              << " NextBD " << elem->m_5.next_bd << " TLASTSuppress "
              << elem->m_5.tlast_suppress << "\n";

  // Return number of instructions written
  return 8;
}

int IPUDebugInstrWriter::write(AIE2::MemBD* elem, uint32_t* instrs) {
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

  // std::cout << "-- Element: " << elem->name() << " Index: " <<
  // (elem->getUniqueId() & 0x0000FFFF) << "\n";

  m_ir_stream << std::setw(sizeof(uint32_t) * 2) << std::setfill('0')
              << std::hex << word << ": opcode "
              << static_cast<uint32_t>(InstOpCode::WRITEBD_EXTEND_MEMTILE)
              << " Col " << elem->m_location.m_col << " Row "
              << elem->m_location.m_row << " NCols " << elem->m_ncols << "\n";
  m_ir_stream << std::setw(sizeof(uint32_t) * 2) << std::setfill('0')
              << std::hex << bd_list << ": BDList " << bd_list << "\n";
  m_ir_stream << std::setw(sizeof(uint32_t) * 2) << std::setfill('0')
              << std::hex << next_bd_list << ": NextBDList " << next_bd_list
              << "\n";
  m_ir_stream << std::setw(sizeof(uint32_t) * 2) << std::setfill('0')
              << std::hex << elem->m_0.reg << ": BufferLength "
              << elem->m_0.buffer_length << " OOOBd "
              << elem->m_0.out_of_order_bd_id << " PacketID "
              << elem->m_0.packet_id << " PacketType " << elem->m_0.packet_type
              << " EnablePacket " << elem->m_0.packet_en << "\n";
  m_ir_stream << std::setw(sizeof(uint32_t) * 2) << std::setfill('0')
              << std::hex << elem->m_1.reg << ": BaseAddr "
              << elem->m_1.base_addr << " UseNexBD " << elem->m_1.use_next_bd
              << " NextBD " << elem->m_1.next_bd << "\n";
  m_ir_stream << std::setw(sizeof(uint32_t) * 2) << std::setfill('0')
              << std::hex << elem->m_2.reg << ": D0step "
              << elem->m_2.d0_step_size << " D0wrap " << elem->m_2.d0_wrap
              << " TLASTSuppress " << elem->m_2.tlast_suppress << "\n";
  m_ir_stream << std::setw(sizeof(uint32_t) * 2) << std::setfill('0')
              << std::hex << elem->m_3.reg << ": D1step "
              << elem->m_3.d1_step_size << " D1wrap " << elem->m_3.d1_wrap
              << "\n";
  m_ir_stream << std::setw(sizeof(uint32_t) * 2) << std::setfill('0')
              << std::hex << elem->m_4.reg << ": D2step "
              << elem->m_4.d2_step_size << " D2wrap " << elem->m_4.d2_wrap
              << " EnableCompression " << elem->m_4.compression_en << "\n";
  m_ir_stream << std::setw(sizeof(uint32_t) * 2) << std::setfill('0')
              << std::hex << elem->m_5.reg << ": D3step "
              << elem->m_5.d3_step_size << "\n";
  m_ir_stream << std::setw(sizeof(uint32_t) * 2) << std::setfill('0')
              << std::hex << elem->m_6.reg << ": IterStep "
              << elem->m_6.iter_step << " IterWrap " << elem->m_6.iter_wrap
              << " IterCurrent " << elem->m_6.iter_curr << "\n";
  m_ir_stream << std::setw(sizeof(uint32_t) * 2) << std::setfill('0')
              << std::hex << elem->m_7.reg << ": LockAcqID "
              << elem->m_7.lock_acq_id << " LockAcqVal "
              << elem->m_7.lock_acq_val << " LockAcqEn "
              << elem->m_7.lock_acq_en << " LockRelID " << elem->m_7.lock_rel_id
              << " LockRelVal " << elem->m_7.lock_rel_val << " ValidBD "
              << elem->m_7.valid_bd << "\n";

  // Return number of instructions written
  return 11;
}

int IPUDebugInstrWriter::write(AIE2::NoCBD* elem, uint32_t* instrs) {
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

  // std::cout << "-- Element: " << elem->name() << " Index: " <<
  // (elem->getUniqueId() & 0x0000FFFF) << "\n";

  m_ir_stream << std::setw(sizeof(uint32_t) * 2) << std::setfill('0')
              << std::hex << word << ": opcode "
              << static_cast<uint32_t>(InstOpCode::WRITEBD_EXTEND_SHIMTILE)
              << " Col " << elem->m_location.m_col << " nCols " << elem->m_ncols
              << " DDRType " << elem->m_ddrtype << " BDId " << elem->m_id
              << "\n";
  m_ir_stream << std::setw(sizeof(uint32_t) * 2) << std::setfill('0')
              << std::hex << addr_incr << ": AddrIncr " << addr_incr << "\n";
  m_ir_stream << std::setw(sizeof(uint32_t) * 2) << std::setfill('0')
              << std::hex << elem->m_0.reg << ": BufferLength "
              << elem->m_0.buffer_length << "\n";
  m_ir_stream << std::setw(sizeof(uint32_t) * 2) << std::setfill('0')
              << std::hex << elem->m_1.reg << ": BaseAddrLow "
              << elem->m_1.base_addr_low << "\n";
  m_ir_stream << std::setw(sizeof(uint32_t) * 2) << std::setfill('0')
              << std::hex << elem->m_2.reg << ": BaseAddrHigh "
              << elem->m_2.base_addr_high << " PacketType "
              << elem->m_2.packet_type << " PacketID " << elem->m_2.packet_id
              << " OOOBd " << elem->m_2.out_of_order_bd_id << " EnablePacket "
              << elem->m_2.packet_en << "\n";
  m_ir_stream << std::setw(sizeof(uint32_t) * 2) << std::setfill('0')
              << std::hex << elem->m_3.reg << ": D0step "
              << elem->m_3.d0_step_size << " D0wrap " << elem->m_3.d0_wrap
              << " SecureAccess " << elem->m_3.secure_access << "\n";
  m_ir_stream << std::setw(sizeof(uint32_t) * 2) << std::setfill('0')
              << std::hex << elem->m_4.reg << ": D1step "
              << elem->m_4.d1_step_size << " D1wrap " << elem->m_4.d1_wrap
              << " BurstLength " << elem->m_4.burst_length << "\n";
  m_ir_stream << std::setw(sizeof(uint32_t) * 2) << std::setfill('0')
              << std::hex << elem->m_5.reg << ": D2step "
              << elem->m_5.d2_step_size << " AxQos " << elem->m_5.ax_qos
              << " AxCache " << elem->m_5.ax_cache << " SMID " << elem->m_5.smid
              << "\n";
  m_ir_stream << std::setw(sizeof(uint32_t) * 2) << std::setfill('0')
              << std::hex << elem->m_6.reg << ": IterStep "
              << elem->m_6.iter_step << " IterWrap " << elem->m_6.iter_wrap
              << " IterCurrent " << elem->m_6.iter_curr << "\n";
  m_ir_stream << std::setw(sizeof(uint32_t) * 2) << std::setfill('0')
              << std::hex << elem->m_7.reg << ": LockAcqID "
              << elem->m_7.lock_acq_id << " LockAcqVal "
              << elem->m_7.lock_acq_val << " LockAcqEn "
              << elem->m_7.lock_acq_en << " LockRelID " << elem->m_7.lock_rel_id
              << " LockRelVal " << elem->m_7.lock_rel_val << " ValidBD "
              << elem->m_7.valid_bd << " UseNexBD " << elem->m_7.use_next_bd
              << " NextBD " << elem->m_7.next_bd << " TLASTSuppress "
              << elem->m_7.tlast_suppress << "\n";

  // Return number of instructions written
  return 10;
}

// DMA Channel writers
//// Control
int IPUDebugInstrWriter::write(AIE2::AieDmaS2MMCtrl* elem, uint32_t* instrs) {
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

  // std::cout << "-- Element: " << elem->name() << " Index: " <<
  // (elem->getUniqueId() & 0x0000FFFF) << "\n";

  m_ir_stream << std::setw(sizeof(uint32_t) * 2) << std::setfill('0')
              << std::hex << word << ": opcode "
              << static_cast<uint32_t>(InstOpCode::WRITE32) << " Col "
              << elem->m_location.m_col << " Row " << elem->m_location.m_row
              << "\n";
  m_ir_stream << std::setw(sizeof(uint32_t) * 2) << std::setfill('0')
              << std::hex << addr << ": Addr " << addr << "\n";
  m_ir_stream << std::setw(sizeof(uint32_t) * 2) << std::setfill('0')
              << std::hex << data << ": Reset " << elem->m_regval.reset
              << " OOOEn " << elem->m_regval.out_of_order_en
              << " decompressionEn " << elem->m_regval.decompression_en
              << " controllerID " << elem->m_regval.controller_id << " FoTmode "
              << elem->m_regval.fot_mode << "\n";

  // Return number of instructions written
  return 3;
}

int IPUDebugInstrWriter::write(AIE2::AieDmaMM2SCtrl* elem, uint32_t* instrs) {
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

  // std::cout << "-- Element: " << elem->name() << " Index: " <<
  // (elem->getUniqueId() & 0x0000FFFF) << "\n";

  m_ir_stream << std::setw(sizeof(uint32_t) * 2) << std::setfill('0')
              << std::hex << word << ": opcode "
              << static_cast<uint32_t>(InstOpCode::WRITE32) << " Col "
              << elem->m_location.m_col << " Row " << elem->m_location.m_row
              << "\n";
  m_ir_stream << std::setw(sizeof(uint32_t) * 2) << std::setfill('0')
              << std::hex << addr << ": Addr " << addr << "\n";
  m_ir_stream << std::setw(sizeof(uint32_t) * 2) << std::setfill('0')
              << std::hex << data << ": Reset " << elem->m_regval.reset
              << " compressionEn " << elem->m_regval.compression_en
              << " controllerID " << elem->m_regval.controller_id << "\n";

  // Return number of instructions written
  return 3;
}

int IPUDebugInstrWriter::write(AIE2::MemDmaS2MMCtrl* elem, uint32_t* instrs) {
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

  // std::cout << "-- Element: " << elem->name() << " Index: " <<
  // (elem->getUniqueId() & 0x0000FFFF) << "\n";

  m_ir_stream << std::setw(sizeof(uint32_t) * 2) << std::setfill('0')
              << std::hex << word << ": opcode "
              << static_cast<uint32_t>(InstOpCode::WRITE32) << " Col "
              << elem->m_location.m_col << " Row " << elem->m_location.m_row
              << "\n";
  m_ir_stream << std::setw(sizeof(uint32_t) * 2) << std::setfill('0')
              << std::hex << addr << ": Addr " << addr << "\n";
  m_ir_stream << std::setw(sizeof(uint32_t) * 2) << std::setfill('0')
              << std::hex << data << ": Reset " << elem->m_regval.reset
              << " OOOEn " << elem->m_regval.out_of_order_en
              << " decompressionEn " << elem->m_regval.decompression_en
              << " controllerID " << elem->m_regval.controller_id << " FoTmode "
              << elem->m_regval.fot_mode << "\n";

  // Return number of instructions written
  return 3;
}

int IPUDebugInstrWriter::write(AIE2::MemDmaMM2SCtrl* elem, uint32_t* instrs) {
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

  // std::cout << "-- Element: " << elem->name() << " Index: " <<
  // (elem->getUniqueId() & 0x0000FFFF) << "\n";

  m_ir_stream << std::setw(sizeof(uint32_t) * 2) << std::setfill('0')
              << std::hex << word << ": opcode "
              << static_cast<uint32_t>(InstOpCode::WRITE32) << " Col "
              << elem->m_location.m_col << " Row " << elem->m_location.m_row
              << "\n";
  m_ir_stream << std::setw(sizeof(uint32_t) * 2) << std::setfill('0')
              << std::hex << addr << ": Addr " << addr << "\n";
  m_ir_stream << std::setw(sizeof(uint32_t) * 2) << std::setfill('0')
              << std::hex << data << ": Reset " << elem->m_regval.reset
              << " compressionEn " << elem->m_regval.compression_en
              << " controllerID " << elem->m_regval.controller_id << "\n";

  // Return number of instructions written
  return 3;
}

int IPUDebugInstrWriter::write(AIE2::NoCDmaS2MMCtrl* elem, uint32_t* instrs) {
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

  // std::cout << "-- Element: " << elem->name() << " Index: " <<
  // (elem->getUniqueId() & 0x0000FFFF) << "\n";

  m_ir_stream << std::setw(sizeof(uint32_t) * 2) << std::setfill('0')
              << std::hex << word << ": opcode "
              << static_cast<uint32_t>(InstOpCode::WRITE32) << " Col "
              << elem->m_location.m_col << " Row " << elem->m_location.m_row
              << "\n";
  m_ir_stream << std::setw(sizeof(uint32_t) * 2) << std::setfill('0')
              << std::hex << addr << ": Addr " << addr << "\n";
  m_ir_stream << std::setw(sizeof(uint32_t) * 2) << std::setfill('0')
              << std::hex << data << ": PauseMem " << elem->m_regval.pause_mem
              << " PauseStream " << elem->m_regval.pause_stream << " OOOEn "
              << elem->m_regval.out_of_order_en << " controllerID "
              << elem->m_regval.controller_id << " FoTmode "
              << elem->m_regval.fot_mode << "\n";

  // Return number of instructions written
  return 3;
}

int IPUDebugInstrWriter::write(AIE2::NoCDmaMM2SCtrl* elem, uint32_t* instrs) {
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

  // std::cout << "-- Element: " << elem->name() << " Index: " <<
  // (elem->getUniqueId() & 0x0000FFFF) << "\n";

  m_ir_stream << std::setw(sizeof(uint32_t) * 2) << std::setfill('0')
              << std::hex << word << ": opcode "
              << static_cast<uint32_t>(InstOpCode::WRITE32) << " Col "
              << elem->m_location.m_col << " Row " << elem->m_location.m_row
              << "\n";
  m_ir_stream << std::setw(sizeof(uint32_t) * 2) << std::setfill('0')
              << std::hex << addr << ": Addr " << addr << "\n";
  m_ir_stream << std::setw(sizeof(uint32_t) * 2) << std::setfill('0')
              << std::hex << data << ": PauseMem " << elem->m_regval.pause_mem
              << " PauseStream " << elem->m_regval.pause_stream
              << " controllerID " << elem->m_regval.controller_id << "\n";

  // Return number of instructions written
  return 3;
}

//// Queue
int IPUDebugInstrWriter::write(AIE2::AieDmaS2MMQueue* elem, uint32_t* instrs) {
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

  // std::cout << "-- Element: " << elem->name() << " Index: " <<
  // (elem->getUniqueId() & 0x0000FFFF) << "\n";

  m_ir_stream << std::setw(sizeof(uint32_t) * 2) << std::setfill('0')
              << std::hex << word << ": opcode "
              << static_cast<uint32_t>(InstOpCode::WRITE32) << " Col "
              << elem->m_location.m_col << " Row " << elem->m_location.m_row
              << "\n";
  m_ir_stream << std::setw(sizeof(uint32_t) * 2) << std::setfill('0')
              << std::hex << addr << ": Addr " << addr << "\n";
  m_ir_stream << std::setw(sizeof(uint32_t) * 2) << std::setfill('0')
              << std::hex << data << ": StartBD " << elem->m_regval.start_bd_id
              << " Repeat " << elem->m_regval.repeat << " EnableTokenIssue "
              << elem->m_regval.token_issue_en << "\n";

  // Return number of instructions written
  return 3;
}

int IPUDebugInstrWriter::write(AIE2::AieDmaMM2SQueue* elem, uint32_t* instrs) {
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

  // std::cout << "-- Element: " << elem->name() << " Index: " <<
  // (elem->getUniqueId() & 0x0000FFFF) << "\n";

  m_ir_stream << std::setw(sizeof(uint32_t) * 2) << std::setfill('0')
              << std::hex << word << ": opcode "
              << static_cast<uint32_t>(InstOpCode::WRITE32) << " Col "
              << elem->m_location.m_col << " Row " << elem->m_location.m_row
              << "\n";
  m_ir_stream << std::setw(sizeof(uint32_t) * 2) << std::setfill('0')
              << std::hex << addr << ": Addr " << addr << "\n";
  m_ir_stream << std::setw(sizeof(uint32_t) * 2) << std::setfill('0')
              << std::hex << data << ": StartBD " << elem->m_regval.start_bd_id
              << " Repeat " << elem->m_regval.repeat << " EnableTokenIssue "
              << elem->m_regval.token_issue_en << "\n";

  // Return number of instructions written
  return 3;
}

int IPUDebugInstrWriter::write(AIE2::MemDmaS2MMQueue* elem, uint32_t* instrs) {
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

  // std::cout << "-- Element: " << elem->name() << " Index: " <<
  // (elem->getUniqueId() & 0x0000FFFF) << "\n";

  m_ir_stream << std::setw(sizeof(uint32_t) * 2) << std::setfill('0')
              << std::hex << word << ": opcode "
              << static_cast<uint32_t>(InstOpCode::WRITE32) << " Col "
              << elem->m_location.m_col << " Row " << elem->m_location.m_row
              << "\n";
  m_ir_stream << std::setw(sizeof(uint32_t) * 2) << std::setfill('0')
              << std::hex << addr << ": Addr " << addr << "\n";
  m_ir_stream << std::setw(sizeof(uint32_t) * 2) << std::setfill('0')
              << std::hex << data << ": StartBD " << elem->m_regval.start_bd_id
              << " Repeat " << elem->m_regval.repeat << " EnableTokenIssue "
              << elem->m_regval.token_issue_en << "\n";

  // Return number of instructions written
  return 3;
}

int IPUDebugInstrWriter::write(AIE2::MemDmaMM2SQueue* elem, uint32_t* instrs) {
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

  // std::cout << "-- Element: " << elem->name() << " Index: " <<
  // (elem->getUniqueId() & 0x0000FFFF) << "\n";

  m_ir_stream << std::setw(sizeof(uint32_t) * 2) << std::setfill('0')
              << std::hex << word << ": opcode "
              << static_cast<uint32_t>(InstOpCode::WRITE32) << " Col "
              << elem->m_location.m_col << " Row " << elem->m_location.m_row
              << "\n";
  m_ir_stream << std::setw(sizeof(uint32_t) * 2) << std::setfill('0')
              << std::hex << addr << ": Addr " << addr << "\n";
  m_ir_stream << std::setw(sizeof(uint32_t) * 2) << std::setfill('0')
              << std::hex << data << ": StartBD " << elem->m_regval.start_bd_id
              << " Repeat " << elem->m_regval.repeat << " EnableTokenIssue "
              << elem->m_regval.token_issue_en << "\n";

  // Return number of instructions written
  return 3;
}

int IPUDebugInstrWriter::write(AIE2::NoCDmaS2MMQueue* elem, uint32_t* instrs) {
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

  // std::cout << "-- Element: " << elem->name() << " Index: " <<
  // (elem->getUniqueId() & 0x0000FFFF) << "\n";

  m_ir_stream << std::setw(sizeof(uint32_t) * 2) << std::setfill('0')
              << std::hex << word << ": opcode "
              << static_cast<uint32_t>(InstOpCode::WRITE32) << " Col "
              << elem->m_location.m_col << " Row " << elem->m_location.m_row
              << "\n";
  m_ir_stream << std::setw(sizeof(uint32_t) * 2) << std::setfill('0')
              << std::hex << addr << ": Addr " << addr << "\n";
  m_ir_stream << std::setw(sizeof(uint32_t) * 2) << std::setfill('0')
              << std::hex << data << ": StartBD " << elem->m_regval.start_bd_id
              << " Repeat " << elem->m_regval.repeat << " EnableTokenIssue "
              << elem->m_regval.token_issue_en << "\n";

  // Return number of instructions written
  return 3;
}

int IPUDebugInstrWriter::write(AIE2::NoCDmaMM2SQueue* elem, uint32_t* instrs) {
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

  // std::cout << "-- Element: " << elem->name() << " Index: " <<
  // (elem->getUniqueId() & 0x0000FFFF) << "\n";

  m_ir_stream << std::setw(sizeof(uint32_t) * 2) << std::setfill('0')
              << std::hex << word << ": opcode "
              << static_cast<uint32_t>(InstOpCode::WRITE32) << " Col "
              << elem->m_location.m_col << " Row " << elem->m_location.m_row
              << "\n";
  m_ir_stream << std::setw(sizeof(uint32_t) * 2) << std::setfill('0')
              << std::hex << addr << ": Addr " << addr << "\n";
  m_ir_stream << std::setw(sizeof(uint32_t) * 2) << std::setfill('0')
              << std::hex << data << ": StartBD " << elem->m_regval.start_bd_id
              << " Repeat " << elem->m_regval.repeat << " EnableTokenIssue "
              << elem->m_regval.token_issue_en << "\n";

  // Return number of instructions written
  return 3;
}

// Lock Writers
int IPUDebugInstrWriter::write(AIE2::ATLock* elem, uint32_t* instrs) {
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

  // std::cout << "-- Element: " << elem->name() << " Index: " <<
  // (elem->getUniqueId() & 0x0000FFFF) << "\n";

  m_ir_stream << std::setw(sizeof(uint32_t) * 2) << std::setfill('0')
              << std::hex << word << ": opcode "
              << static_cast<uint32_t>(InstOpCode::WRITE32) << " Col "
              << elem->m_location.m_col << " Row " << elem->m_location.m_row
              << "\n";
  m_ir_stream << std::setw(sizeof(uint32_t) * 2) << std::setfill('0')
              << std::hex << addr << ": Addr " << addr << "\n";
  m_ir_stream << std::setw(sizeof(uint32_t) * 2) << std::setfill('0')
              << std::hex << data << ": LockValue " << elem->m_value.value
              << "\n";

  // Return number of instructions written
  return 3;
}

int IPUDebugInstrWriter::write(AIE2::MTLock* elem, uint32_t* instrs) {
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

  // std::cout << "-- Element: " << elem->name() << " Index: " <<
  // (elem->getUniqueId() & 0x0000FFFF) << "\n";

  m_ir_stream << std::setw(sizeof(uint32_t) * 2) << std::setfill('0')
              << std::hex << word << ": opcode "
              << static_cast<uint32_t>(InstOpCode::WRITE32) << " Col "
              << elem->m_location.m_col << " Row " << elem->m_location.m_row
              << "\n";
  m_ir_stream << std::setw(sizeof(uint32_t) * 2) << std::setfill('0')
              << std::hex << addr << ": Addr " << addr << "\n";
  m_ir_stream << std::setw(sizeof(uint32_t) * 2) << std::setfill('0')
              << std::hex << data << ": LockValue " << elem->m_value.value
              << "\n";

  // Return number of instructions written
  return 3;
}

int IPUDebugInstrWriter::write(AIE2::NTLock* elem, uint32_t* instrs) {
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

  // std::cout << "-- Element: " << elem->name() << " Index: " <<
  // (elem->getUniqueId() & 0x0000FFFF) << "\n";

  m_ir_stream << std::setw(sizeof(uint32_t) * 2) << std::setfill('0')
              << std::hex << word << ": opcode "
              << static_cast<uint32_t>(InstOpCode::WRITE32) << " Col "
              << elem->m_location.m_col << " Row " << elem->m_location.m_row
              << "\n";
  m_ir_stream << std::setw(sizeof(uint32_t) * 2) << std::setfill('0')
              << std::hex << addr << ": Addr " << addr << "\n";
  m_ir_stream << std::setw(sizeof(uint32_t) * 2) << std::setfill('0')
              << std::hex << data << ": LockValue " << elem->m_value.value
              << "\n";

  // Return number of instructions written
  return 3;
}

// Word writers
int IPUDebugInstrWriter::write(AIE2::Word* elem, uint32_t* instrs) {
  // Get Opcode
  constexpr uint32_t opcode = genOpCode<InstOpCode::WRITE32>();
  uint32_t word =
      opcode | (elem->m_location.m_col << 16) | (elem->m_location.m_row << 8);

  // Write to instr buffer
  instrs[0] = word;
  instrs[1] = elem->m_addr;
  instrs[2] = elem->m_data;

  // std::cout << "-- Element: " << elem->name() << " Index: " <<
  // (elem->getUniqueId() & 0x0000FFFF) << "\n";

  m_ir_stream << std::setw(sizeof(uint32_t) * 2) << std::setfill('0')
              << std::hex << word << ": opcode "
              << static_cast<uint32_t>(InstOpCode::WRITE32) << " Col "
              << elem->m_location.m_col << " Row " << elem->m_location.m_row
              << "\n";
  m_ir_stream << std::setw(sizeof(uint32_t) * 2) << std::setfill('0')
              << std::hex << elem->m_addr << ": Addr " << elem->m_addr << "\n";
  m_ir_stream << std::setw(sizeof(uint32_t) * 2) << std::setfill('0')
              << std::hex << elem->m_data << ": data " << elem->m_data << "\n";

  // Return number of instructions written
  return 3;
}

int IPUDebugInstrWriter::write(AIE2::IndexWord* elem, uint32_t* instrs) {
  // Get Opcode
  constexpr uint32_t opcode = genOpCode<InstOpCode::WRITE32_RTP>();
  uint32_t word =
      opcode | (elem->m_location.m_col << 16) | (elem->m_location.m_row << 8);

  // Write to instr buffer
  instrs[0] = word;
  instrs[1] = elem->m_addr;
  instrs[2] = elem->m_data;

  // std::cout << "-- Element: " << elem->name() << " Index: " <<
  // (elem->getUniqueId() & 0x0000FFFF) << "\n";

  m_ir_stream << std::setw(sizeof(uint32_t) * 2) << std::setfill('0')
              << std::hex << word << ": opcode "
              << static_cast<uint32_t>(InstOpCode::WRITE32) << " Col "
              << elem->m_location.m_col << " Row " << elem->m_location.m_row
              << "\n";
  m_ir_stream << std::setw(sizeof(uint32_t) * 2) << std::setfill('0')
              << std::hex << elem->m_addr << ": Addr " << elem->m_addr << "\n";
  m_ir_stream << std::setw(sizeof(uint32_t) * 2) << std::setfill('0')
              << std::hex << elem->m_data << ": data " << elem->m_data << "\n";

  // Return number of instructions written
  return 3;
}

int IPUDebugInstrWriter::write(AIE2::SyncWord* elem, uint32_t* instrs) {
  // Get Opcode
  constexpr uint32_t opcode = genOpCode<InstOpCode::SYNC>();
  // Generate word
  uint32_t direction = static_cast<uint32_t>(elem->m_direction);
  uint32_t word = (opcode) | (elem->m_location.m_col << 16) |
                  (elem->m_location.m_row << 8) | (direction);
  uint32_t data =
      (elem->m_channel << 24) | (elem->m_ncols << 16) | (elem->m_nrows << 8);

  // Write to instr buffer
  instrs[0] = word;
  instrs[1] = data;

  // std::cout << "-- Element: " << elem->name() << " Index: " <<
  // (elem->getUniqueId() & 0x0000FFFF) << "\n";

  m_ir_stream << std::setw(sizeof(uint32_t) * 2) << std::setfill('0')
              << std::hex << word << ": opcode "
              << static_cast<uint32_t>(InstOpCode::SYNC) << " Col "
              << elem->m_location.m_col << " Row " << elem->m_location.m_row
              << " Direction " << direction << "\n";
  m_ir_stream << std::setw(sizeof(uint32_t) * 2) << std::setfill('0')
              << std::hex << data << ": Channel " << elem->m_channel
              << " nCols " << elem->m_ncols << " nRows " << elem->m_nrows
              << "\n";

  // Return number of instructions written
  return 2;
}
