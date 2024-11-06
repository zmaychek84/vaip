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
#include "aie2_instr_ir_writer.hpp"
#include "aie2_dma_types.hpp"

// DMA BD writers
int InstructionIRWriter::write(AIE2::AieBD* elem, uint32_t* instrs) {
  // AIE tile config
  m_ir_stream << "#AIEDMA Col " << elem->m_location.m_col << " Row "
              << elem->m_location.m_row << " NCols " << elem->m_ncols
              << " NRows " << elem->m_nrows << "\n"
              << " BufferLength " << elem->m_0.buffer_length << " BaseAddr "
              << elem->m_0.base_addr << "\n"
              << " PacketType " << elem->m_1.packet_type << " PacketID "
              << elem->m_1.packet_id << " EnablePacket " << elem->m_1.packet_en
              << " OOOBd " << elem->m_1.out_of_order_bd_id
              << " EnableCompression " << elem->m_1.compression_en << "\n"
              << " D0step " << elem->m_2.d0_step_size << " D0wrap "
              << elem->m_3.d0_wrap << " D1step " << elem->m_2.d1_step_size
              << " D1wrap " << elem->m_3.d1_wrap << " D2step "
              << elem->m_3.d2_step_size << "\n"
              << " IterStep " << elem->m_4.iter_step << " IterWrap "
              << elem->m_4.iter_wrap << " IterCurrent " << elem->m_4.iter_curr
              << "\n"
              << " LockAcqID " << elem->m_5.lock_acq_id << " LockAcqVal "
              << elem->m_5.lock_acq_val << " LockAcqEn "
              << elem->m_5.lock_acq_en << " LockRelID " << elem->m_5.lock_rel_id
              << " LockRelVal " << elem->m_5.lock_rel_val << "\n"
              << " ValidBD " << elem->m_5.valid_bd << " NextBD "
              << elem->m_5.next_bd << " TLASTSuppress "
              << elem->m_5.tlast_suppress << "\n";

  // Return number of instructions written
  return 8;
}

int InstructionIRWriter::write(AIE2::MemBD* elem, uint32_t* instrs) {
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
  // MEM tile config
  m_ir_stream << "#MEMDMA Col " << elem->m_location.m_col << " Row "
              << elem->m_location.m_row << " NCols " << elem->m_ncols
              << " BDList " << bd_list << " NextBDList " << next_bd_list << "\n"
              << " BufferLength " << elem->m_0.buffer_length << " BaseAddr "
              << elem->m_1.base_addr << "\n"
              << " PacketType " << elem->m_0.packet_type << " PacketID "
              << elem->m_0.packet_id << " EnablePacket " << elem->m_0.packet_en
              << " OOOBd " << elem->m_0.out_of_order_bd_id
              << " EnableCompression " << elem->m_4.compression_en << "\n"
              << " D0step " << elem->m_2.d0_step_size << " D0wrap "
              << elem->m_2.d0_wrap << " D1step " << elem->m_3.d1_step_size
              << " D1wrap " << elem->m_3.d1_wrap << " D2step "
              << elem->m_4.d2_step_size << " D2wrap " << elem->m_4.d2_wrap
              << " D3step " << elem->m_5.d3_step_size << "\n"
              << " IterStep " << elem->m_6.iter_step << " IterWrap "
              << elem->m_6.iter_wrap << " IterCurrent " << elem->m_6.iter_curr
              << "\n"
              << " LockAcqID " << elem->m_7.lock_acq_id << " LockAcqVal "
              << elem->m_7.lock_acq_val << " LockAcqEn "
              << elem->m_7.lock_acq_en << " LockRelID " << elem->m_7.lock_rel_id
              << " LockRelVal " << elem->m_7.lock_rel_val << "\n"
              << " UseNexBD " << elem->m_1.use_next_bd << " NextBD "
              << elem->m_1.next_bd << " ValidBD " << elem->m_7.valid_bd
              << " TLASTSuppress " << elem->m_2.tlast_suppress << "\n";

  // Return number of instructions written
  return 11;
}

int InstructionIRWriter::write(AIE2::NoCBD* elem, uint32_t* instrs) {
  // NoC tile config
  m_ir_stream << "#NoCDMA Col " << elem->m_location.m_col << " nCols "
              << elem->m_ncols << " DDRType " << elem->m_ddrtype << " BDId "
              << elem->m_id << " AddrIncr " << 0 << "\n"
              << " BufferLength " << elem->m_0.buffer_length << " BaseAddrLow "
              << elem->m_1.base_addr_low << " BaseAddrHigh "
              << elem->m_2.base_addr_high << "\n"
              << " PacketType " << elem->m_2.packet_type << " PacketID "
              << elem->m_2.packet_id << " EnablePacket " << elem->m_2.packet_en
              << " OOOBd " << elem->m_2.out_of_order_bd_id << " D0step "
              << elem->m_3.d0_step_size << " D0wrap " << elem->m_3.d0_wrap
              << " D1step " << elem->m_4.d1_step_size << " D1wrap "
              << elem->m_4.d1_wrap << " D2step " << elem->m_5.d2_step_size
              << "\n"
              << " IterStep " << elem->m_6.iter_step << " IterWrap "
              << elem->m_6.iter_wrap << " IterCurrent " << elem->m_6.iter_curr
              << "\n"
              << " LockAcqID " << elem->m_7.lock_acq_id << " LockAcqVal "
              << elem->m_7.lock_acq_val << " LockAcqEn "
              << elem->m_7.lock_acq_en << " LockRelID " << elem->m_7.lock_rel_id
              << " LockRelVal " << elem->m_7.lock_rel_val << "\n"
              << " UseNexBD " << elem->m_7.use_next_bd << " NextBD "
              << elem->m_7.next_bd << " ValidBD " << elem->m_7.valid_bd << "\n"
              << " TLASTSuppress " << elem->m_7.tlast_suppress
              << " SecureAccess " << elem->m_3.secure_access << " BurstLength "
              << elem->m_4.burst_length << " AxQos " << elem->m_5.ax_qos
              << " AxCache " << elem->m_5.ax_cache << " SMID " << elem->m_5.smid
              << "\n";

  // Return number of instructions written
  return 10;
}

// DMA Channel writers
//// Control
int InstructionIRWriter::write(AIE2::AieDmaS2MMCtrl* elem, uint32_t* instrs) {
  // Channel Control
  if (elem->m_regval.reg == 2)
    m_ir_stream << "#AIEChannelReset";
  else if (elem->m_regval.reg == 0)
    m_ir_stream << "#AIEChannelUnreset";
  else
    m_ir_stream << "#AIEChannel";

  m_ir_stream << " Col " << elem->m_location.m_col << " Row "
              << elem->m_location.m_row << " ID "
              << static_cast<uint32_t>(elem->m_id) << " Direction S2MM"
              << "\n";

  // Return number of instructions written
  return 3;
}

int InstructionIRWriter::write(AIE2::AieDmaMM2SCtrl* elem, uint32_t* instrs) {
  // Channel Control
  if (elem->m_regval.reg == 2)
    m_ir_stream << "#AIEChannelReset";
  else if (elem->m_regval.reg == 0)
    m_ir_stream << "#AIEChannelUnreset";
  else
    m_ir_stream << "#AIEChannel";

  m_ir_stream << " Col " << elem->m_location.m_col << " Row "
              << elem->m_location.m_row << " ID "
              << static_cast<uint32_t>(elem->m_id) << " Direction MM2S"
              << "\n";

  // Return number of instructions written
  return 3;
}

int InstructionIRWriter::write(AIE2::MemDmaS2MMCtrl* elem, uint32_t* instrs) {
  // Channel Control
  if (elem->m_regval.reg == 2)
    m_ir_stream << "#MEMChannelReset";
  else if (elem->m_regval.reg == 0)
    m_ir_stream << "#MEMChannelUnreset";
  else
    m_ir_stream << "#MEMChannel";

  m_ir_stream << " Col " << elem->m_location.m_col << " Row "
              << elem->m_location.m_row << " ID "
              << static_cast<uint32_t>(elem->m_id) << " Direction S2MM"
              << "\n";

  // Return number of instructions written
  return 3;
}

int InstructionIRWriter::write(AIE2::MemDmaMM2SCtrl* elem, uint32_t* instrs) {
  // Channel Control
  if (elem->m_regval.reg == 2)
    m_ir_stream << "#MEMChannelReset";
  else if (elem->m_regval.reg == 0)
    m_ir_stream << "#MEMChannelUnreset";
  else
    m_ir_stream << "#MEMChannel";

  m_ir_stream << " Col " << elem->m_location.m_col << " Row "
              << elem->m_location.m_row << " ID "
              << static_cast<uint32_t>(elem->m_id) << " Direction MM2S"
              << "\n";

  // Return number of instructions written
  return 3;
}

int InstructionIRWriter::write(AIE2::NoCDmaS2MMCtrl* elem, uint32_t* instrs) {
  // Channel Control
  if (elem->m_regval.reg == 2)
    m_ir_stream << "#NoCChannelReset";
  else if (elem->m_regval.reg == 0)
    m_ir_stream << "#NoCChannelUnreset";
  else
    m_ir_stream << "#NoCChannel";

  m_ir_stream << " Col " << elem->m_location.m_col << " Row "
              << elem->m_location.m_row << " ID "
              << static_cast<uint32_t>(elem->m_id) << " Direction S2MM"
              << "\n";

  // Return number of instructions written
  return 3;
}

int InstructionIRWriter::write(AIE2::NoCDmaMM2SCtrl* elem, uint32_t* instrs) {
  // Channel Control
  if (elem->m_regval.reg == 2)
    m_ir_stream << "#NoCChannelReset";
  else if (elem->m_regval.reg == 0)
    m_ir_stream << "#NoCChannelUnreset";
  else
    m_ir_stream << "#NoCChannel";

  m_ir_stream << " Col " << elem->m_location.m_col << " Row "
              << elem->m_location.m_row << " ID "
              << static_cast<uint32_t>(elem->m_id) << " Direction MM2S"
              << "\n";

  // Return number of instructions written
  return 3;
}

//// Queue
int InstructionIRWriter::write(AIE2::AieDmaS2MMQueue* elem, uint32_t* instrs) {
  // Channel Queue
  m_ir_stream << "#AIEPush2Queue"
              << " Col " << elem->m_location.m_col << " Row "
              << elem->m_location.m_row << " ID "
              << static_cast<uint32_t>(elem->m_id) << " Direction S2MM"
              << "\n"
              << " StartBD " << elem->m_regval.start_bd_id << " Repeat "
              << elem->m_regval.repeat << " EnableTokenIssue "
              << elem->m_regval.token_issue_en << "\n";

  // Return number of instructions written
  return 3;
}

int InstructionIRWriter::write(AIE2::AieDmaMM2SQueue* elem, uint32_t* instrs) {
  // Channel Queue
  m_ir_stream << "#AIEPush2Queue"
              << " Col " << elem->m_location.m_col << " Row "
              << elem->m_location.m_row << " ID "
              << static_cast<uint32_t>(elem->m_id) << " Direction MM2S"
              << "\n"
              << " StartBD " << elem->m_regval.start_bd_id << " Repeat "
              << elem->m_regval.repeat << " EnableTokenIssue "
              << elem->m_regval.token_issue_en << "\n";

  // Return number of instructions written
  return 3;
}

int InstructionIRWriter::write(AIE2::MemDmaS2MMQueue* elem, uint32_t* instrs) {
  // Channel Queue
  m_ir_stream << "#MEMPush2Queue"
              << " Col " << elem->m_location.m_col << " Row "
              << elem->m_location.m_row << " ID "
              << static_cast<uint32_t>(elem->m_id) << " Direction S2MM"
              << "\n"
              << " StartBD " << elem->m_regval.start_bd_id << " Repeat "
              << elem->m_regval.repeat << " EnableTokenIssue "
              << elem->m_regval.token_issue_en << "\n";

  // Return number of instructions written
  return 3;
}

int InstructionIRWriter::write(AIE2::MemDmaMM2SQueue* elem, uint32_t* instrs) {
  // Channel Queue
  m_ir_stream << "#MEMPush2Queue"
              << " Col " << elem->m_location.m_col << " Row "
              << elem->m_location.m_row << " ID "
              << static_cast<uint32_t>(elem->m_id) << " Direction MM2S"
              << "\n"
              << " StartBD " << elem->m_regval.start_bd_id << " Repeat "
              << elem->m_regval.repeat << " EnableTokenIssue "
              << elem->m_regval.token_issue_en << "\n";

  // Return number of instructions written
  return 3;
}

int InstructionIRWriter::write(AIE2::NoCDmaS2MMQueue* elem, uint32_t* instrs) {
  // Channel Queue
  m_ir_stream << "#NoCPush2Queue"
              << " Col " << elem->m_location.m_col << " Row "
              << elem->m_location.m_row << " ID "
              << static_cast<uint32_t>(elem->m_id) << " Direction S2MM"
              << "\n"
              << " StartBD " << elem->m_regval.start_bd_id << " Repeat "
              << elem->m_regval.repeat << " EnableTokenIssue "
              << elem->m_regval.token_issue_en << "\n";

  // Return number of instructions written
  return 3;
}

int InstructionIRWriter::write(AIE2::NoCDmaMM2SQueue* elem, uint32_t* instrs) {
  // Channel Queue
  m_ir_stream << "#NoCPush2Queue"
              << " Col " << elem->m_location.m_col << " Row "
              << elem->m_location.m_row << " ID "
              << static_cast<uint32_t>(elem->m_id) << " Direction MM2S"
              << "\n"
              << " StartBD " << elem->m_regval.start_bd_id << " Repeat "
              << elem->m_regval.repeat << " EnableTokenIssue "
              << elem->m_regval.token_issue_en << "\n";

  // Return number of instructions written
  return 3;
}

// Lock Writers
int InstructionIRWriter::write(AIE2::ATLock* elem, uint32_t* instrs) {
  // AIE Lock
  m_ir_stream << "#AIESetLock"
              << " Col " << elem->m_location.m_col << " Row "
              << elem->m_location.m_row << " ID "
              << static_cast<uint32_t>(elem->m_id) << " Value "
              << elem->m_value.value << "\n";

  // Return number of instructions written
  return 3;
}

int InstructionIRWriter::write(AIE2::MTLock* elem, uint32_t* instrs) {
  // MEM Lock
  m_ir_stream << "#MEMSetLock"
              << " Col " << elem->m_location.m_col << " Row "
              << elem->m_location.m_row << " ID "
              << static_cast<uint32_t>(elem->m_id) << " Value "
              << elem->m_value.value << "\n";

  // Return number of instructions written
  return 3;
}

int InstructionIRWriter::write(AIE2::NTLock* elem, uint32_t* instrs) {
  // NoC Lock
  m_ir_stream << "#NoCSetLock"
              << " Col " << elem->m_location.m_col << " Row "
              << elem->m_location.m_row << " ID "
              << static_cast<uint32_t>(elem->m_id) << " Value "
              << elem->m_value.value << "\n";

  // Return number of instructions written
  return 3;
}

// Word writers
int InstructionIRWriter::write(AIE2::Word* elem, uint32_t* instrs) {
  // AIE Word
  m_ir_stream << "#WriteWord32"
              << " Col " << elem->m_location.m_col << " Row "
              << elem->m_location.m_row << " Addr " << std::hex << elem->m_addr
              << " Value " << std::dec << elem->m_data << "\n";

  // Return number of instructions written
  return 3;
}

int InstructionIRWriter::write(AIE2::IndexWord* elem, uint32_t* instrs) {
  // AIE Index Word
  m_ir_stream << "#WriteIndex32"
              << " Col " << elem->m_location.m_col << " Row "
              << elem->m_location.m_row << " Addr " << std::hex << elem->m_addr
              << " Value " << std::dec << elem->m_data << "\n";

  // Return number of instructions written
  return 3;
}

int InstructionIRWriter::write(AIE2::SyncWord* elem, uint32_t* instrs) {
  // AIE Index Word
  m_ir_stream << "#Sync"
              << " Col " << elem->m_location.m_col << " Row "
              << elem->m_location.m_row << " Direction "
              << (static_cast<uint8_t>(elem->m_direction) ? "MM2S" : "S2MM")
              << "\n"
              << " Channel " << static_cast<uint32_t>(elem->m_channel)
              << " NCols " << static_cast<uint32_t>(elem->m_ncols) << " NRows "
              << static_cast<uint32_t>(elem->m_nrows) << "\n";

  // Return number of instructions written
  return 2;
}
