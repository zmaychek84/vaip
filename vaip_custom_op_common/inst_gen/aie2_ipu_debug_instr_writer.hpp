/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */
/**
 * @file  aie2_ipu_debug_instr_writer.hpp
 * @brief File containing class for writing instruction with break down of each
 * machine instruction.
 */
#pragma once

#include "aie2_ipu_instr_writer_impl.hpp"
#include "aie2_ipu_isa.hpp"

/**
 * @class IPUDebugInstrWriter
 * @brief Class to generate machine code with break down of each machine code.
 */
class IPUDebugInstrWriter : public InstructionWriterImpl {
public:
  // DMA BD writers
  int write(AIE2::AieBD* elem, uint32_t* instr_buffer) override;
  int write(AIE2::MemBD* elem, uint32_t* instr_buffer) override;
  int write(AIE2::NoCBD* elem, uint32_t* instr_buffer) override;
  // DMA Channel writers
  //// Control
  int write(AIE2::AieDmaS2MMCtrl* elem, uint32_t* instr_buffer) override;
  int write(AIE2::AieDmaMM2SCtrl* elem, uint32_t* instr_buffer) override;
  int write(AIE2::MemDmaS2MMCtrl* elem, uint32_t* instr_buffer) override;
  int write(AIE2::MemDmaMM2SCtrl* elem, uint32_t* instr_buffer) override;
  int write(AIE2::NoCDmaS2MMCtrl* elem, uint32_t* instr_buffer) override;
  int write(AIE2::NoCDmaMM2SCtrl* elem, uint32_t* instr_buffer) override;
  //// Queue
  int write(AIE2::AieDmaS2MMQueue* elem, uint32_t* instr_buffer) override;
  int write(AIE2::AieDmaMM2SQueue* elem, uint32_t* instr_buffer) override;
  int write(AIE2::MemDmaS2MMQueue* elem, uint32_t* instr_buffer) override;
  int write(AIE2::MemDmaMM2SQueue* elem, uint32_t* instr_buffer) override;
  int write(AIE2::NoCDmaS2MMQueue* elem, uint32_t* instr_buffer) override;
  int write(AIE2::NoCDmaMM2SQueue* elem, uint32_t* instr_buffer) override;
  // Lock Writers
  int write(AIE2::ATLock* elem, uint32_t* instr_buffer) override;
  int write(AIE2::MTLock* elem, uint32_t* instr_buffer) override;
  int write(AIE2::NTLock* elem, uint32_t* instr_buffer) override;
  // Word writers
  int write(AIE2::Word* elem, uint32_t* instr_buffer) override;
  int write(AIE2::IndexWord* elem, uint32_t* instr_buffer) override;
  int write(AIE2::SyncWord* elem, uint32_t* instr_buffer) override;

  // Write to console
  void writeToConsole(void) { std::cout << m_ir_stream.str() << "\n"; }

  // Write to file
  int writeToFile(const std::string& fname) {
    std::ofstream instr_fp(fname);
    if (!instr_fp.is_open()) {
      std::cerr << "Error: Unable to open file " << fname << " for writing!\n";
      return 1;
    }
    instr_fp << m_ir_stream.str() << "\n";
    instr_fp.close();
    return 0;
  }

private:
  std::stringstream m_ir_stream;
};
