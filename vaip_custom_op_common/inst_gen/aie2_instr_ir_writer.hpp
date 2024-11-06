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
/**
 * @file  aie2_instr_ir_writer.hpp
 * @brief File containing class for writing instruction IR for different DMA
 * elements.
 */
#pragma once

#include "aie2_ipu_instr_writer_impl.hpp"
#include "aie2_ipu_isa.hpp"

/**
 * @class InstructionIRWriter
 * @brief Class to create IR (Intermediate Representation) for generating
 * machine code. Extends the abstract visitor class to create IR for different
 * DMA elements.
 */
class InstructionIRWriter : public InstructionWriterImpl {
public:
  // DMA BD writers
  int write(AIE2::AieBD* elem, uint32_t* instr_buffer = nullptr) override;
  int write(AIE2::MemBD* elem, uint32_t* instr_buffer = nullptr) override;
  int write(AIE2::NoCBD* elem, uint32_t* instr_buffer = nullptr) override;
  // DMA Channel writers
  //// Control
  int write(AIE2::AieDmaS2MMCtrl* elem,
            uint32_t* instr_buffer = nullptr) override;
  int write(AIE2::AieDmaMM2SCtrl* elem,
            uint32_t* instr_buffer = nullptr) override;
  int write(AIE2::MemDmaS2MMCtrl* elem,
            uint32_t* instr_buffer = nullptr) override;
  int write(AIE2::MemDmaMM2SCtrl* elem,
            uint32_t* instr_buffer = nullptr) override;
  int write(AIE2::NoCDmaS2MMCtrl* elem,
            uint32_t* instr_buffer = nullptr) override;
  int write(AIE2::NoCDmaMM2SCtrl* elem,
            uint32_t* instr_buffer = nullptr) override;
  //// Queue
  int write(AIE2::AieDmaS2MMQueue* elem,
            uint32_t* instr_buffer = nullptr) override;
  int write(AIE2::AieDmaMM2SQueue* elem,
            uint32_t* instr_buffer = nullptr) override;
  int write(AIE2::MemDmaS2MMQueue* elem,
            uint32_t* instr_buffer = nullptr) override;
  int write(AIE2::MemDmaMM2SQueue* elem,
            uint32_t* instr_buffer = nullptr) override;
  int write(AIE2::NoCDmaS2MMQueue* elem,
            uint32_t* instr_buffer = nullptr) override;
  int write(AIE2::NoCDmaMM2SQueue* elem,
            uint32_t* instr_buffer = nullptr) override;
  // Lock Writers
  int write(AIE2::ATLock* elem, uint32_t* instr_buffer = nullptr) override;
  int write(AIE2::MTLock* elem, uint32_t* instr_buffer = nullptr) override;
  int write(AIE2::NTLock* elem, uint32_t* instr_buffer = nullptr) override;
  // Word writers
  int write(AIE2::Word* elem, uint32_t* instr_buffer = nullptr) override;
  int write(AIE2::IndexWord* elem, uint32_t* instr_buffer = nullptr) override;
  int write(AIE2::SyncWord* elem, uint32_t* instr_buffer = nullptr) override;

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
