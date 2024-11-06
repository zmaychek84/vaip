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
 * @file  aie2_ipu_instr_writer.hpp
 * @brief File containing class for genetating machine codes for IPU.
 */
#pragma once

#include "aie2_ipu_instr_writer_impl.hpp"
#include "aie2_ipu_isa.hpp"

/**
 * @class IPUInstructionWriter
 * @brief Class to generate machine code for IPU.
 */
class IPUInstructionWriter : public InstructionWriterImpl {
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
};
