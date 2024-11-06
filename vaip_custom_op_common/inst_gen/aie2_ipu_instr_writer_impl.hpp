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
 * @file  aie2_ipu_instr_writer_impl.hpp
 * @brief File containing abstract class for generating instruction IPU.
 */
#pragma once

#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace AIE2 {
// BD Elements
class AieBD;
class MemBD;
class NoCBD;
// Channel Control Elements
class AieDmaS2MMCtrl;
class AieDmaMM2SCtrl;
class MemDmaS2MMCtrl;
class MemDmaMM2SCtrl;
class NoCDmaS2MMCtrl;
class NoCDmaMM2SCtrl;
// Channel Queue Elements
class AieDmaS2MMQueue;
class AieDmaMM2SQueue;
class MemDmaS2MMQueue;
class MemDmaMM2SQueue;
class NoCDmaS2MMQueue;
class NoCDmaMM2SQueue;
// Lock Elements
class ATLock;
class MTLock;
class NTLock;
// Word Elements
class Word;
class IndexWord;
class SyncWord;
// Location
struct Location;
} // namespace AIE2

/**
 * @class InstructionWriterImpl
 * @brief Abstract class for machine instruction generation.
 *        Extend this class to create different instruction generators.
 */
class InstructionWriterImpl {
public:
  // DMA BD writers
  ////////////////////////////////////////////////////////////////////////////
  /**
   * @brief Generate instructions for AIE-tile BD.
   * @param elem          pointer to the BD element
   * @param instr_buffer  pointer to memory where the instruction to be written
   * @returns             number of instructions generated
   */
  virtual int write(AIE2::AieBD* elem, uint32_t* instr_buffer) = 0;

  /**
   * @brief Generate instructions for Mem-tile BD.
   * @param elem          pointer to the BD element
   * @param instr_buffer  pointer to memory where the instruction to be written
   * @returns             number of instructions generated
   */
  virtual int write(AIE2::MemBD* elem, uint32_t* instr_buffer) = 0;

  /**
   * @brief Generate instructions for NoC-tile BD.
   * @param elem          pointer to the BD element
   * @param instr_buffer  pointer to memory where the instruction to be written
   * @returns             number of instructions generated
   */
  virtual int write(AIE2::NoCBD* elem, uint32_t* instr_buffer) = 0;

  // DMA Channel writers
  ////////////////////////////////////////////////////////////////////////////

  //// Control
  /**
   * @brief Generate instructions for AIE-tile S2MM Control.
   * @param elem          pointer to the S2MM control element
   * @param instr_buffer  pointer to memory where the instruction to be written
   * @returns             number of instructions generated
   */
  virtual int write(AIE2::AieDmaS2MMCtrl* elem, uint32_t* instr_buffer) = 0;

  /**
   * @brief Generate instructions for AIE-tile MM2S Control.
   * @param elem          pointer to the MM2S control element
   * @param instr_buffer  pointer to memory where the instruction to be written
   * @returns             number of instructions generated
   */
  virtual int write(AIE2::AieDmaMM2SCtrl* elem, uint32_t* instr_buffer) = 0;

  /**
   * @brief Generate instructions for Mem-tile S2MM Control.
   * @param elem          pointer to the S2MM control element
   * @param instr_buffer  pointer to memory where the instruction to be written
   * @returns             number of instructions generated
   */
  virtual int write(AIE2::MemDmaS2MMCtrl* elem, uint32_t* instr_buffer) = 0;

  /**
   * @brief Generate instructions for Mem-tile MM2S Control.
   * @param elem          pointer to the MM2S control element
   * @param instr_buffer  pointer to memory where the instruction to be written
   * @returns             number of instructions generated
   */
  virtual int write(AIE2::MemDmaMM2SCtrl* elem, uint32_t* instr_buffer) = 0;

  /**
   * @brief Generate instructions for NoC-tile MM2S Control.
   * @param elem          pointer to the S2MM control element
   * @param instr_buffer  pointer to memory where the instruction to be written
   * @returns             number of instructions generated
   */
  virtual int write(AIE2::NoCDmaS2MMCtrl* elem, uint32_t* instr_buffer) = 0;

  /**
   * @brief Generate instructions for NoC-tile MM2S Control.
   * @param elem          pointer to the MM2S control element
   * @param instr_buffer  pointer to memory where the instruction to be written
   * @returns             number of instructions generated
   */
  virtual int write(AIE2::NoCDmaMM2SCtrl* elem, uint32_t* instr_buffer) = 0;

  //// Queue
  ////////////////////////////////////////////////////////////////////////////
  /**
   * @brief Generate instructions for AIE-tile S2MM Queue.
   * @param elem          pointer to the S2MM queue element
   * @param instr_buffer  pointer to memory where the instruction to be written
   * @returns             number of instructions generated
   */
  virtual int write(AIE2::AieDmaS2MMQueue* elem, uint32_t* instr_buffer) = 0;

  /**
   * @brief Generate instructions for AIE-tile MM2S Queue.
   * @param elem          pointer to the MM2S queue element
   * @param instr_buffer  pointer to memory where the instruction to be written
   * @returns             number of instructions generated
   */
  virtual int write(AIE2::AieDmaMM2SQueue* elem, uint32_t* instr_buffer) = 0;

  /**
   * @brief Generate instructions for Mem-tile S2MM Queue.
   * @param elem          pointer to the S2MM queue element
   * @param instr_buffer  pointer to memory where the instruction to be written
   * @returns             number of instructions generated
   */
  virtual int write(AIE2::MemDmaS2MMQueue* elem, uint32_t* instr_buffer) = 0;

  /**
   * @brief Generate instructions for Mem-tile MM2S Queue.
   * @param elem          pointer to the MM2S queue element
   * @param instr_buffer  pointer to memory where the instruction to be written
   * @returns             number of instructions generated
   */
  virtual int write(AIE2::MemDmaMM2SQueue* elem, uint32_t* instr_buffer) = 0;

  /**
   * @brief Generate instructions for NoC-tile S2MM Queue.
   * @param elem          pointer to the S2MM queue element
   * @param instr_buffer  pointer to memory where the instruction to be written
   * @returns             number of instructions generated
   */
  virtual int write(AIE2::NoCDmaS2MMQueue* elem, uint32_t* instr_buffer) = 0;

  /**
   * @brief Generate instructions for NoC-tile MM2S Queue.
   * @param elem          pointer to the MM2S queue element
   * @param instr_buffer  pointer to memory where the instruction to be written
   * @returns             number of instructions generated
   */
  virtual int write(AIE2::NoCDmaMM2SQueue* elem, uint32_t* instr_buffer) = 0;

  // Lock Writers
  ////////////////////////////////////////////////////////////////////////////
  /**
   * @brief Generate instructions for AIE-tile Lock.
   * @param elem          pointer to the ATLock element
   * @param instr_buffer  pointer to memory where the instruction to be written
   * @returns             number of instructions generated
   */
  virtual int write(AIE2::ATLock* elem, uint32_t* instr_buffer) = 0;

  /**
   * @brief Generate instructions for Mem-tile Lock.
   * @param elem          pointer to the MTLock element
   * @param instr_buffer  pointer to memory where the instruction to be written
   * @returns             number of instructions generated
   */
  virtual int write(AIE2::MTLock* elem, uint32_t* instr_buffer) = 0;

  /**
   * @brief Generate instructions for NoC-tile Lock.
   * @param elem          pointer to the NTLock element
   * @param instr_buffer  pointer to memory where the instruction to be written
   * @returns             number of instructions generated
   */
  virtual int write(AIE2::NTLock* elem, uint32_t* instr_buffer) = 0;

  // Word writers
  ////////////////////////////////////////////////////////////////////////////
  /**
   * @brief Generate instructions for AIE-tile Word.
   * @param elem          pointer to the word element
   * @param instr_buffer  pointer to memory w/here the instruction to be written
   * @returns             number of instructions generated
   */
  virtual int write(AIE2::Word* elem, uint32_t* instr_buffer) = 0;

  /**
   * @brief Generate instructions for AIE-tile IndexWord.
   * @param elem          pointer to the index word element
   * @param instr_buffer  pointer to memory where the instruction to be written
   * @returns             number of instructions generated
   */
  virtual int write(AIE2::IndexWord* elem, uint32_t* instr_buffer) = 0;

  /**
   * @brief Generate instructions for NoC-tile SyncWord.
   * @param elem          pointer to the sync word element
   * @param instr_buffer  pointer to memory where the instruction to be written
   * @returns             number of instructions generated
   */
  virtual int write(AIE2::SyncWord* elem, uint32_t* instr_buffer) = 0;
};
