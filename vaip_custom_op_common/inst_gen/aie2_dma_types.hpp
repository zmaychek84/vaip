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
 * @file  aie2_dma_types.hpp
 * @brief File containing classes, representing elements for DMA transactions
 *        to/from AIE, Mem and NoC tiles.
 */
#ifdef __GNUC__
#  pragma GCC diagnostic ignored "-Wpedantic"
#  pragma GCC diagnostic ignored "-Wconversion"
#  pragma GCC diagnostic ignored "-Wsign-compare"
#endif
#pragma once

// #define DEBUG_FWR 1
#define USE_VISITOR 1

#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>

#include "aie2_ipu_isa.hpp"
#include "aie2_registers.hpp"
#include "aie2_sanity_check.hpp"
#include "aie2_types.hpp"

class InstructionWriterImpl;

namespace AIE2 {

/**
 * @enum    TileType
 * @brief   Type of tile.
 */
enum TileType : uint32_t { AIE, MEM, NoC };

/**
 * @enum    ElementType
 * @brief   Type of DMA element.
 */
enum ElementType : uint32_t {
  BD,
  S2MMCTRL,
  MM2SCTRL,
  S2MMQUEUE,
  MM2SQUEUE,
  LOCK,
  WORD,
  INDEXWORD,
  SYNCWORD
};

/**
 * @brief   Function to generate DMA element code.
 * @returns element code in an integer
 */
template <TileType TType, ElementType EType>
constexpr uint32_t getElementCode() {
  return ((TType << 28) | (EType << 24));
}

/**
 * @var   NUM_AIE_DMA_BDS
 * @brief Total number of BDs available in AIE-tile.
 */
constexpr int NUM_AIE_DMA_BDS = 16;
/**
 * @var   NUM_AIE_DMA_CH_S2MM
 * @brief Total number of DMA S2MM Channels available in AIE-tile.
 */
constexpr int NUM_AIE_DMA_CH_S2MM = 2;
/**
 * @var   NUM_AIE_DMA_CH_MM2S
 * @brief Total number of DMA MM2S Channels available in AIE-tile.
 */
constexpr int NUM_AIE_DMA_CH_MM2S = 2;
/**
 * @var   NUM_AIE_LOCKS
 * @brief Total number of Locks available in AIE-tile.
 */
constexpr int NUM_AIE_LOCKS = 16;

/**
 * @var   NUM_MEM_DMA_BDS
 * @brief Total number of BDs available in Mem-tile.
 */
constexpr int NUM_MEM_DMA_BDS = 48;
/**
 * @var   NUM_MEM_DMA_CH_S2MM
 * @brief Total number of DMA S2MM Channels available in Mem-tile.
 */
constexpr int NUM_MEM_DMA_CH_S2MM = 6;
/**
 * @var   NUM_MEM_DMA_CH_MM2S
 * @brief Total number of DMA MM2S Channels available in Mem-tile.
 */
constexpr int NUM_MEM_DMA_CH_MM2S = 6;
/**
 * @var   NUM_MEM_LOCKS
 * @brief Total number of Locks available in Mem-tile.
 */
constexpr int NUM_MEM_LOCKS = 64;

/**
 * @var   NUM_NOC_DMA_BDS
 * @brief Total number of BDs available in NoC-tile.
 */
constexpr int NUM_NOC_DMA_BDS = 16;
/**
 * @var   NUM_NOC_DMA_CH_S2MM
 * @brief Total number of DMA S2MM Channels available in NoC-tile.
 */
constexpr int NUM_NOC_DMA_CH_S2MM = 2;
/**
 * @var   NUM_NOC_DMA_CH_MM2S
 * @brief Total number of DMA MM2S Channels available in NoC-tile.
 */
constexpr int NUM_NOC_DMA_CH_MM2S = 2;
/**
 * @var   NUM_NOC_LOCKS
 * @brief Total number of Locks available in NoC-tile.
 */
constexpr int NUM_NOC_LOCKS = 16;

/**
 * @class DmaElement
 * @brief Abstract class for all DMA elements.
 *
 * This class should be extended for all types of DMA elements.
 */
class DmaElement {
public:
  /**
   * @brief Default constructor.
   */
  DmaElement(int32_t id = 0) : m_id(id) {}
  /**
   * Virtual destructor.
   */
  virtual ~DmaElement() {}
  /**
   * @brief Method to accept a Visitor to generate instructions.
   * @param writer        pointer to the instruction writer implementation
   * @param instr_buffer  pointer to memory where the instruction should be
   * written
   * @return              number of instructions generated
   */
  virtual int accept(InstructionWriterImpl* writer, uint32_t* instr_buffer) = 0;

  /**
   * @brief Set ID for the element.
   * @param id  ID to be set
   */
  inline void setId(int8_t id) { m_id = id; }

  /**
   * @brief Set a unique ID for the element.
   * @param index a unique index to be set
   */
  template <TileType TType, ElementType EType>
  inline void setUniqueId(uint32_t index) {
    m_unique_id = getElementCode<TType, EType>() | index;
  }

  /**
   * @brief   Get the unique ID of the element.
   * @returns unique ID of the element
   */
  inline uint32_t getUniqueId(void) { return m_unique_id; }

public:
  int32_t m_id;        ///< Element ID
  int32_t m_unique_id; ///< Unique ID
};

/**
 * @class AieBD
 * @brief Class representing AIE-tile BD.
 *
 * This class implements Buffer descriptor for AIE-tile.
 */
class AieBD : public DmaElement {
public:
  /**
   * @brief Constructor.
   * @param id  BD index/identifier
   */
  AieBD(std::uint8_t id = 255) : DmaElement(id) { reset(); }

  /**
   * @brief Destructor.
   */
  ~AieBD() {
    // std::cout << "-- " << name() << ": destroyed!\n";
  }

  /**
   * @brief Reset all registers to the initial values.
   */
  inline void reset(void) {
    m_0.reg = 0;
    m_1.reg = 0;
    m_2.reg = 0;
    m_3.reg = 0;
    m_4.reg = 0;
    m_5.reg = 0;
    m_5.valid_bd = 1;
  }

  AieBDReg0 m_0;       ///< Register 0
  AieBDReg1 m_1;       ///< Register 1
  AieBDReg2 m_2;       ///< Register 2
  AieBDReg3 m_3;       ///< Register 3
  AieBDReg4 m_4;       ///< Register 4
  AieBDReg5 m_5;       ///< Register 5

  Location m_location; ///< Tile location
  uint32_t m_ncols;    ///< Number of columns
  uint32_t m_nrows;    ///< Number of rows

public:
  /**
   * @brief Method to accept a visitor to generate and write instructions.
   * @param writer        pointer to the visitor implementation
   * @param instr_buffer  pointer to memory where the instructions to be written
   * @return              number of instructions generated
   */
  int accept(InstructionWriterImpl* writer, uint32_t* instr_buffer) override {
    return writer->write(this, instr_buffer);
  }

  /**
   * @brief Create and returns a unique name for the BD.
   * @returns name of the BD as string
   */
  inline std::string name(void) {
    return std::string("AieBD_") + std::to_string(m_id);
  }

  /**
   * @brief   Reset BD registers to their initial values.
   * @returns pointer to itself
   */
  inline AieBD* resetBDs(void) {
    reset();
    return this;
  }

  /**
   * @brief   Update base address in BD register.
   * @param   addr  base address (byte address, not 32-bit word address)
   * @returns pointer to itself
   */
  inline AieBD* updateAddr(std::uint32_t addr) {
    assert(Sanity::AieTile::BD::checkBaseAddr(addr));
    m_0.base_addr = addr >> 2;
    return this;
  }

  /**
   * @brief   Update length in BD register.
   * @param   len Length in bytes (not 32-bit words)
   * @returns pointer to itself
   */
  inline AieBD* updateLength(std::uint32_t len) {
    assert(Sanity::AieTile::BD::checkLength(len));
    m_0.buffer_length = len >> 2;
    return this;
  }

  /**
   * @brief   Update Packet in BD register.
   * @param   type value of inserted packet type field in packet header
   * @param   id   value of inserted packet ID field in packet header
   * @returns pointer to itself
   */
  inline AieBD* updatePacket(std::uint32_t type, std::uint32_t id) {
    assert(Sanity::AieTile::BD::checkPacketType(type));
    assert(Sanity::AieTile::BD::checkPacketId(id));
    m_1.packet_type = type;
    m_1.packet_id = id;
    return this;
  }

  /**
   * @brief   Enables adding packet header at start of transfer.
   * @returns pointer to itself
   */
  inline AieBD* enablePacket(void) {
    m_1.packet_en = 1;
    return this;
  }

  /**
   * @brief   Enables Compression (MM2S), decompression (S2MM).
   *          Only effective if channel has (de)compression enabled.
   * @returns pointer to itself
   */
  inline AieBD* enableCompression(void) {
    m_1.compression_en = 1;
    return this;
  }

  /**
   * @brief   Update the Out of order BD ID
   * @param   bd_id value of Inserted Out of Order ID field in packet header
   * @returns pointer to itself
   */
  inline AieBD* updateOutOfOrderBD(std::uint32_t bd_id) {
    assert(Sanity::AieTile::BD::checkOutOfOrderBdId(bd_id));
    m_1.out_of_order_bd_id = bd_id;
    return this;
  }

  /**
   * @brief   Update Tensor dimentions in BD registers.
   * @param   tensor 4-dimential tensor represting shape as (step/wrap)
   * @returns pointer to itself
   * @see DmaTensor
   * @see DmaTensor4D
   */
  inline AieBD* updateTensorDims(const DmaTensor4D& tensor) {
    assert(Sanity::AieTile::BD::checkStep(tensor[0].m_step));
    assert(Sanity::AieTile::BD::checkStep(tensor[1].m_step));
    assert(Sanity::AieTile::BD::checkStep(tensor[2].m_step));
    assert(Sanity::AieTile::BD::checkWrap(tensor[0].m_wrap));
    assert(Sanity::AieTile::BD::checkWrap(tensor[1].m_wrap));
    m_2.d0_step_size = tensor[0].m_step - 1;
    m_2.d1_step_size = tensor[1].m_step - 1;
    m_3.d2_step_size = tensor[2].m_step - 1;
    m_3.d0_wrap = tensor[0].m_wrap;
    m_3.d1_wrap = tensor[1].m_wrap;
    return this;
  }

  /**
   * @brief   Update Iteration dimentions in BD registers.
   * @param   iter iteration step/wrap
   * @returns pointer to itself
   * @see DmaIteration
   */
  inline AieBD* updateIterationDims(const DmaIteration& iter) {
    assert(Sanity::AieTile::BD::checkIterStep(iter.m_step));
    assert(Sanity::AieTile::BD::checkIterWrap(iter.m_wrap));
    m_4.iter_step = iter.m_step - 1;
    m_4.iter_wrap = iter.m_wrap - 1;
    m_4.iter_curr = 0;
    return this;
  }

  /**
   * @brief   Update lock acquire and release settings.
   * @param   config  reference to LockConfig structure.
   * @returns pointer to itself
   * @see LockConfig
   */
  inline AieBD* updateLockConfigs(const LockConfig& config) {
    assert(Sanity::AieTile::BD::checkLockVal<4>(config.m_acq_val));
    assert(Sanity::AieTile::BD::checkLockVal<4>(config.m_rel_val));
    assert(Sanity::AieTile::BD::checkLockId(config.m_acq_id));
    assert(Sanity::AieTile::BD::checkLockId(config.m_rel_id));
    m_5.lock_acq_id = config.m_acq_id;
    m_5.lock_acq_val = config.m_acq_val;
    m_5.lock_acq_en = config.m_acq_en;
    m_5.lock_rel_id = config.m_rel_id;
    m_5.lock_rel_val = config.m_rel_val;
    return this;
  }

  /**
   * @brief   Enables use of next BD.
   * @returns pointer to itself
   */
  inline AieBD* enableNextBD(void) {
    m_5.use_next_bd = 1;
    return this;
  }

  /**
   * @brief   Enables TLAST suppress.
   *          MM2S channel: when set suppress assert of TLAST at the end of
   * transfer.
   * @returns pointer to itself
   */
  inline AieBD* enableTLASTSuppress(void) {
    m_5.tlast_suppress = 1;
    return this;
  }

  /**
   * @brief   Update next BD to be used.
   * @param   next_bd ID of next BD to be used
   * @returns pointer to itself
   */
  inline AieBD* updateNextBD(std::uint32_t next_bd) {
    assert(Sanity::AieTile::BD::checkNextBd(next_bd));
    m_5.next_bd = next_bd;
    return this;
  }

  /**
   * @brief Update the location, number of colums and rows info for the BD.
   * @param loc   Location of the tile
   * @param ncols number of columns
   * @param nrows number of rows
   */
  inline void write(Location& loc, std::uint8_t ncols, std::uint8_t nrows) {
    m_location = loc;
    m_ncols = ncols;
    m_nrows = nrows;
  }
};

/**
 * @class MemBD
 * @brief Class representing Mem-tile BD.
 *
 * This class implements Buffer descriptor for Mem-tile.
 */
class MemBD : public DmaElement {
public:
  /**
   * @brief Constructor.
   * @param id  BD index/identifier
   */
  MemBD(uint8_t id = 255) : DmaElement(id) { reset(); }

  /**
   * @brief Destructor.
   */
  ~MemBD() {
    // std::cout << "-- " << name() << ": destroyed!\n";
  }

  /**
   * @brief Reset all registers to the initial values.
   */
  inline void reset(void) {
    m_0.reg = 0;
    m_1.reg = 0;
    m_2.reg = 0;
    m_3.reg = 0;
    m_4.reg = 0;
    m_5.reg = 0;
    m_6.reg = 0;
    m_7.reg = 0;
    m_7.valid_bd = 1;
  }

  // BD Registers
  MemBDReg0 m_0;       ///< Register 0
  MemBDReg1 m_1;       ///< Register 1
  MemBDReg2 m_2;       ///< Register 2
  MemBDReg3 m_3;       ///< Register 3
  MemBDReg4 m_4;       ///< Register 4
  MemBDReg5 m_5;       ///< Register 5
  MemBDReg6 m_6;       ///< Register 6
  MemBDReg7 m_7;       ///< Register 7

  Location m_location; ///< Tile location
  uint32_t m_ncols;    ///< Number of columns

public:
  /**
   * @brief Method to accept a visitor to generate and write instructions.
   * @param writer        pointer to the visitor implementation
   * @param instr_buffer  pointer to memory where the instructions to be written
   * @return              number of instructions generated
   */
  int accept(InstructionWriterImpl* writer, uint32_t* instr_buffer) override {
    return writer->write(this, instr_buffer);
  }

  /**
   * @brief Create and returns a unique name for the BD.
   * @returns name of the BD as string
   */
  inline std::string name(void) {
    return std::string("MemBD_") + std::to_string(m_id);
  }

  /**
   * @brief   Reset BD registers to their initial values.
   * @returns pointer to itself
   */
  inline MemBD* resetBDs(void) {
    reset();
    return this;
  }

  /**
   * @brief   Update base address in BD register.
   * @param   addr  base address (byte address, not 32-bit word address)
   * @returns pointer to itself
   */
  inline MemBD* updateAddr(std::uint32_t addr) {
    assert(Sanity::MemTile::BD::checkBaseAddr(addr));
    m_1.base_addr = addr >> 2;
    return this;
  }

  /**
   * @brief   Update length in BD register.
   * @param   len Length in bytes (not 32-bit words)
   * @returns pointer to itself
   */
  inline MemBD* updateLength(std::uint32_t len) {
    assert(Sanity::MemTile::BD::checkLength(len));
    m_0.buffer_length = len >> 2;
    return this;
  }

  /**
   * @brief   Update Packet in BD register.
   * @param   type value of inserted packet type field in packet header
   * @param   id   value of inserted packet ID field in packet header
   * @returns pointer to itself
   */
  inline MemBD* updatePacket(std::uint32_t type, std::uint32_t id) {
    assert(Sanity::MemTile::BD::checkPacketType(type));
    assert(Sanity::MemTile::BD::checkPacketId(id));
    m_0.packet_type = type;
    m_0.packet_id = id;
    return this;
  }

  /**
   * @brief   Enables adding packet header at start of transfer.
   * @returns pointer to itself
   */
  inline MemBD* enablePacket(void) {
    m_0.packet_en = 1;
    return this;
  }

  /**
   * @brief   Enables Compression (MM2S), decompression (S2MM).
   *          Only effective if channel has (de)compression enabled.
   * @returns pointer to itself
   */
  inline MemBD* enableCompression(void) {
    m_4.compression_en = 1;
    return this;
  }

  /**
   * @brief   Update the Out of order BD ID
   * @param   bd_id value of Inserted Out of Order ID field in packet header
   * @returns pointer to itself
   */
  inline MemBD* updateOutOfOrderBD(std::uint32_t bd_id) {
    assert(Sanity::MemTile::BD::checkOutOfOrderBdId(bd_id));
    m_0.out_of_order_bd_id = bd_id;
    return this;
  }

  /**
   * @brief   Update Tensor dimentions in BD registers.
   * @param   tensor 4-dimential tensor represting shape as (step/wrap)
   * @returns pointer to itself
   * @see DmaTensor
   * @see DmaTensor4D
   */
  inline MemBD* updateTensorDims(const DmaTensor4D& tensor) {
    assert(Sanity::MemTile::BD::checkStep(tensor[0].m_step));
    assert(Sanity::MemTile::BD::checkStep(tensor[1].m_step));
    assert(Sanity::MemTile::BD::checkStep(tensor[2].m_step));
    assert(Sanity::MemTile::BD::checkStep(tensor[3].m_step));
    assert(Sanity::MemTile::BD::checkWrap(tensor[0].m_wrap));
    assert(Sanity::MemTile::BD::checkWrap(tensor[1].m_wrap));
    assert(Sanity::MemTile::BD::checkWrap(tensor[2].m_wrap));
    m_2.d0_step_size = tensor[0].m_step - 1;
    m_3.d1_step_size = tensor[1].m_step - 1;
    m_4.d2_step_size = tensor[2].m_step - 1;
    m_5.d3_step_size = tensor[3].m_step - 1;
    m_2.d0_wrap = tensor[0].m_wrap;
    m_3.d1_wrap = tensor[1].m_wrap;
    m_4.d2_wrap = tensor[2].m_wrap;
    return this;
  }

  /**
   * @brief   Update Iteration dimentions in BD registers.
   * @param   iter iteration step/wrap
   * @returns pointer to itself
   * @see DmaIteration
   */
  inline MemBD* updateIterationDims(const DmaIteration& iter) {
    assert(Sanity::MemTile::BD::checkIterStep(iter.m_step));
    assert(Sanity::MemTile::BD::checkIterWrap(iter.m_wrap));
    m_6.iter_step = iter.m_step - 1;
    m_6.iter_wrap = iter.m_wrap - 1;
    m_6.iter_curr = 0;
    return this;
  }

  /**
   * @brief   Update lock acquire and release settings.
   * @param   config  reference to LockConfig structure.
   * @returns pointer to itself
   * @see LockConfig
   */
  inline MemBD* updateLockConfigs(const LockConfig& config) {
    assert(Sanity::MemTile::BD::checkLockVal<4>(config.m_acq_val));
    assert(Sanity::MemTile::BD::checkLockVal<4>(config.m_rel_val));
    assert(Sanity::MemTile::BD::checkLockId(config.m_acq_id));
    assert(Sanity::MemTile::BD::checkLockId(config.m_rel_id));
    m_7.lock_acq_id = config.m_acq_id;
    m_7.lock_acq_val = config.m_acq_val;
    m_7.lock_acq_en = config.m_acq_en;
    m_7.lock_rel_id = config.m_rel_id;
    m_7.lock_rel_val = config.m_rel_val;
    return this;
  }

  /**
   * @brief   Enables use of next BD.
   * @returns pointer to itself
   */
  inline MemBD* enableNextBD(void) {
    m_1.use_next_bd = 1;
    return this;
  }

  /**
   * @brief   Enables TLAST suppress.
   *          MM2S channel: when set suppress assert of TLAST at the end of
   * transfer.
   * @returns pointer to itself
   */
  inline MemBD* enableTLASTSuppress(void) {
    m_2.tlast_suppress = 1;
    return this;
  }

  /**
   * @brief   Update next BD to be used.
   * @param   next_bd ID of next BD to be used
   * @returns pointer to itself
   */
  inline MemBD* updateNextBD(std::uint32_t next_bd) {
    assert(Sanity::MemTile::BD::checkNextBd(next_bd));
    m_1.next_bd = next_bd;
    return this;
  }

  /**
   * @brief Update the location, number of colums and rows info for the BD.
   * @param loc   Location of the tile
   * @param ncols number of columns
   */
  inline void write(Location& loc, std::uint8_t ncols) {
    m_location = loc;
    m_ncols = ncols;
  }
};

/**
 * @class NoCBD
 * @brief Class representing NoC-tile BD.
 *
 * This class implements Buffer descriptor for NoC-tile.
 */
class NoCBD : public DmaElement {
public:
  /**
   * @brief Constructor.
   * @param id  BD index/identifier
   */
  NoCBD(uint8_t id = 255) : DmaElement(id) { reset(); }

  /**
   * @brief Destructor.
   */
  ~NoCBD() {
    // std::cout << "-- " << name() << ": destroyed!\n";
  }

  /**
   * @brief Reset all registers to the initial values.
   */
  inline void reset(void) {
    m_0.reg = 0;
    m_1.reg = 0;
    m_2.reg = 0;
    m_3.reg = 0;
    m_4.reg = 0;
    m_5.reg = 0;
    m_6.reg = 0;
    m_7.reg = 0;
    m_4.burst_length = 2;
    m_7.valid_bd = 1;
  }

  // BD Registers
  NoCBDReg0 m_0;       ///< Register 0
  NoCBDReg1 m_1;       ///< Register 1
  NoCBDReg2 m_2;       ///< Register 2
  NoCBDReg3 m_3;       ///< Register 3
  NoCBDReg4 m_4;       ///< Register 4
  NoCBDReg5 m_5;       ///< Register 5
  NoCBDReg6 m_6;       ///< Register 6
  NoCBDReg7 m_7;       ///< Register 7

  Location m_location; ///< Tile location
  uint32_t m_ncols;    ///< Number of columns
  uint32_t m_ddrtype;  ///< DDR type (relates to base addr and BO matching)

public:
  /**
   * @brief Method to accept a visitor to generate and write instructions.
   * @param writer        pointer to the visitor implementation
   * @param instr_buffer  pointer to memory where the instructions to be written
   * @return              number of instructions generated
   */
  int accept(InstructionWriterImpl* writer, uint32_t* instr_buffer) override {
    return writer->write(this, instr_buffer);
  }

  /**
   * @brief Create and returns a unique name for the BD.
   * @returns name of the BD as string
   */
  inline std::string name(void) {
    return std::string("NoCBD_") + std::to_string(m_id);
  }

  /**
   * @brief   Reset BD registers to their initial values.
   * @returns pointer to itself
   */
  inline NoCBD* resetBDs(void) {
    reset();
    return this;
  }

  /**
   * @brief   Update base address in BD register.
   * @param   addrlow  base address low (byte address, not 32-bit word address)
   * @param   addrhigh base address low (byte address, not 32-bit word address)
   * @returns pointer to itself
   */
  inline NoCBD* updateAddr(std::uint32_t addrlow, std::uint32_t addrhigh) {
    assert(Sanity::NoCTile::BD::checkBaseAddrLow(addrlow));
    assert(Sanity::NoCTile::BD::checkBaseAddrHigh(addrhigh));
    m_1.base_addr_low = addrlow >> 2;
    m_2.base_addr_high = addrhigh >> 2;
    return this;
  }

  /**
   * @brief   Update length in BD register.
   * @param   len Length in bytes (not 32-bit words)
   * @returns pointer to itself
   */
  inline NoCBD* updateLength(std::uint32_t len) {
    assert(Sanity::NoCTile::BD::checkLength(len));
    m_0.buffer_length = len >> 2;
    return this;
  }

  /**
   * @brief   Update Packet in BD register.
   * @param   type value of inserted packet type field in packet header
   * @param   id   value of inserted packet ID field in packet header
   * @returns pointer to itself
   */
  inline NoCBD* updatePacket(std::uint32_t type, std::uint32_t id) {
    assert(Sanity::NoCTile::BD::checkPacketType(type));
    assert(Sanity::NoCTile::BD::checkPacketId(id));
    m_2.packet_type = type;
    m_2.packet_id = id;
    return this;
  }

  /**
   * @brief   Enables adding packet header at start of transfer.
   * @returns pointer to itself
   */
  inline NoCBD* enablePacket(void) {
    m_2.packet_en = 1;
    return this;
  }

  /**
   * @brief   Update the Out of order BD ID
   * @param   bd_id value of Inserted Out of Order ID field in packet header
   * @returns pointer to itself
   */
  inline NoCBD* updateOutOfOrderBD(std::uint32_t bd_id) {
    assert(Sanity::NoCTile::BD::checkOutOfOrderBdId(bd_id));
    m_2.out_of_order_bd_id = bd_id;
    return this;
  }

  /**
   * @brief   Update Tensor dimentions in BD registers.
   * @param   tensor 4-dimential tensor represting shape as (step/wrap)
   * @returns pointer to itself
   * @see DmaTensor
   * @see DmaTensor4D
   */
  inline NoCBD* updateTensorDims(const DmaTensor4D& tensor) {
    assert(Sanity::NoCTile::BD::checkStep(tensor[0].m_step));
    assert(Sanity::NoCTile::BD::checkStep(tensor[1].m_step));
    assert(Sanity::NoCTile::BD::checkStep(tensor[2].m_step));
    assert(Sanity::NoCTile::BD::checkWrap(tensor[0].m_wrap));
    assert(Sanity::NoCTile::BD::checkWrap(tensor[1].m_wrap));
    m_3.d0_step_size = tensor[0].m_step - 1;
    m_4.d1_step_size = tensor[1].m_step - 1;
    m_5.d2_step_size = tensor[2].m_step - 1;
    m_3.d0_wrap = tensor[0].m_wrap;
    m_4.d1_wrap = tensor[1].m_wrap;
    return this;
  }

  /**
   * @brief   Update Iteration dimentions in BD registers.
   * @param   iter iteration step/wrap
   * @returns pointer to itself
   * @see DmaIteration
   */
  inline NoCBD* updateIterationDims(const DmaIteration& iter) {
    assert(Sanity::NoCTile::BD::checkIterStep(iter.m_step));
    assert(Sanity::NoCTile::BD::checkIterWrap(iter.m_wrap));
    m_6.iter_step = iter.m_step - 1;
    m_6.iter_wrap = iter.m_wrap - 1;
    m_6.iter_curr = 0;
    return this;
  }

  /**
   * @brief   Update lock acquire and release settings.
   * @param   config  reference to LockConfig structure.
   * @returns pointer to itself
   * @see LockConfig
   */
  inline NoCBD* updateLockConfigs(const LockConfig& config) {
    assert(Sanity::NoCTile::BD::checkLockVal<4>(config.m_acq_val));
    assert(Sanity::NoCTile::BD::checkLockVal<4>(config.m_rel_val));
    assert(Sanity::NoCTile::BD::checkLockId(config.m_acq_id));
    assert(Sanity::NoCTile::BD::checkLockId(config.m_rel_id));
    m_7.lock_acq_id = config.m_acq_id;
    m_7.lock_acq_val = config.m_acq_val;
    m_7.lock_acq_en = config.m_acq_en;
    m_7.lock_rel_id = config.m_rel_id;
    m_7.lock_rel_val = config.m_rel_val;
    return this;
  }

  /**
   * @brief   Enables use of next BD.
   * @returns pointer to itself
   */
  inline NoCBD* enableNextBD(void) {
    m_7.use_next_bd = 1;
    return this;
  }

  /**
   * @brief   Enables TLAST suppress.
   *          MM2S channel: when set suppress assert of TLAST at the end of
   * transfer.
   * @returns pointer to itself
   */
  inline NoCBD* enableTLASTSuppress(void) {
    m_7.tlast_suppress = 1;
    return this;
  }

  /**
   * @brief   Update next BD to be used.
   * @param   next_bd ID of next BD to be used
   * @returns pointer to itself
   */
  inline NoCBD* updateNextBD(std::uint32_t next_bd) {
    assert(Sanity::NoCTile::BD::checkNextBd(next_bd));
    m_7.next_bd = next_bd;
    return this;
  }

  /**
   * @brief Update the location, number of colums and rows info for the BD.
   * @param loc       location of the tile
   * @param ncols     number of columns
   * @param ddrtype   DDR type (ID of buffer object)
   * @see Location
   */
  inline void write(Location& loc, std::uint32_t ncols, std::uint8_t ddrtype) {
    m_location = loc;
    m_ncols = ncols;
    m_ddrtype = ddrtype;
  }
};

/**
 * @class DmaChannel
 * @brief Class representing a DMA Channel.
 */
template <class Derived> class DmaChannel : public DmaElement {
public:
  /**
   * @brief Constructor.
   * @param id  ID of the channel
   */
  DmaChannel(uint8_t id = 0) : DmaElement(id), m_addr(0) {}

  /**
   * @brief Destructor.
   */
  ~DmaChannel() {
    // std::cout << "-- DmaChannel base - destroyed\n";
  }

  /**
   * @brief Update location of the tile.
   * @param loc tile location (col, row)
   * @see Location
   */
  void write(Location& loc) { m_location = loc; }

  /**
   * @brief Accept a visitor to generate instructions for this element.
   * @param   writer        pointer to a visitor instruction generator
   * @param   instr_buffer  pointer to memory where the instruction to be
   * written
   * @return              number of instructions generated
   */
  int accept(InstructionWriterImpl* writer, uint32_t* instr_buffer) override {
    return writer->write(static_cast<Derived*>(this), instr_buffer);
  }

public:
  // uint8_t  m_id;        ///< ID of the channel element
  uint32_t m_addr;     ///< Address of channel element
  Location m_location; ///< Tile location
};

/**
 * @class AieDmaS2MMCtrl
 * @brief Class representing AIE-tile DMA Channel S2MM control.
 */
class AieDmaS2MMCtrl : public DmaChannel<AieDmaS2MMCtrl> {
public:
  /**
   * @brief Constructor.
   * @param id  ID/index of the channel control
   */
  AieDmaS2MMCtrl(uint8_t id = 0)
      : DmaChannel(id), m_base(ATChannelInfo::ATChCtrlBase),
        m_offset(ATChannelInfo::ATChIdxOffset) {
    m_regval.reg = 0;
  }

public:
  const uint32_t m_base;     ///< Channel control base addr
  const uint32_t m_offset;   ///< Channel control offset
  AieDmaChCtrlS2MM m_regval; ///< Channel control register value
public:
  /**
   * @brief   Creates a unique name for the element.
   * @returns returns string with a unique name
   */
  inline std::string name(void) {
    return std::string("AieDmaS2MMCtrl_") + std::to_string(m_id);
  }

  /**
   * @brief   Resets DMA channel.
   * @returns pointer to itself
   */
  inline AieDmaS2MMCtrl* reset(void) {
    m_regval.reg = 2;
    return this;
  }

  /**
   * @brief   Un-resets DMA channel.
   * @returns pointer to itself
   */
  inline AieDmaS2MMCtrl* unreset(void) {
    m_regval.reg = 0;
    return this;
  }

  /**
   * @brief   Sets controller ID.
   * @param   id  controller ID
   * @returns pointer to itself
   */
  inline AieDmaS2MMCtrl* setControllerId(std::uint8_t id) {
    m_regval.controller_id = id;
    return this;
  }

  /**
   * @brief   Enables out of order mode.
   * @returns pointer to itself
   */
  inline AieDmaS2MMCtrl* enableOutOfOrder(void) {
    m_regval.out_of_order_en = 1;
    return this;
  }

  /**
   * @brief   Enables decompression.
   * @returns pointer to itself
   */
  inline AieDmaS2MMCtrl* enableDecompression(void) {
    m_regval.decompression_en = 1;
    return this;
  }

  /**
   * @brief   Set FoT mode.
   * @param   mode FoT mode, see mode options:
   *          [0] FoT disabled, [1] FoT no counts
   *          [2] FoT counts with task tokens
   *          [3] FoT counts from MM register
   * @returns pointer to itself
   */
  inline AieDmaS2MMCtrl* setFoTMode(std::uint8_t mode) {
    m_regval.fot_mode = mode;
    return this;
  }
};

/**
 * @class AieDmaS2MMQueue
 * @brief Class representing AIE-tile DMA Channel S2MM start queue.
 */
class AieDmaS2MMQueue : public DmaChannel<AieDmaS2MMQueue> {
public:
  /**
   * @brief     Constructor.
   * @param id  ID/index of the channel start queue
   */
  AieDmaS2MMQueue(uint8_t id = 0)
      : DmaChannel(id), m_base(ATChannelInfo::ATChQueueBase),
        m_offset(ATChannelInfo::ATChIdxOffset) {
    m_regval.reg = 0;
  }

public:
  const uint32_t m_base;      ///< Channel start queue base addr
  const uint32_t m_offset;    ///< Channel start queue offset
  AieDmaChQueueS2MM m_regval; ///< Channel start queue register value
public:
  /**
   * @brief   Creates a unique name for the element.
   * @returns returns string with a unique name
   */
  inline std::string name(void) {
    return std::string("AieDmaS2MMQueue_") + std::to_string(m_id);
  }

  /**
   * @brief   Sets start BD ID.
   * @param   bd  BD id to start the queue
   * @returns pointer to itself
   */
  inline AieDmaS2MMQueue* setStartBd(std::uint8_t bd) {
    assert(Sanity::AieTile::Queue::checkStartBd(bd));
    m_regval.start_bd_id = bd;
    return this;
  }

  /**
   * @brief   Sets BD repeat count.
   * @param   count repeat count
   * @returns pointer to itself
   */
  inline AieDmaS2MMQueue* setRepeatCount(std::uint8_t count) {
    assert(Sanity::AieTile::Queue::checkRepeatCount(count));
    m_regval.repeat = count - 1;
    return this;
  }

  /**
   * @brief   Enables token issue.
   * @returns pointer to itself
   */
  inline AieDmaS2MMQueue* enableTokenIssue(void) {
    m_regval.token_issue_en = 1;
    return this;
  }
};

/**
 * @class AieDmaMM2SCtrl
 * @brief Class representing AIE-tile DMA Channel MM2S control.
 */
class AieDmaMM2SCtrl : public DmaChannel<AieDmaMM2SCtrl> {
public:
  /**
   * @brief Constructor.
   * @param id  ID/index of the channel control
   */
  AieDmaMM2SCtrl(uint8_t id = 0)
      : DmaChannel(id),
        m_base(ATChannelInfo::ATChCtrlBase +
               (ATChannelInfo::ATChIdxOffset * ATChannelInfo::ATNumChannels)),
        m_offset(ATChannelInfo::ATChIdxOffset) {
    m_regval.reg = 0;
  }

public:
  const uint32_t m_base;     ///< Channel control base addr
  const uint32_t m_offset;   ///< Channel control offset
  AieDmaChCtrlMM2S m_regval; ///< Channel control register value
public:
  /**
   * @brief   Creates a unique name for the element.
   * @returns returns string with a unique name
   */
  inline std::string name(void) {
    return std::string("AieDmaMM2SCtrl_") + std::to_string(m_id);
  }

  /**
   * @brief   Resets DMA channel.
   * @returns pointer to itself
   */
  inline AieDmaMM2SCtrl* reset(void) {
    m_regval.reg = 2;
    return this;
  }

  /**
   * @brief   Un-resets DMA channel.
   * @returns pointer to itself
   */
  inline AieDmaMM2SCtrl* unreset(void) {
    m_regval.reg = 0;
    return this;
  }

  /**
   * @brief   Sets controller ID.
   * @param   id  controller ID
   * @returns pointer to itself
   */
  inline AieDmaMM2SCtrl* setControllerId(std::uint8_t id) {
    m_regval.controller_id = id;
    return this;
  }

  /**
   * @brief   Enables compression.
   * @returns pointer to itself
   */
  inline AieDmaMM2SCtrl* enableCompression(void) {
    m_regval.compression_en = 1;
    return this;
  }
};

/**
 * @class AieDmaMM2SQueue
 * @brief Class representing AIE-tile DMA Channel MM2S start queue.
 */
class AieDmaMM2SQueue : public DmaChannel<AieDmaMM2SQueue> {
public:
  /**
   * @brief Constructor.
   * @param id  ID/index of the channel start queue
   */
  AieDmaMM2SQueue(uint8_t id = 0)
      : DmaChannel(id),
        m_base(ATChannelInfo::ATChQueueBase +
               (ATChannelInfo::ATChIdxOffset * ATChannelInfo::ATNumChannels)),
        m_offset(ATChannelInfo::ATChIdxOffset) {
    m_regval.reg = 0;
  }

public:
  const uint32_t m_base;      ///< Channel start queue base addr
  const uint32_t m_offset;    ///< Channel start queue offset
  AieDmaChQueueMM2S m_regval; ///< Channel start queue register value
public:
  /**
   * @brief   Creates a unique name for the element.
   * @returns returns string with a unique name
   */
  inline std::string name(void) {
    return std::string("AieDmaMM2SQueue_") + std::to_string(m_id);
  }

  /**
   * @brief   Sets start BD ID.
   * @param   bd  BD id to start the queue
   * @returns pointer to itself
   */
  inline AieDmaMM2SQueue* setStartBd(std::uint8_t bd) {
    assert(Sanity::AieTile::Queue::checkStartBd(bd));
    m_regval.start_bd_id = bd;
    return this;
  }

  /**
   * @brief   Sets BD repeat count.
   * @param   count repeat count
   * @returns pointer to itself
   */
  inline AieDmaMM2SQueue* setRepeatCount(std::uint8_t count) {
    assert(Sanity::AieTile::Queue::checkRepeatCount(count));
    m_regval.repeat = count - 1;
    return this;
  }

  /**
   * @brief   Enables token issue.
   * @returns pointer to itself
   */
  inline AieDmaMM2SQueue* enableTokenIssue(void) {
    m_regval.token_issue_en = 1;
    return this;
  }
};

/**
 * @class MemDmaS2MMCtrl
 * @brief Class representing Mem-tile DMA Channel S2MM control.
 */
class MemDmaS2MMCtrl : public DmaChannel<MemDmaS2MMCtrl> {
public:
  /**
   * @brief Constructor.
   * @param id  ID/index of the channel control
   */
  MemDmaS2MMCtrl(uint8_t id = 0)
      : DmaChannel(id), m_base(MTChannelInfo::MTChCtrlBase),
        m_offset(MTChannelInfo::MTChIdxOffset) {
    m_regval.reg = 0;
  }

public:
  const uint32_t m_base;     ///< Channel control base addr
  const uint32_t m_offset;   ///< Channel control offset
  MemDmaChCtrlS2MM m_regval; ///< Channel control register value
public:
  /**
   * @brief   Creates a unique name for the element.
   * @returns returns string with a unique name
   */
  inline std::string name(void) {
    return std::string("MemDmaS2MMCtrl_") + std::to_string(m_id);
  }

  /**
   * @brief   Resets DMA channel.
   * @returns pointer to itself
   */
  inline MemDmaS2MMCtrl* reset(void) {
    m_regval.reg = 2;
    return this;
  }

  /**
   * @brief   Un-resets DMA channel.
   * @returns pointer to itself
   */
  inline MemDmaS2MMCtrl* unreset(void) {
    m_regval.reg = 0;
    return this;
  }

  /**
   * @brief   Sets controller ID.
   * @param   id  controller ID
   * @returns pointer to itself
   */
  inline MemDmaS2MMCtrl* setControllerId(std::uint8_t id) {
    m_regval.controller_id = id;
    return this;
  }

  /**
   * @brief   Enables out of order mode.
   * @returns pointer to itself
   */
  inline MemDmaS2MMCtrl* enableOutOfOrder(void) {
    m_regval.out_of_order_en = 1;
    return this;
  }

  /**
   * @brief   Enables decompression.
   * @returns pointer to itself
   */
  inline MemDmaS2MMCtrl* enableDecompression(void) {
    m_regval.decompression_en = 1;
    return this;
  }

  /**
   * @brief   Sets FoT mode.
   * @param   mode FoT mode, see mode options:
   *          [0] FoT disabled, [1] FoT no counts
   *          [2] FoT counts with task tokens
   *          [3] FoT counts from MM register
   * @returns pointer to itself
   */
  inline MemDmaS2MMCtrl* setFoTMode(std::uint8_t mode) {
    m_regval.fot_mode = mode;
    return this;
  }
};

/**
 * @class MemDmaS2MMQueue
 * @brief Class representing Mem-tile DMA Channel S2MM start queue.
 */
class MemDmaS2MMQueue : public DmaChannel<MemDmaS2MMQueue> {
public:
  /**
   * @brief     Constructor.
   * @param id  ID/index of the channel start queue
   */
  MemDmaS2MMQueue(uint8_t id = 0)
      : DmaChannel(id), m_base(MTChannelInfo::MTChQueueBase),
        m_offset(MTChannelInfo::MTChIdxOffset) {
    m_regval.reg = 0;
  }

public:
  const uint32_t m_base;      ///< Channel start queue base addr
  const uint32_t m_offset;    ///< Channel start queue offset
  MemDmaChQueueS2MM m_regval; ///< Channel start queue register value
public:
  /**
   * @brief   Creates a unique name for the element.
   * @returns returns string with a unique name
   */
  inline std::string name(void) {
    return std::string("MemDmaS2MMQueue_") + std::to_string(m_id);
  }

  /**
   * @brief   Sets start BD ID.
   * @param   bd  BD id to start the queue
   * @returns pointer to itself
   */
  inline MemDmaS2MMQueue* setStartBd(std::uint8_t bd) {
    assert(Sanity::MemTile::Queue::checkStartBd(bd));
    m_regval.start_bd_id = bd;
    return this;
  }

  /**
   * @brief   Sets BD repeat count.
   * @param   count repeat count
   * @returns pointer to itself
   */
  inline MemDmaS2MMQueue* setRepeatCount(std::uint8_t count) {
    assert(Sanity::MemTile::Queue::checkRepeatCount(count));
    m_regval.repeat = count - 1;
    return this;
  }

  /**
   * @brief   Enables token issue.
   * @returns pointer to itself
   */
  inline MemDmaS2MMQueue* enableTokenIssue(void) {
    m_regval.token_issue_en = 1;
    return this;
  }
};

/**
 * @class MemDmaMM2SCtrl
 * @brief Class representing Mem-tile DMA Channel MM2S control.
 */
class MemDmaMM2SCtrl : public DmaChannel<MemDmaMM2SCtrl> {
public:
  /**
   * @brief Constructor.
   * @param id  ID/index of the channel control
   */
  MemDmaMM2SCtrl(uint8_t id = 0)
      : DmaChannel(id),
        m_base(MTChannelInfo::MTChCtrlBase +
               (MTChannelInfo::MTChIdxOffset * MTChannelInfo::MTNumChannels)),
        m_offset(MTChannelInfo::MTChIdxOffset) {
    m_regval.reg = 0;
  }

public:
  const uint32_t m_base;     ///< Channel control base addr
  const uint32_t m_offset;   ///< Channel control offset
  MemDmaChCtrlMM2S m_regval; ///< Channel control register value
public:
  /**
   * @brief   Creates a unique name for the element.
   * @returns returns string with a unique name
   */
  inline std::string name(void) {
    return std::string("MemDmaMM2SCtrl_") + std::to_string(m_id);
  }

  /**
   * @brief   Resets DMA channel.
   * @returns pointer to itself
   */
  inline MemDmaMM2SCtrl* reset(void) {
    m_regval.reg = 2;
    return this;
  }

  /**
   * @brief   Un-resets DMA channel.
   * @returns pointer to itself
   */
  inline MemDmaMM2SCtrl* unreset(void) {
    m_regval.reg = 0;
    return this;
  }

  /**
   * @brief   Sets controller ID.
   * @param   id  controller ID
   * @returns pointer to itself
   */
  inline MemDmaMM2SCtrl* setControllerId(std::uint8_t id) {
    m_regval.controller_id = id;
    return this;
  }

  /**
   * @brief   Enables compression.
   * @returns pointer to itself
   */
  inline MemDmaMM2SCtrl* enableCompression(void) {
    m_regval.compression_en = 1;
    return this;
  }
};

/**
 * @class MemDmaMM2SQueue
 * @brief Class representing Mem-tile DMA Channel MM2S start queue.
 */
class MemDmaMM2SQueue : public DmaChannel<MemDmaMM2SQueue> {
public:
  /**
   * @brief Constructor.
   * @param id  ID/index of the channel start queue
   */
  MemDmaMM2SQueue(uint8_t id = 0)
      : DmaChannel(id),
        m_base(MTChannelInfo::MTChQueueBase +
               (MTChannelInfo::MTChIdxOffset * MTChannelInfo::MTNumChannels)),
        m_offset(MTChannelInfo::MTChIdxOffset) {
    m_regval.reg = 0;
  }

public:
  const uint32_t m_base;      ///< Channel start queue base addr
  const uint32_t m_offset;    ///< Channel start queue offset
  MemDmaChQueueMM2S m_regval; ///< Channel start queue register value
public:
  /**
   * @brief   Creates a unique name for the element.
   * @returns returns string with a unique name
   */
  inline std::string name(void) {
    return std::string("MemDmaMM2SQueue_") + std::to_string(m_id);
  }

  /**
   * @brief   Sets start BD ID.
   * @param   bd  BD id to start the queue
   * @returns pointer to itself
   */
  inline MemDmaMM2SQueue* setStartBd(std::uint8_t bd) {
    assert(Sanity::MemTile::Queue::checkStartBd(bd));
    m_regval.start_bd_id = bd;
    return this;
  }

  /**
   * @brief   Sets BD repeat count.
   * @param   count repeat count
   * @returns pointer to itself
   */
  inline MemDmaMM2SQueue* setRepeatCount(std::uint8_t count) {
    assert(Sanity::MemTile::Queue::checkRepeatCount(count));
    m_regval.repeat = count - 1;
    return this;
  }

  /**
   * @brief   Enables token issue.
   * @returns pointer to itself
   */
  inline MemDmaMM2SQueue* enableTokenIssue(void) {
    m_regval.token_issue_en = 1;
    return this;
  }
};

/**
 * @class NoCDmaS2MMCtrl
 * @brief Class representing NoC-tile DMA Channel S2MM control.
 */
class NoCDmaS2MMCtrl : public DmaChannel<NoCDmaS2MMCtrl> {
public:
  /**
   * @brief Constructor.
   * @param id  ID/index of the channel control
   */
  NoCDmaS2MMCtrl(uint8_t id = 0)
      : DmaChannel(id), m_base(ShimChannelInfo::ShimChCtrlBase),
        m_offset(ShimChannelInfo::ShimChIdxOffset) {
    m_regval.reg = 0;
  }

public:
  const uint32_t m_base;
  const uint32_t m_offset;
  NoCDmaChCtrlS2MM m_regval;

public:
  /**
   * @brief   Creates a unique name for the element.
   * @returns returns string with a unique name
   */
  inline std::string name(void) {
    return std::string("NoCDmaS2MMCtrl_") + std::to_string(m_id);
  }

  /**
   * @brief   When set, pauses the issuing of new AXI-MM commands.
   * @returns pointer to itself
   */
  inline NoCDmaS2MMCtrl* setPauseMem(void) {
    m_regval.pause_mem = 1;
    return this;
  }

  /**
   * @brief   When set, pauses the stream traffic.
   * @returns pointer to itself
   */
  inline NoCDmaS2MMCtrl* setPauseStream(void) {
    m_regval.pause_stream = 1;
    return this;
  }

  /**
   * @brief   Enables out of order mode.
   * @returns pointer to itself
   */
  inline NoCDmaS2MMCtrl* enableOutOfOrder(void) {
    m_regval.out_of_order_en = 1;
    return this;
  }

  /**
   * @brief   Sets controller ID.
   * @param   id  controller ID
   * @returns pointer to itself
   */
  inline NoCDmaS2MMCtrl* setControllerId(std::uint8_t id) {
    m_regval.controller_id = id;
    return this;
  }

  /**
   * @brief   Sets FoT mode.
   * @param   mode FoT mode, see mode options:
   *          [0] FoT disabled, [1] FoT no counts
   *          [2] FoT counts with task tokens
   *          [3] FoT counts from MM register
   * @returns pointer to itself
   */
  inline NoCDmaS2MMCtrl* setFoTMode(std::uint8_t mode) {
    m_regval.fot_mode = mode;
    return this;
  }
};

/**
 * @class NoCDmaS2MMQueue
 * @brief Class representing NoC-tile DMA Channel S2MM start queue.
 */
class NoCDmaS2MMQueue : public DmaChannel<NoCDmaS2MMQueue> {
public:
  /**
   * @brief     Constructor.
   * @param id  ID/index of the channel start queue
   */
  NoCDmaS2MMQueue(uint8_t id = 0)
      : DmaChannel(id), m_base(ShimChannelInfo::ShimChQueueBase),
        m_offset(ShimChannelInfo::ShimChIdxOffset) {
    m_regval.reg = 0;
  }

public:
  const uint32_t m_base;      ///< Channel start queue base addr
  const uint32_t m_offset;    ///< Channel start queue offset
  NoCDmaChQueueS2MM m_regval; ///< Channel start queue register value
public:
  /**
   * @brief   Creates a unique name for the element.
   * @returns returns string with a unique name
   */
  inline std::string name(void) {
    return std::string("NoCDmaS2MMQueue_") + std::to_string(m_id);
  }

  /**
   * @brief   Sets start BD ID.
   * @param   bd  BD id to start the queue
   * @returns pointer to itself
   */
  inline NoCDmaS2MMQueue* setStartBd(std::uint8_t bd) {
    assert(Sanity::NoCTile::Queue::checkStartBd(bd));
    m_regval.start_bd_id = bd;
    return this;
  }

  /**
   * @brief   Sets BD repeat count.
   * @param   count repeat count
   * @returns pointer to itself
   */
  inline NoCDmaS2MMQueue* setRepeatCount(std::uint8_t count) {
    assert(Sanity::NoCTile::Queue::checkRepeatCount(count));
    m_regval.repeat = count - 1;
    return this;
  }

  /**
   * @brief   Enables token issue.
   * @returns pointer to itself
   */
  inline NoCDmaS2MMQueue* enableTokenIssue(void) {
    m_regval.token_issue_en = 1;
    return this;
  }
};

/**
 * @class NoCDmaMM2SCtrl
 * @brief Class representing NoC-tile DMA Channel MM2S control.
 */
class NoCDmaMM2SCtrl : public DmaChannel<NoCDmaMM2SCtrl> {
public:
  /**
   * @brief Constructor.
   * @param id  ID/index of the channel control
   */
  NoCDmaMM2SCtrl(uint8_t id = 0)
      : DmaChannel(id), m_base(ShimChannelInfo::ShimChCtrlBase +
                               (ShimChannelInfo::ShimChIdxOffset *
                                ShimChannelInfo::ShimNumChannels)),
        m_offset(ShimChannelInfo::ShimChIdxOffset) {
    m_regval.reg = 0;
  }

public:
  const uint32_t m_base;     ///< Channel control base addr
  const uint32_t m_offset;   ///< Channel control offset
  NoCDmaChCtrlMM2S m_regval; ///< Channel control register value
public:
  /**
   * @brief   Creates a unique name for the element.
   * @returns returns string with a unique name
   */
  inline std::string name(void) {
    return std::string("NoCDmaMM2SCtrl_") + std::to_string(m_id);
  }

  /**
   * @brief   When set, pauses the issuing of new AXI-MM commands.
   * @returns pointer to itself
   */
  inline NoCDmaMM2SCtrl* setPauseMem(void) {
    m_regval.pause_mem = 1;
    return this;
  }

  /**
   * @brief   When set, pauses the stream traffic.
   * @returns pointer to itself
   */
  inline NoCDmaMM2SCtrl* setPauseStream(void) {
    m_regval.pause_stream = 1;
    return this;
  }

  /**
   * @brief   Sets controller ID.
   * @param   id  controller ID
   * @returns pointer to itself
   */
  inline NoCDmaMM2SCtrl* setControllerId(std::uint8_t id) {
    m_regval.controller_id = id;
    return this;
  }
};

/**
 * @class NoCDmaMM2SQueue
 * @brief Class representing NoC-tile DMA Channel MM2S start queue.
 */
class NoCDmaMM2SQueue : public DmaChannel<NoCDmaMM2SQueue> {
public:
  /**
   * @brief     Constructor.
   * @param id  ID/index of the channel start queue
   */
  NoCDmaMM2SQueue(uint8_t id = 0)
      : DmaChannel(id), m_base(ShimChannelInfo::ShimChQueueBase +
                               (ShimChannelInfo::ShimChIdxOffset *
                                ShimChannelInfo::ShimNumChannels)),
        m_offset(ShimChannelInfo::ShimChIdxOffset) {
    m_regval.reg = 0;
  }

public:
  const uint32_t m_base;      ///< Channel start queue base addr
  const uint32_t m_offset;    ///< Channel start queue offset
  NoCDmaChQueueMM2S m_regval; ///< Channel start queue register value
public:
  /**
   * @brief   Creates a unique name for the element.
   * @returns returns string with a unique name
   */
  inline std::string name(void) {
    return std::string("NoCDmaMM2SQueue_") + std::to_string(m_id);
  }

  /**
   * @brief   Sets start BD ID.
   * @param   bd  BD id to start the queue
   * @returns pointer to itself
   */
  inline NoCDmaMM2SQueue* setStartBd(std::uint8_t bd) {
    assert(Sanity::NoCTile::Queue::checkStartBd(bd));
    m_regval.start_bd_id = bd;
    return this;
  }

  /**
   * @brief   Sets BD repeat count.
   * @param   count repeat count
   * @returns pointer to itself
   */
  inline NoCDmaMM2SQueue* setRepeatCount(std::uint8_t count) {
    assert(Sanity::NoCTile::Queue::checkRepeatCount(count));
    m_regval.repeat = count - 1;
    return this;
  }

  /**
   * @brief   Enables token issue.
   * @returns pointer to itself
   */
  inline NoCDmaMM2SQueue* enableTokenIssue(void) {
    m_regval.token_issue_en = 1;
    return this;
  }
};

/**
 * @class Lock
 * @brief Base class representing locks.
 */
template <class Derived> class Lock : public DmaElement {
public:
  /**
   * @brief Constructor.
   * @param id  lock ID/index
   */
  Lock(int8_t id = -1) : DmaElement(id), m_addr(0) { m_value.reg = 0; }

  /**
   * @brief   Sets lock value.
   * @param   val lock value to be set
   * @returns returns pointer to itself
   */
  inline Derived* setValue(std::int8_t val) {
    assert(Sanity::Lock::checkValue(val));
    m_value.value = val;
    return static_cast<Derived*>(this);
  }

  /**
   * @brief Sets tile Location.
   * @param loc tile location
   */
  inline void write(Location& loc) { m_location = loc; }

  /**
   * @brief Method to accept a Visitor to generate instructions.
   * @param writer        pointer to the instruction writer implementation
   * @param instr_buffer  pointer to memory where the instruction should be
   * written
   * @return              number of instructions generated
   */
  int accept(InstructionWriterImpl* writer, uint32_t* instr_buffer) override {
    return writer->write(static_cast<Derived*>(this), instr_buffer);
  }

public:
  // int8_t    m_id;       ///< lock ID
  uint32_t m_addr;     ///< lock address
  LockValue m_value;   ///< lock value
  Location m_location; ///< tile-location
};

/**
 * @class ATLock
 * @brief Class representing AIE-tile locks.
 */
class ATLock : public Lock<ATLock> {
public:
  /**
   * @brief Constructor.
   * @param id  lock ID
   */
  ATLock(int8_t id = -1)
      : Lock(id), m_base(ATLockInfo::AT_LOCK_BASE_ADDR),
        m_offset(ATLockInfo::AT_LOCK_OFFSET) {}

  /**
   * @brief   Creates a unique name for the element.
   * @returns returns string with a unique name
   */
  inline std::string name(void) {
    return std::string("ATLock_") + std::to_string(m_id);
  }

public:
  uint32_t m_base;   ///< lock base address
  uint32_t m_offset; ///< lock offset
};

/**
 * @class MTLock
 * @brief Class representing Mem-tile locks.
 */
class MTLock : public Lock<MTLock> {
public:
  /**
   * @brief Constructor.
   * @param id  lock ID
   */
  MTLock(int8_t id = -1)
      : Lock(id), m_base(MTLockInfo::MT_LOCK_BASE_ADDR),
        m_offset(MTLockInfo::MT_LOCK_OFFSET) {}

  /**
   * @brief   Creates a unique name for the element.
   * @returns returns string with a unique name
   */
  inline std::string name(void) {
    return std::string("MTLock_") + std::to_string(m_id);
  }

public:
  uint32_t m_base;   ///< lock base address
  uint32_t m_offset; ///< lock offset
};

/**
 * @class NTLock
 * @brief Class representing NoC-tile locks.
 */
class NTLock : public Lock<NTLock> {
public:
  /**
   * @brief Constructor.
   * @param id  lock ID
   */
  NTLock(int8_t id = -1)
      : Lock(id), m_base(NTLockInfo::NT_LOCK_BASE_ADDR),
        m_offset(NTLockInfo::NT_LOCK_OFFSET) {}

  /**
   * @brief   Creates a unique name for the element.
   * @returns returns string with a unique name
   */
  inline std::string name(void) {
    return std::string("NTLock_") + std::to_string(m_id);
  }

public:
  uint32_t m_base;   ///< lock base address
  uint32_t m_offset; ///< lock offset
};

/**
 * @class Word
 * @brief Class representing a 32-bit word in AIE-time memory.
 */
class Word : public DmaElement {
public:
  /**
   * @brief Constructor.
   * @param addr address where the word to be written
   * @param data word to be written
   */
  explicit Word(std::uint32_t addr = 0, std::uint32_t data = 0)
      : DmaElement(0), m_addr(addr), m_data(data) {}

  /**
   * @brief Sets tile Location.
   * @param loc tile location
   */
  inline void write(Location& loc) { m_location = loc; }

  /**
   * @brief   Creates a unique name for the element.
   * @returns returns string with a unique name
   */
  inline std::string name(void) {
    return std::string("Word_") + std::to_string(m_addr);
  }

  /**
   * @brief Method to accept a Visitor to generate instructions.
   * @param writer        pointer to the instruction writer implementation
   * @param instr_buffer  pointer to memory where the instruction should be
   * written
   * @return              number of instructions generated
   */
  int accept(InstructionWriterImpl* writer, uint32_t* instr_buffer) override {
    return writer->write(this, instr_buffer);
  }

public:
  std::uint32_t m_addr; ///< address of the word
  std::uint32_t m_data; ///< data word
  Location m_location;  ///< tile-location
};

/**
 * @class IndexWord
 * @brief Class representing an index to SRAM buffer.
 */
class IndexWord : public DmaElement {
public:
  /**
   * @brief Constructor.
   * @param addr address where the word to be written taken from SRAM buffer
   * index
   * @param data index to be written
   */
  explicit IndexWord(std::uint32_t addr = 0, std::uint32_t data = 0)
      : DmaElement(0), m_addr(addr), m_data(data) {}

  /**
   * @brief Sets tile Location.
   * @param loc tile location
   */
  inline void write(Location& loc) { m_location = loc; }

  /**
   * @brief   Creates a unique name for the element.
   * @returns returns string with a unique name
   */
  inline std::string name(void) {
    return std::string("IndexWord_") + std::to_string(m_addr);
  }

  /**
   * @brief Method to accept a Visitor to generate instructions.
   * @param writer        pointer to the instruction writer implementation
   * @param instr_buffer  pointer to memory where the instruction should be
   * written
   * @return              number of instructions generated
   */
  int accept(InstructionWriterImpl* writer, uint32_t* instr_buffer) override {
    return writer->write(this, instr_buffer);
  }

public:
  std::uint32_t m_addr; ///< address of the word
  std::uint32_t m_data; ///< data word
  Location m_location;  ///< tile-location
};

/**
 * @class SyncWord
 * @brief Class representing Sync instruction word.
 */
class SyncWord : public DmaElement {
public:
  /**
   * @brief Default constructor.
   */
  SyncWord()
      : DmaElement(0), m_direction(DmaDirection::S2MM), m_channel(0),
        m_ncols(1), m_nrows(1) {}

  /**
   * @brief Constructor.
   * @param direction DMA direction
   * @param ch        channel ID
   * @param ncols     number of columns
   * @param nrows     number of rows
   */
  SyncWord(DmaDirection direction, std::uint32_t ch, std::uint32_t ncols = 1,
           std::uint32_t nrows = 1)
      : m_direction(direction), m_channel(ch), m_ncols(ncols), m_nrows(nrows) {}

  /**
   * @brief Sets tile Location.
   * @param loc tile location
   */
  inline void write(Location& loc) { m_location = loc; }

  /**
   * @brief Method to accept a Visitor to generate instructions.
   * @param writer        pointer to the instruction writer implementation
   * @param instr_buffer  pointer to memory where the instruction should be
   * written
   * @return              number of instructions generated
   */
  int accept(InstructionWriterImpl* writer, uint32_t* instr_buffer) override {
    return writer->write(this, instr_buffer);
  }

  /**
   * @brief   Creates a unique name for the element.
   * @returns returns string with a unique name
   */
  inline std::string name(void) {
    return std::string("SyncWord_") + std::to_string(m_channel) + "_" +
           std::to_string(static_cast<uint32_t>(m_direction));
  }

public:
  DmaDirection m_direction; ///< DMA direction (S2MM/MM2S)
  std::uint32_t m_channel;  ///< channel ID
  std::uint32_t m_ncols;    ///< number of columns
  std::uint32_t m_nrows;    ///< number of rows
  Location m_location;      ///< tile-location
};

// ------------------------------------------------------------------
// IMPORTANT:
// ------------------------------------------------------------------
// These don't represent actual number of elements available on Silicon.
// We are using them to define size of vector of elements. This makes
// us not to book-keep if DMA element is already in use or not
// Saves a ton on time, particularly important when we need to
// generate instructions per frame i.e. for every new input.
// ------------------------------------------------------------------
/**@brief Number of AIE BD elements being used, does not represent actual
 * available number of BDs */
constexpr int NUM_AIE_BDS_USED = 32;
/**@brief Number of AIE Lock elements being used, does not represent actual
 * available number of Locks */
constexpr int NUM_AIE_LOCKS_USED = 32;
/**@brief Number of AIE Channel Control elements being used, does not represent
 * actual available number of Channels */
constexpr int NUM_AIE_CHANNEL_CTRL_USED = 16;
/**@brief Number of AIE Channel Queue elements being used, does not represent
 * actual available number of Queues */
constexpr int NUM_AIE_CHANNEL_QUEUES_USED = 8;
/**@brief Number of AIE Word elements being used, does not represent actual
 * available number of Words */
constexpr int NUM_AIE_WORDS_USED = 256;
/**@brief Number of AIE IndexWord elements being used, does not represent actual
 * available number of Words */
constexpr int NUM_AIE_IDX_WORDS_USED = 128;

/**
 * @class AieTileDma
 * @brief Class encapsulates all the elements required for DMA transactions
 * to/from AIE-tile.
 */
class AieTileDma {
public:
  /**
   * @brief Default constructor.
   */
  AieTileDma()
      : m_s2mm_ctrl(NUM_AIE_CHANNEL_CTRL_USED),
        m_mm2s_ctrl(NUM_AIE_CHANNEL_CTRL_USED),
        m_s2mm_queue(NUM_AIE_CHANNEL_QUEUES_USED),
        m_mm2s_queue(NUM_AIE_CHANNEL_QUEUES_USED), m_locks(NUM_AIE_LOCKS_USED),
        m_bds(NUM_AIE_BDS_USED), m_words(NUM_AIE_WORDS_USED),
        m_idx_words(NUM_AIE_IDX_WORDS_USED) {}

  /**
   * @brief Default destructor.
   */
  ~AieTileDma() {
#ifdef DEBUG_FWR
    std::cout << "-- Aie BDs Used: " << bd_idx << std::endl;
    std::cout << "-- Aie Locks Used: " << lock_idx << std::endl;
    std::cout << "-- Aie S2MM Ctrl Used: " << s2mm_ctrl_idx << std::endl;
    std::cout << "-- Aie MM2S Ctrl Used: " << mm2s_ctrl_idx << std::endl;
    std::cout << "-- Aie S2MM Queue Used: " << s2mm_queue_idx << std::endl;
    std::cout << "-- Aie MM2S Queue Used: " << mm2s_queue_idx << std::endl;
    std::cout << "-- Aie Words Used: " << word_idx << std::endl;
    std::cout << "-- Aie Idx Word Used: " << idx_word_idx << std::endl;
#endif
  }

  /**
   * @brief   Get a BD element.
   * @param   id  ID of the BD element
   * @returns pointer to BD element
   */
  inline AieBD* getBD(std::uint8_t id) {
    if (bd_idx == m_bds.size())
      m_bds.resize(m_bds.size() + NUM_AIE_BDS_USED);
    m_bds[bd_idx].setId(id);
    m_bds[bd_idx].setUniqueId<TileType::AIE, ElementType::BD>(bd_idx);
    return m_bds[bd_idx++].resetBDs();
  }

  /**
   * @brief   Get a Lock element.
   * @param   id  ID of the Lock element
   * @returns pointer to lock element
   */
  inline ATLock* getLock(std::uint8_t id) {
    if (lock_idx == m_locks.size())
      m_locks.resize(m_locks.size() + NUM_AIE_LOCKS_USED);
    m_locks[lock_idx].setId(id);
    m_locks[lock_idx].setUniqueId<TileType::AIE, ElementType::LOCK>(lock_idx);
    return &(m_locks[lock_idx++]);
  }

  /**
   * @brief   Get a S2MM control element.
   * @param   ch  ID of the control element
   * @returns pointer to control element
   */
  inline AieDmaS2MMCtrl* getS2MMCtrl(std::uint8_t ch) {
    if (s2mm_ctrl_idx == m_s2mm_ctrl.size())
      m_s2mm_ctrl.resize(m_s2mm_ctrl.size() + NUM_AIE_CHANNEL_CTRL_USED);
    m_s2mm_ctrl[s2mm_ctrl_idx].setId(ch);
    m_s2mm_ctrl[s2mm_ctrl_idx]
        .setUniqueId<TileType::AIE, ElementType::S2MMCTRL>(s2mm_ctrl_idx);
    return &(m_s2mm_ctrl[s2mm_ctrl_idx++]);
  }

  /**
   * @brief   Get a MM2S control element.
   * @param   ch  ID of the control element
   * @returns pointer to control element
   */
  inline AieDmaMM2SCtrl* getMM2SCtrl(std::uint8_t ch) {
    if (mm2s_ctrl_idx == m_mm2s_ctrl.size())
      m_mm2s_ctrl.resize(m_mm2s_ctrl.size() + NUM_AIE_CHANNEL_CTRL_USED);
    m_mm2s_ctrl[mm2s_ctrl_idx].setId(ch);
    m_mm2s_ctrl[mm2s_ctrl_idx]
        .setUniqueId<TileType::AIE, ElementType::MM2SCTRL>(mm2s_ctrl_idx);
    return &(m_mm2s_ctrl[mm2s_ctrl_idx++]);
  }

  /**
   * @brief   Get a S2MM start queue element.
   * @param   ch  ID of the queue element
   * @returns pointer to queue element
   */
  inline AieDmaS2MMQueue* getS2MMQueue(std::uint8_t ch) {
    if (s2mm_queue_idx == m_s2mm_queue.size())
      m_s2mm_queue.resize(m_s2mm_queue.size() + NUM_AIE_CHANNEL_QUEUES_USED);
    m_s2mm_queue[s2mm_queue_idx].setId(ch);
    m_s2mm_queue[s2mm_queue_idx]
        .setUniqueId<TileType::AIE, ElementType::S2MMQUEUE>(s2mm_queue_idx);
    return &(m_s2mm_queue[s2mm_queue_idx++]);
  }

  /**
   * @brief   Get a MM2S start queue element.
   * @param   ch  ID of the queue element
   * @returns pointer to queue element
   */
  inline AieDmaMM2SQueue* getMM2SQueue(std::uint8_t ch) {
    if (mm2s_queue_idx == m_mm2s_queue.size())
      m_mm2s_queue.resize(m_mm2s_queue.size() + NUM_AIE_CHANNEL_QUEUES_USED);
    m_mm2s_queue[mm2s_queue_idx].setId(ch);
    m_mm2s_queue[mm2s_queue_idx]
        .setUniqueId<TileType::AIE, ElementType::MM2SQUEUE>(mm2s_queue_idx);
    return &(m_mm2s_queue[mm2s_queue_idx++]);
  }

  /**
   * @brief   Get a Word element.
   * @param   addr  word address
   * @param   data  word data
   * @returns pointer to word element
   */
  inline Word* getWord(std::uint32_t addr, std::uint32_t data) {
    if (word_idx == m_words.size())
      m_words.resize(m_words.size() + NUM_AIE_WORDS_USED);
    m_words[word_idx].m_addr = addr;
    m_words[word_idx].m_data = data;
    m_words[word_idx].setUniqueId<TileType::AIE, ElementType::WORD>(word_idx);
    return &(m_words[word_idx++]);
  }

  /**
   * @brief   Get an IndexWord element.
   * @param   addr  word address
   * @param   data  index word data
   * @returns pointer to index word element
   */
  inline IndexWord* getIndexWord(std::uint32_t addr, std::uint32_t data) {
    if (idx_word_idx == m_idx_words.size())
      m_idx_words.resize(m_idx_words.size() + NUM_AIE_IDX_WORDS_USED);
    m_idx_words[idx_word_idx].m_addr = addr;
    m_idx_words[idx_word_idx].m_data = data;
    m_idx_words[idx_word_idx]
        .setUniqueId<TileType::AIE, ElementType::INDEXWORD>(idx_word_idx);
    return &(m_idx_words[idx_word_idx++]);
  }

  /**
   * @brief Resets all DMA element indices.
   */
  inline void reset(void) {
    s2mm_ctrl_idx = 0;
    mm2s_ctrl_idx = 0;
    s2mm_queue_idx = 0;
    mm2s_queue_idx = 0;
    lock_idx = 0;
    bd_idx = 0;
    word_idx = 0;
    idx_word_idx = 0;
  }

  /**
   * @brief   Gets a DMA element based on unique ID.
   * @param   uid unique ID
   * @returns pointer to the DMA element as base class pointer
   */
  DmaElement* getElement(uint32_t uid) {
    uint32_t elemType = (uid & 0x0F000000) >> 24;
    uint32_t elemIndex = (uid & 0x0000FFFF);
    // std::cout << "Elem (Type/Index): " << elemType << " / " << elemIndex <<
    // "\n";
    switch (elemType) {
    case ElementType::BD:
      return &m_bds[elemIndex];
    case ElementType::LOCK:
      return &m_locks[elemIndex];
    case ElementType::S2MMCTRL:
      return &m_s2mm_ctrl[elemIndex];
    case ElementType::MM2SCTRL:
      return &m_mm2s_ctrl[elemIndex];
    case ElementType::S2MMQUEUE:
      return &m_s2mm_queue[elemIndex];
    case ElementType::MM2SQUEUE:
      return &m_mm2s_queue[elemIndex];
    case ElementType::WORD:
      return &m_words[elemIndex];
    case ElementType::INDEXWORD:
      return &m_idx_words[elemIndex];
    default:
      throw std::runtime_error("Invalid ElementType");
    }
  }

private:
  int s2mm_ctrl_idx;  ///< S2MM control element vector index
  int mm2s_ctrl_idx;  ///< MM2S control element vector index
  int s2mm_queue_idx; ///< S2MM control queue element vector index
  int mm2s_queue_idx; ///< MM2S control queue element vector index
  int lock_idx;       ///< lock element vector index
  int bd_idx;         ///< BD element vector index
  int word_idx;       ///< word element vector index
  int idx_word_idx;   ///< index word element vector index

  std::vector<AieDmaS2MMCtrl> m_s2mm_ctrl;   ///< S2MM control elements
  std::vector<AieDmaMM2SCtrl> m_mm2s_ctrl;   ///< MM2S control elements
  std::vector<AieDmaS2MMQueue> m_s2mm_queue; ///< S2MM queue elements
  std::vector<AieDmaMM2SQueue> m_mm2s_queue; ///< MM2S queue elements
  std::vector<ATLock> m_locks;               ///< lock elements
  std::vector<AieBD> m_bds;                  ///< BD elements
  std::vector<Word> m_words;                 ///< word elements
  std::vector<IndexWord> m_idx_words;        ///< index word elements
};

// ------------------------------------------------------------------
// IMPORTANT:
// ------------------------------------------------------------------
// These don't represent actual number of elements available on Silicon.
// We are using them to define size of vector of elements. This makes
// us not to book-keep if DMA element is already in use or not
// Saves a ton on time, particularly important when we need to
// generate instructions per frame i.e. for every new input.
// ------------------------------------------------------------------
/**@brief Number of Mem BD elements being used, does not represent actual
 * available number of BDs */
constexpr int NUM_MEM_BDS_USED = 64;
/**@brief Number of Mem Lock elements being used, does not represent actual
 * available number of Locks */
constexpr int NUM_MEM_LOCKS_USED = 16;
/**@brief Number of Mem Channel Control elements being used, does not represent
 * actual available number of Channels */
constexpr int NUM_MEM_CHANNEL_CTRL_USED = 16;
/**@brief Number of Mem Channel Queue elements being used, does not represent
 * actual available number of Queues */
constexpr int NUM_MEM_CHANNEL_QUEUES_USED = 8;

/**
 * @class MemTileDma
 * @brief Class encapsulates all the elements required for DMA transactions
 * to/from Mem-tile.
 */
class MemTileDma {
public:
  /**
   * @brief Default constructor.
   */
  MemTileDma()
      : m_s2mm_ctrl(NUM_MEM_CHANNEL_CTRL_USED),
        m_mm2s_ctrl(NUM_MEM_CHANNEL_CTRL_USED),
        m_s2mm_queue(NUM_MEM_CHANNEL_QUEUES_USED),
        m_mm2s_queue(NUM_MEM_CHANNEL_QUEUES_USED), m_locks(NUM_MEM_LOCKS_USED),
        m_bds(NUM_MEM_BDS_USED) {}

  /**
   * @brief Default destructor.
   */
  ~MemTileDma() {
#ifdef DEBUG_FWR
    std::cout << "-- Mem BDs Used: " << bd_idx << std::endl;
    std::cout << "-- Mem Locks Used: " << lock_idx << std::endl;
    std::cout << "-- Mem S2MM Ctrl Used: " << s2mm_ctrl_idx << std::endl;
    std::cout << "-- Mem MM2S Ctrl Used: " << mm2s_ctrl_idx << std::endl;
    std::cout << "-- Mem S2MM Queue Used: " << s2mm_queue_idx << std::endl;
    std::cout << "-- Mem MM2S Queue Used: " << mm2s_queue_idx << std::endl;
#endif
  }

  /**
   * @brief   Get a BD element.
   * @param   id  ID of the BD element
   * @returns pointer to BD element
   */
  inline MemBD* getBD(std::uint8_t id) {
    if (bd_idx == m_bds.size())
      m_bds.resize(m_bds.size() + NUM_MEM_BDS_USED);
    m_bds[bd_idx].setId(id);
    m_bds[bd_idx].setUniqueId<TileType::MEM, ElementType::BD>(bd_idx);
    return m_bds[bd_idx++].resetBDs();
  }

  /**
   * @brief   Get a Lock element.
   * @param   id  ID of the Lock element
   * @returns pointer to lock element
   */
  inline MTLock* getLock(std::uint8_t id) {
    if (lock_idx == m_locks.size())
      m_locks.resize(m_locks.size() + NUM_MEM_LOCKS_USED);
    m_locks[lock_idx].setId(id);
    m_locks[lock_idx].setUniqueId<TileType::MEM, ElementType::LOCK>(lock_idx);
    return &(m_locks[lock_idx++]);
  }

  /**
   * @brief   Get a S2MM control element.
   * @param   ch  ID of the control element
   * @returns pointer to control element
   */
  inline MemDmaS2MMCtrl* getS2MMCtrl(std::uint8_t ch) {
    if (s2mm_ctrl_idx == m_s2mm_ctrl.size())
      m_s2mm_ctrl.resize(m_s2mm_ctrl.size() + NUM_MEM_CHANNEL_CTRL_USED);
    m_s2mm_ctrl[s2mm_ctrl_idx].setId(ch);
    m_s2mm_ctrl[s2mm_ctrl_idx]
        .setUniqueId<TileType::MEM, ElementType::S2MMCTRL>(s2mm_ctrl_idx);
    return &(m_s2mm_ctrl[s2mm_ctrl_idx++]);
  }

  /**
   * @brief   Get a MM2S control element.
   * @param   ch  ID of the control element
   * @returns pointer to control element
   */
  inline MemDmaMM2SCtrl* getMM2SCtrl(std::uint8_t ch) {
    if (mm2s_ctrl_idx == m_mm2s_ctrl.size())
      m_mm2s_ctrl.resize(m_mm2s_ctrl.size() + NUM_MEM_CHANNEL_CTRL_USED);
    m_mm2s_ctrl[mm2s_ctrl_idx].setId(ch);
    m_mm2s_ctrl[mm2s_ctrl_idx]
        .setUniqueId<TileType::MEM, ElementType::MM2SCTRL>(mm2s_ctrl_idx);
    return &(m_mm2s_ctrl[mm2s_ctrl_idx++]);
  }

  /**
   * @brief   Get a S2MM start queue element.
   * @param   ch  ID of the queue element
   * @returns pointer to queue element
   */
  inline MemDmaS2MMQueue* getS2MMQueue(std::uint8_t ch) {
    if (s2mm_queue_idx == m_s2mm_queue.size())
      m_s2mm_queue.resize(m_s2mm_queue.size() + NUM_MEM_CHANNEL_QUEUES_USED);
    m_s2mm_queue[s2mm_queue_idx].setId(ch);
    m_s2mm_queue[s2mm_queue_idx]
        .setUniqueId<TileType::MEM, ElementType::S2MMQUEUE>(s2mm_queue_idx);
    return &(m_s2mm_queue[s2mm_queue_idx++]);
  }

  /**
   * @brief   Get a MM2S start queue element.
   * @param   ch  ID of the queue element
   * @returns pointer to queue element
   */
  inline MemDmaMM2SQueue* getMM2SQueue(std::uint8_t ch) {
    if (mm2s_queue_idx == m_mm2s_queue.size())
      m_mm2s_queue.resize(m_mm2s_queue.size() + NUM_MEM_CHANNEL_QUEUES_USED);
    m_mm2s_queue[mm2s_queue_idx].setId(ch);
    m_mm2s_queue[mm2s_queue_idx]
        .setUniqueId<TileType::MEM, ElementType::MM2SQUEUE>(mm2s_queue_idx);
    return &(m_mm2s_queue[mm2s_queue_idx++]);
  }

  /**
   * @brief Resets all DMA element indices.
   */
  inline void reset(void) {
    s2mm_ctrl_idx = 0;
    mm2s_ctrl_idx = 0;
    s2mm_queue_idx = 0;
    mm2s_queue_idx = 0;
    lock_idx = 0;
    bd_idx = 0;
  }

  /**
   * @brief   Gets a DMA element based on unique ID.
   * @param   uid unique ID
   * @returns pointer to the DMA element as base class pointer
   */
  DmaElement* getElement(uint32_t uid) {
    auto elemType = (uid & 0x0F000000) >> 24;
    auto elemIndex = (uid & 0x0000FFFF);
    // std::cout << "Elem (Type/Index): " << elemType << " / " << elemIndex <<
    // "\n";
    switch (elemType) {
    case ElementType::BD:
      return &m_bds[elemIndex];
    case ElementType::LOCK:
      return &m_locks[elemIndex];
    case ElementType::S2MMCTRL:
      return &m_s2mm_ctrl[elemIndex];
    case ElementType::MM2SCTRL:
      return &m_mm2s_ctrl[elemIndex];
    case ElementType::S2MMQUEUE:
      return &m_s2mm_queue[elemIndex];
    case ElementType::MM2SQUEUE:
      return &m_mm2s_queue[elemIndex];
    default:
      throw std::runtime_error("Invalid ElementType");
    }
  }

private:
  int s2mm_ctrl_idx;  ///< S2MM control element vector index
  int mm2s_ctrl_idx;  ///< MM2S control element vector index
  int s2mm_queue_idx; ///< S2MM control queue element vector index
  int mm2s_queue_idx; ///< MM2S control queue element vector index
  int lock_idx;       ///< lock element vector index
  int bd_idx;         ///< BD element vector index

  std::vector<MemDmaS2MMCtrl> m_s2mm_ctrl;   ///< S2MM control elements
  std::vector<MemDmaMM2SCtrl> m_mm2s_ctrl;   ///< MM2S control elements
  std::vector<MemDmaS2MMQueue> m_s2mm_queue; ///< S2MM queue elements
  std::vector<MemDmaMM2SQueue> m_mm2s_queue; ///< MM2S queue elements
  std::vector<MTLock> m_locks;               ///< lock elements
  std::vector<MemBD> m_bds;                  ///< BD elements
};

// ------------------------------------------------------------------
// IMPORTANT:
// ------------------------------------------------------------------
// These don't represent actual number of elements available on Silicon.
// We are using them to define size of vector of elements. This makes
// us not to book-keep if DMA element is already in use or not
// Saves a ton on time, particularly important when we need to
// generate instructions per frame i.e. for every new input.
// ------------------------------------------------------------------
/**@brief Number of NoC BD elements being used, does not represent actual
 * available number of BDs */
constexpr int NUM_NOC_BDS_USED = 256;
/**@brief Number of NoC Lock elements being used, does not represent actual
 * available number of Locks */
constexpr int NUM_NOC_LOCKS_USED = 4;
/**@brief Number of NoC Channel Control elements being used, does not represent
 * actual available number of Channels */
constexpr int NUM_NOC_CHANNEL_CTRL_USED = 8;
/**@brief Number of NoC Channel Queue elements being used, does not represent
 * actual available number of Queues */
constexpr int NUM_NOC_CHANNEL_QUEUES_USED = 128;
/**@brief Number of NoC SyncWord elements being used, does not represent actual
 * available number of Queues */
constexpr int NUM_NOC_SYNC_WORDS_USED = 128;

/**
 * @class NoCTileDma
 * @brief Class encapsulates all the elements required for DMA transactions
 * to/from NoC-tile.
 */
class NoCTileDma {
public:
  /**
   * @brief Default constructor.
   */
  NoCTileDma()
      : m_s2mm_ctrl(NUM_NOC_CHANNEL_CTRL_USED),
        m_mm2s_ctrl(NUM_NOC_CHANNEL_CTRL_USED),
        m_s2mm_queue(NUM_NOC_CHANNEL_QUEUES_USED),
        m_mm2s_queue(NUM_NOC_CHANNEL_QUEUES_USED), m_locks(NUM_NOC_LOCKS_USED),
        m_bds(NUM_NOC_BDS_USED), m_sync_words(NUM_NOC_SYNC_WORDS_USED) {}

  /**
   * @brief Default destructor.
   */
  ~NoCTileDma() {
#ifdef DEBUG_FWR
    std::cout << "-- NoC BDs Used: " << bd_idx << std::endl;
    std::cout << "-- NoC Locks Used: " << lock_idx << std::endl;
    std::cout << "-- NoC S2MM Ctrl Used: " << s2mm_ctrl_idx << std::endl;
    std::cout << "-- NoC MM2S Ctrl Used: " << mm2s_ctrl_idx << std::endl;
    std::cout << "-- NoC S2MM Queue Used: " << s2mm_queue_idx << std::endl;
    std::cout << "-- NoC MM2S Queue Used: " << mm2s_queue_idx << std::endl;
    std::cout << "-- NoC Sync Words Used: " << sync_word_idx << std::endl;
#endif
  }

  /**
   * @brief   Get a BD element.
   * @param   id  ID of the BD element
   * @returns pointer to BD element
   */
  inline NoCBD* getBD(std::uint8_t id) {
    if (bd_idx == m_bds.size())
      m_bds.resize(m_bds.size() + NUM_NOC_BDS_USED);
    m_bds[bd_idx].setId(id);
    m_bds[bd_idx].setUniqueId<TileType::NoC, ElementType::BD>(bd_idx);
    return m_bds[bd_idx++].resetBDs();
  }

  /**
   * @brief   Get a Lock element.
   * @param   id  ID of the Lock element
   * @returns pointer to lock element
   */
  inline NTLock* getLock(std::uint8_t id) {
    if (lock_idx == m_locks.size())
      m_locks.resize(m_locks.size() + NUM_NOC_LOCKS_USED);
    m_locks[lock_idx].setId(id);
    m_locks[lock_idx].setUniqueId<TileType::NoC, ElementType::LOCK>(lock_idx);
    return &(m_locks[lock_idx++]);
  }

  /**
   * @brief   Get a S2MM control element.
   * @param   ch  ID of the control element
   * @returns pointer to control element
   */
  inline NoCDmaS2MMCtrl* getS2MMCtrl(std::uint8_t ch) {
    if (s2mm_ctrl_idx == m_s2mm_ctrl.size())
      m_s2mm_ctrl.resize(m_s2mm_ctrl.size() + NUM_NOC_CHANNEL_CTRL_USED);
    m_s2mm_ctrl[s2mm_ctrl_idx].setId(ch);
    m_s2mm_ctrl[s2mm_ctrl_idx]
        .setUniqueId<TileType::NoC, ElementType::S2MMCTRL>(s2mm_ctrl_idx);
    return &(m_s2mm_ctrl[s2mm_ctrl_idx++]);
  }

  /**
   * @brief   Get a MM2S control element.
   * @param   ch  ID of the control element
   * @returns pointer to control element
   */
  inline NoCDmaMM2SCtrl* getMM2SCtrl(std::uint8_t ch) {
    if (mm2s_ctrl_idx == m_mm2s_ctrl.size())
      m_mm2s_ctrl.resize(m_mm2s_ctrl.size() + NUM_NOC_CHANNEL_CTRL_USED);
    m_mm2s_ctrl[mm2s_ctrl_idx].setId(ch);
    m_mm2s_ctrl[mm2s_ctrl_idx]
        .setUniqueId<TileType::NoC, ElementType::MM2SCTRL>(mm2s_ctrl_idx);
    return &(m_mm2s_ctrl[mm2s_ctrl_idx++]);
  }

  /**
   * @brief   Get a S2MM start queue element.
   * @param   ch  ID of the queue element
   * @returns pointer to queue element
   */
  inline NoCDmaS2MMQueue* getS2MMQueue(std::uint8_t ch) {
    if (s2mm_queue_idx == m_s2mm_queue.size())
      m_s2mm_queue.resize(m_s2mm_queue.size() + NUM_NOC_CHANNEL_QUEUES_USED);
    m_s2mm_queue[s2mm_queue_idx].setId(ch);
    m_s2mm_queue[s2mm_queue_idx]
        .setUniqueId<TileType::NoC, ElementType::S2MMQUEUE>(s2mm_queue_idx);
    return &(m_s2mm_queue[s2mm_queue_idx++]);
  }

  /**
   * @brief   Get a MM2S start queue element.
   * @param   ch  ID of the queue element
   * @returns pointer to queue element
   */
  inline NoCDmaMM2SQueue* getMM2SQueue(std::uint8_t ch) {
    if (mm2s_queue_idx == m_mm2s_queue.size())
      m_mm2s_queue.resize(m_mm2s_queue.size() + NUM_NOC_CHANNEL_QUEUES_USED);
    m_mm2s_queue[mm2s_queue_idx].setId(ch);
    m_mm2s_queue[mm2s_queue_idx]
        .setUniqueId<TileType::NoC, ElementType::MM2SQUEUE>(mm2s_queue_idx);
    return &(m_mm2s_queue[mm2s_queue_idx++]);
  }

  /**
   * @brief   Get a SyncWord element.
   * @param   dir   DMA direction
   * @param   ch    channel ID
   * @param   ncol  number of columns
   * @param   nrow  number of rows
   * @returns pointer to sync word element
   */
  inline SyncWord* getSyncWord(DmaDirection dir, std::uint8_t ch,
                               std::uint8_t ncol, std::uint8_t nrow) {
    if (sync_word_idx == m_sync_words.size())
      m_sync_words.resize(m_sync_words.size() + NUM_NOC_SYNC_WORDS_USED);
    m_sync_words[sync_word_idx].m_direction = dir;
    m_sync_words[sync_word_idx].m_channel = ch;
    m_sync_words[sync_word_idx].m_ncols = ncol;
    m_sync_words[sync_word_idx].m_nrows = nrow;
    m_sync_words[sync_word_idx]
        .setUniqueId<TileType::NoC, ElementType::SYNCWORD>(sync_word_idx);
    return &(m_sync_words[sync_word_idx++]);
  }

  /**
   * @brief Resets all DMA element indices.
   */
  inline void reset(void) {
    s2mm_ctrl_idx = 0;
    mm2s_ctrl_idx = 0;
    s2mm_queue_idx = 0;
    mm2s_queue_idx = 0;
    lock_idx = 0;
    bd_idx = 0;
    sync_word_idx = 0;
  }

  /**
   * @brief   Gets a DMA element based on unique ID.
   * @param   uid unique ID
   * @returns pointer to the DMA element as base class pointer
   */
  DmaElement* getElement(uint32_t uid) {
    auto elemType = (uid & 0x0F000000) >> 24;
    auto elemIndex = (uid & 0x0000FFFF);
    // std::cout << "Elem (Type/Index): " << elemType << " / " << elemIndex <<
    // "\n";
    switch (elemType) {
    case ElementType::BD:
      return &m_bds[elemIndex];
    case ElementType::LOCK:
      return &m_locks[elemIndex];
    case ElementType::S2MMCTRL:
      return &m_s2mm_ctrl[elemIndex];
    case ElementType::MM2SCTRL:
      return &m_mm2s_ctrl[elemIndex];
    case ElementType::S2MMQUEUE:
      return &m_s2mm_queue[elemIndex];
    case ElementType::MM2SQUEUE:
      return &m_mm2s_queue[elemIndex];
    case ElementType::SYNCWORD:
      return &m_sync_words[elemIndex];
    default:
      throw std::runtime_error("Invalid ElementType");
    }
  }

private:
  int s2mm_ctrl_idx;  ///< S2MM control element vector index
  int mm2s_ctrl_idx;  ///< MM2S control element vector index
  int s2mm_queue_idx; ///< S2MM control queue element vector index
  int mm2s_queue_idx; ///< MM2S control queue element vector index
  int lock_idx;       ///< lock element vector index
  int bd_idx;         ///< BD element vector index
  int sync_word_idx;  ///< sync word element vector index

  std::vector<NoCDmaS2MMCtrl> m_s2mm_ctrl;   ///< S2MM control elements
  std::vector<NoCDmaMM2SCtrl> m_mm2s_ctrl;   ///< MM2S control elements
  std::vector<NoCDmaS2MMQueue> m_s2mm_queue; ///< S2MM queue elements
  std::vector<NoCDmaMM2SQueue> m_mm2s_queue; ///< MM2S queue elements
  std::vector<NTLock> m_locks;               ///< lock elements
  std::vector<NoCBD> m_bds;                  ///< BD elements
  std::vector<SyncWord> m_sync_words;        ///< sync word elements
};

} // namespace AIE2
