/*
 *     The Xilinx Vitis AI Vaip in this distribution are provided under the
 * following free and permissive binary-only license, but are not provided in
 * source code form.  While the following free and permissive license is similar
 * to the BSD open source license, it is NOT the BSD open source license nor
 * other OSI-approved open source license.
 *
 *      Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights
 * reserved.
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
#pragma once

#include <fstream>
#include <iomanip>
#include <string>
#include <utility>
#include <vector>

#include "aie2_dma_types.hpp"
#include "xf_aie_const.hpp"

static int byte_align(int n) { return 32 * ((n + 31) / 32); }

// Number of locks used for Mem and AIE tile
constexpr int NUM_AT_LOCKS = 4;
constexpr int NUM_MT_LOCKS = 8;
// Number of channels used for Mem-tile DMA
constexpr int NUM_MT_DMA_S2MM_CHS = 6;
constexpr int NUM_MT_DMA_MM2S_CHS = 6;
// Number of channels used AIE-tile DMA
constexpr int NUM_AT_DMA_S2MM_CHS = 1;
constexpr int NUM_AT_DMA_MM2S_CHS = 1;

template <class Derived> class InstructionCompiler {
public:
  // Ctor
  InstructionCompiler(std::uint32_t* instr_buffer = nullptr)
      : m_aie_dma(std::make_unique<AIE2::AieTileDma>()),
        m_mem_dma(std::make_unique<AIE2::MemTileDma>()),
        m_noc_dma(std::make_unique<AIE2::NoCTileDma>()),
        m_instr_buffer(instr_buffer), m_instr_counter(0)
#ifndef USE_VISITOR
        ,
        m_instr_writer(std::make_unique<IPUInstructionWriter>())
#endif
  {
    assert(DebugModeWarning());
  }

  // Dtor
  virtual ~InstructionCompiler() {}

  // Dump Text file
  int dumpTxt(const std::string& fname);

  // Dump Hpp file
  int dumpHpp(const std::string& fname);

  // Dump binary file
  int dumpBin(const std::string& fname);

  int generate(const uint16_t* rtps) {
    return static_cast<Derived*>(this)->generate(rtps);
  }

private:
  bool DebugModeWarning(void) {
    std::cout << "\n\033[33m" // Yellow
              << "[WARNING] Running in Debug Mode. Performance will be lower "
                 "due to additional checks."
              << "\033[0m\n\n";
    return true;
  }

protected:
  void init(std::vector<AIE2::TileMetaData>& tiles_in,
            std::vector<AIE2::TileMetaData>& tiles_out,
            int out_ptr2_offset = 0);
  void init1(std::vector<AIE2::TileMetaData>& tiles_in,
             std::vector<AIE2::TileMetaData>& tiles_out);

  AIE2::DmaElement* getDmaElement(uint32_t uid);

  void generateMTPingPongConfigs(
      std::vector<uint32_t>& components, const uint32_t ping_addr,
      const uint32_t pong_addr, const uint32_t len, AIE2::DmaTensor4D& tensor,
      const uint8_t start_bd, const uint8_t start_lock, const uint8_t lock_val,
      const uint32_t channel, const uint32_t repeat, AIE2::Location& loc,
      bool is_input);

  void generateInitAieDma(std::vector<uint32_t>& components);
  void generateAieDma(std::vector<uint32_t>& components);
  void generateAieDma1(std::vector<uint32_t>& components);
  void generateInitMemDma(std::vector<uint32_t>& components);
  void generateMemDma(std::vector<uint32_t>& components);
  void generateMemDma1(std::vector<uint32_t>& components);
  void generateMemDma2(std::vector<uint32_t>& components);
  void generateMemDma_singlecore(std::vector<uint32_t>& components);
  void
  generateMemDma_singlecore_singlechannel(std::vector<uint32_t>& components);

protected:
  // Tile DMAs
  std::unique_ptr<AIE2::AieTileDma> m_aie_dma;
  std::unique_ptr<AIE2::MemTileDma> m_mem_dma;
  std::unique_ptr<AIE2::NoCTileDma> m_noc_dma;

  // Instruction buffer and counter
  uint32_t* m_instr_buffer;
  uint32_t m_instr_counter;

#ifndef USE_VISITOR
  std::unique_ptr<InstructionWriterImpl> m_instr_writer;
#endif

  // RTPs
  const uint32_t m_rtp_size = RTP_SIZE;
  const uint32_t m_rtp_addr = 0x2000;
  const uint32_t m_rtp_offset_size = RTP_OFFSETS_SIZE;
  // Indices
  const uint32_t m_in_ind = RTP_SIZE;
  const uint32_t m_out_ind = m_in_ind + 32;
  const uint32_t m_buf_ind = m_out_ind + 32;
  // Num of AIE Cores
  const uint32_t m_num_at = 4;
  // Channels
  const int8_t m_mt_s2mm_chs[NUM_MT_DMA_S2MM_CHS] = {0, 1, 2, 3, 4, 5};
  const int8_t m_mt_mm2s_chs[NUM_MT_DMA_MM2S_CHS] = {0, 1, 2, 3, 4, 5};
  const int8_t m_at_s2mm_chs[NUM_AT_DMA_S2MM_CHS] = {0};
  const int8_t m_at_mm2s_chs[NUM_AT_DMA_MM2S_CHS] = {0};
  // Locks
  const int8_t m_at_locks[NUM_AT_LOCKS] = {0, 1, 2, 3};
  const int8_t m_mt_locks[NUM_MT_LOCKS] = {0, 1, 2, 3, 4, 5, 6, 7};
  // Offsets
  uint16_t m_in_offsets[16] = {};
  uint16_t m_out_offsets[16] = {};
  uint16_t m_buf_offsets[16] = {};
  // Window Sizes
  std::vector<uint32_t> m_tile_window_size_in;
  std::vector<uint32_t> m_tile_window_size_out;
  // Total input/output windows size
  uint32_t m_tile_window_size_in_total{0};
  uint32_t m_tile_window_size_out_total{0};
  // Ping/pong offsets
  uint32_t m_ping_offset_in{0};
  uint32_t m_pong_offset_in{0};
  uint32_t m_ping_offset_out{0};
  uint32_t m_pong_offset_out{0};
  // Buffer offset
  uint32_t m_buffer_offset{0};
  // Ping/pong address
  uint32_t m_ping_addr_in{0};
  uint32_t m_pong_addr_in{0};
  uint32_t m_ping_addr_out{0};
  uint32_t m_pong_addr_out{0};
  // Mem-tile ping/pong
  uint32_t m_ping_mem_in{0};
  uint32_t m_pong_mem_in{0};
  uint32_t m_ping_mem_out{0};
  uint32_t m_pong_mem_out{0};
  // Input/Output shim ports
  uint32_t m_in_ports = 2;
  uint32_t m_out_ports = 1;
};

template <class Derived>
AIE2::DmaElement* InstructionCompiler<Derived>::getDmaElement(uint32_t uid) {
  auto tileType = uid >> 28;
  switch (tileType) {
  case AIE2::TileType::AIE:
    return m_aie_dma->getElement(uid);
  case AIE2::TileType::MEM:
    return m_mem_dma->getElement(uid);
  case AIE2::TileType::NoC:
    return m_noc_dma->getElement(uid);
  default:
    throw std::runtime_error("Invalid TileType!");
  }
}

template <class Derived>
int InstructionCompiler<Derived>::dumpTxt(const std::string& fname) {
  std::ofstream instr_fp(fname);
  if (!instr_fp.is_open()) {
    std::cerr << "Error: Unable to open file " << fname << " for writing!\n";
    return 1;
  }
  for (uint32_t i = 0; i < m_instr_counter; ++i) {
    instr_fp << std::setfill('0') << std::setw(sizeof(uint32_t) * 2) << std::hex
             << m_instr_buffer[i] << "\n";
  }
  instr_fp.close();
  return 0;
}

template <class Derived>
int InstructionCompiler<Derived>::dumpHpp(const std::string& fname) {
  std::ofstream instr_fp(fname);
  if (!instr_fp.is_open()) {
    std::cerr << "Error: Unable to open file " << fname << " for writing!\n";
    return 1;
  }
  instr_fp << "#pragma once\n#include <vector>\n\n";
  instr_fp << "static std::vector<uint32_t> instr_buffer = {\n";
  for (uint32_t i = 0; i < m_instr_counter; ++i) {
    instr_fp << "    0x";
    instr_fp << std::setfill('0') << std::setw(sizeof(uint32_t) * 2) << std::hex
             << m_instr_buffer[i];
    instr_fp << ",\n";
  }
  instr_fp << "};\n";
  instr_fp.close();
  return 0;
}

template <class Derived>
int InstructionCompiler<Derived>::dumpBin(const std::string& fname) {
  std::ofstream instr_fp(fname, std::ios::binary | std::ios::out);
  if (!instr_fp.is_open()) {
    std::cerr << "Error: Unable to open file " << fname << " for writing!\n";
    return 1;
  }
  instr_fp.write((const char*)(m_instr_buffer),
                 m_instr_counter * sizeof(uint32_t));
  instr_fp.close();
  return 0;
}

template <class Derived>
void InstructionCompiler<Derived>::init(
    std::vector<AIE2::TileMetaData>& tiles_in,
    std::vector<AIE2::TileMetaData>& tiles_out, int out_ptr2_offset) {

  // Reset all DMAs
  m_aie_dma->reset();
  m_mem_dma->reset();
  m_noc_dma->reset();
  // Reset instruction counter
  m_instr_counter = 0;

  m_tile_window_size_in_total = 0;
  for (uint32_t i = 0; i < tiles_in.size(); ++i) {
    auto size = tiles_in[i].height * tiles_in[i].width * tiles_in[i].channel;
    // std::cout << tiles_in[i].height << " " << tiles_in[i].width << " " <<
    // tiles_in[i].channel << std::endl;
    m_tile_window_size_in.push_back(size);
    m_tile_window_size_in_total += size;
  }

  m_tile_window_size_out_total = 0;
  for (uint32_t i = 0; i < tiles_out.size(); ++i) {
    auto size = tiles_out[i].height * tiles_out[i].width * tiles_out[i].channel;
    // std::cout << tiles_out[i].height << " " << tiles_out[i].width << " " <<
    // tiles_out[i].channel << std::endl;
    m_tile_window_size_out.push_back(size);
    m_tile_window_size_out_total += size;
  }

  // Compute offsets/addresses for AT/MT Dma
  // TODO :: make it generic or all nodes, this wont run for resize .
  m_ping_offset_in = byte_align(0);
  m_pong_offset_in = byte_align(m_ping_offset_in + m_tile_window_size_in_total);
  m_ping_offset_out =
      byte_align(m_pong_offset_in + m_tile_window_size_in_total);
  m_pong_offset_out =
      byte_align(m_ping_offset_out + m_tile_window_size_out_total);
  m_buffer_offset =
      byte_align(m_pong_offset_out + m_tile_window_size_out_total);

  m_ping_addr_in = m_rtp_addr + m_rtp_offset_size + m_ping_offset_in;
  m_pong_addr_in = m_ping_addr_in + m_pong_offset_in;
  m_ping_addr_out = m_ping_addr_in + m_ping_offset_out;
  m_pong_addr_out = m_ping_addr_in + m_pong_offset_out;

  m_in_offsets[0] = m_ping_offset_in;
  m_in_offsets[1] = m_pong_offset_in;
  m_out_offsets[0] = m_ping_offset_out;
  m_out_offsets[1] = m_pong_offset_out;
  m_out_offsets[2] = 0;
  // # set offset for second output port
  m_out_offsets[3] = out_ptr2_offset;
  m_buf_offsets[0] = m_buffer_offset;
  // TODO:: make generic
  m_in_offsets[2] = 0;
  for (int i = 1; i < tiles_in.size(); ++i) {
    m_in_offsets[i + 2] = m_in_offsets[i + 1] + m_tile_window_size_in[i - 1];
  }

  // std::cout << "-- Init Done ...\n";
}
template <class Derived>
void InstructionCompiler<Derived>::init1(
    std::vector<AIE2::TileMetaData>& tiles_in,
    std::vector<AIE2::TileMetaData>& tiles_out) {
  // Reset all DMAs
  m_aie_dma->reset();
  m_mem_dma->reset();
  m_noc_dma->reset();
  // Reset instruction counter
  m_instr_counter = 0;

  m_tile_window_size_in_total = 0;
  for (uint32_t i = 0; i < tiles_in.size(); ++i) {
    auto size = tiles_in[i].height * tiles_in[i].width * tiles_in[i].channel;
    m_tile_window_size_in.push_back(size);
    m_tile_window_size_in_total += size;
  }

  m_tile_window_size_out_total = 0;
  for (uint32_t i = 0; i < tiles_out.size(); ++i) {
    auto size = tiles_out[i].height * tiles_out[i].width * tiles_out[i].channel;
    m_tile_window_size_out.push_back(size);
    m_tile_window_size_out_total += size;
  }

  // Compute offsets/addresses for AT/MT Dma
  // TODO :: make it generic or all nodes, this wont run for resize .
  m_ping_offset_in = byte_align(0);
  m_pong_offset_in =
      byte_align(m_ping_offset_in + m_tile_window_size_in_total / 2);
  m_ping_offset_out =
      byte_align(m_pong_offset_in + m_tile_window_size_in_total / 2);
  m_pong_offset_out =
      byte_align(m_ping_offset_out + m_tile_window_size_out_total / 2);
  m_buffer_offset =
      byte_align(m_pong_offset_out + m_tile_window_size_out_total / 2);

  m_ping_addr_in = m_rtp_addr + m_rtp_offset_size + m_ping_offset_in;
  m_pong_addr_in = m_ping_addr_in + m_pong_offset_in;
  m_ping_addr_out = m_ping_addr_in + m_ping_offset_out;
  m_pong_addr_out = m_ping_addr_in + m_pong_offset_out;

  m_in_offsets[0] = m_ping_offset_in;
  m_in_offsets[1] = m_pong_offset_in;
  m_out_offsets[0] = m_ping_offset_out;
  m_out_offsets[1] = m_pong_offset_out;
  m_buf_offsets[0] = m_buffer_offset;

  // std::cout << "-- Init Done ...\n";
}

template <class Derived>
void InstructionCompiler<Derived>::generateMTPingPongConfigs(
    std::vector<uint32_t>& components, const uint32_t ping_addr,
    const uint32_t pong_addr, const uint32_t len, AIE2::DmaTensor4D& tensor,
    const uint8_t start_bd, const uint8_t start_lock, const uint8_t lock_val,
    const uint32_t channel, const uint32_t repeat, AIE2::Location& loc,
    bool is_input) {
  // std::cout << "-- Mem DMA Channel: " << channel << " for " << (is_input ?
  // "Input" : "Output")
  //           << " Start Lock: " << uint32_t(start_lock)
  //           << " Start BD: " << uint32_t(start_bd) << "\n";

  AIE2::DmaIteration iter;
  AIE2::LockConfig lock_conf;
  std::uint8_t ncols = 1;

  uint8_t valid_bd = 1;
  uint8_t use_next_bd = 1;

  uint8_t startBD = start_bd;
  uint8_t nextBD = start_bd + 1;
  // 1. Ping
  if (is_input) {
    lock_conf.m_acq_id = start_lock + 1;
    lock_conf.m_acq_val = 0;
    lock_conf.m_acq_en = 1;
    lock_conf.m_rel_id = start_lock;
    lock_conf.m_rel_val = 1;
  } else {
    lock_conf.m_acq_id = start_lock;
    lock_conf.m_acq_val = lock_val;
    lock_conf.m_acq_en = 1;
    lock_conf.m_rel_id = start_lock + 1;
    lock_conf.m_rel_val = 1;
  }
  // std::cout << "   Lock Acq ID: " << uint32_t(lock_conf.m_acq_id) << " Lock
  // Rel ID: " << uint32_t(lock_conf.m_rel_id) << "\n";

  auto bd1 = m_mem_dma->getBD(startBD);
  bd1->updateAddr(524288 + ping_addr)
      ->updateLength(len)
      ->updateTensorDims(tensor)
      ->updateIterationDims(iter)
      ->enableNextBD()
      ->updateNextBD(nextBD)
      ->updateLockConfigs(lock_conf)
      ->write(loc, ncols);

  // 2. Pong
  startBD = start_bd + 1;
  nextBD = start_bd + 2;
  if (is_input) {
    lock_conf.m_acq_id = start_lock + 3;
    lock_conf.m_acq_val = 0;
    lock_conf.m_rel_id = start_lock + 2;
    lock_conf.m_rel_val = 1;
  } else {
    lock_conf.m_acq_id = start_lock + 2;
    lock_conf.m_acq_val = lock_val;
    lock_conf.m_rel_id = start_lock + 3;
    lock_conf.m_rel_val = 1;
  }
  // std::cout << "   Lock Acq ID: " << uint32_t(lock_conf.m_acq_id) << " Lock
  // Rel ID: " << uint32_t(lock_conf.m_rel_id) << "\n";

  auto bd2 = m_mem_dma->getBD(startBD);
  bd2->updateAddr(524288 + pong_addr)
      ->updateLength(len)
      ->updateTensorDims(tensor)
      ->updateIterationDims(iter)
      ->enableNextBD()
      ->updateNextBD(nextBD)
      ->updateLockConfigs(lock_conf)
      ->write(loc, ncols);

  // 3. Ping
  startBD = start_bd + 2;
  nextBD = start_bd + 3;
  if (is_input) {
    lock_conf.m_acq_id = start_lock + 1;
    lock_conf.m_acq_val = lock_val;
    lock_conf.m_rel_id = start_lock + 0;
    lock_conf.m_rel_val = -1;
  } else {
    lock_conf.m_acq_id = start_lock + 0;
    lock_conf.m_acq_val = 0;
    lock_conf.m_rel_id = start_lock + 1;
    lock_conf.m_rel_val = -1;
  }
  // std::cout << "   Lock Acq ID: " << uint32_t(lock_conf.m_acq_id) << " Lock
  // Rel ID: " << uint32_t(lock_conf.m_rel_id) << "\n";

  auto bd3 = m_mem_dma->getBD(startBD);
  bd3->updateAddr(524288 + ping_addr)
      ->updateLength(len)
      ->updateTensorDims(tensor)
      ->updateIterationDims(iter)
      ->enableNextBD()
      ->updateNextBD(nextBD)
      ->updateLockConfigs(lock_conf)
      ->write(loc, ncols);

  // 4. Pong
  startBD = start_bd + 3;
  nextBD = start_bd + 0;
  if (is_input) {
    lock_conf.m_acq_id = start_lock + 3;
    lock_conf.m_acq_val = lock_val;
    lock_conf.m_rel_id = start_lock + 2;
    lock_conf.m_rel_val = -1;
  } else {
    lock_conf.m_acq_id = start_lock + 2;
    lock_conf.m_acq_val = 0;
    lock_conf.m_rel_id = start_lock + 3;
    lock_conf.m_rel_val = -1;
  }
  // std::cout << "   Lock Acq ID: " << uint32_t(lock_conf.m_acq_id) << " Lock
  // Rel ID: " << uint32_t(lock_conf.m_rel_id) << "\n";

  auto bd4 = m_mem_dma->getBD(startBD);
  bd4->updateAddr(524288 + pong_addr)
      ->updateLength(len)
      ->updateTensorDims(tensor)
      ->updateIterationDims(iter)
      ->enableNextBD()
      ->updateNextBD(nextBD)
      ->updateLockConfigs(lock_conf)
      ->write(loc, ncols);

#ifdef USE_VISITOR
  components.push_back(bd1->getUniqueId());
  components.push_back(bd2->getUniqueId());
  components.push_back(bd3->getUniqueId());
  components.push_back(bd4->getUniqueId());
#else
  m_instr_counter +=
      m_instr_writer->write(bd1, m_instr_buffer + m_instr_counter);
  m_instr_counter +=
      m_instr_writer->write(bd2, m_instr_buffer + m_instr_counter);
  m_instr_counter +=
      m_instr_writer->write(bd3, m_instr_buffer + m_instr_counter);
  m_instr_counter +=
      m_instr_writer->write(bd4, m_instr_buffer + m_instr_counter);
#endif

  // Start Queue
  if (is_input) {
    // std::cout << "START QUEUE S2MM " << uint32_t(channel) << "\n";
    auto queue = m_mem_dma->getS2MMQueue(channel);
    queue->setStartBd(start_bd)->setRepeatCount(repeat)->write(loc);
#ifdef USE_VISITOR
    components.push_back(queue->getUniqueId());
#else
    m_instr_counter +=
        m_instr_writer->write(queue, m_instr_buffer + m_instr_counter);
#endif
  } else {
    // std::cout << "START QUEUE MM2S " << uint32_t(channel) << "\n";
    auto queue = m_mem_dma->getMM2SQueue(channel);
    queue->setStartBd(start_bd)->setRepeatCount(repeat)->write(loc);
#ifdef USE_VISITOR
    components.push_back(queue->getUniqueId());
#else
    m_instr_counter +=
        m_instr_writer->write(queue, m_instr_buffer + m_instr_counter);
#endif
  }
}

template <class Derived>
void InstructionCompiler<Derived>::generateInitAieDma(
    std::vector<uint32_t>& components) {
  // AIE Tile Dma Init
  for (uint32_t ct = 0; ct < m_num_at; ct++) {
    // DMA channel Reset/Un-reset
    AIE2::Location at_location(0, 2 + ct);
    for (auto& ch : m_at_s2mm_chs) { // S2MM
      // std::cout << "RESET CHANNEL S2MM " << uint32_t(ch) << "\n";
      auto ch_r = m_aie_dma->getS2MMCtrl(ch);
      ch_r->reset()->write(at_location); // Reset
#ifdef USE_VISITOR
      components.push_back(ch_r->getUniqueId());
#else
      m_instr_counter +=
          m_instr_writer->write(ch_r, m_instr_buffer + m_instr_counter);
#endif
      // std::cout << "UNRESET CHANNEL S2MM " << uint32_t(ch) << "\n";
      auto ch_u = m_aie_dma->getS2MMCtrl(ch);
      ch_u->unreset()->write(at_location); // Unreset
#ifdef USE_VISITOR
      components.push_back(ch_u->getUniqueId());
#else
      m_instr_counter +=
          m_instr_writer->write(ch_u, m_instr_buffer + m_instr_counter);
#endif
    }
    for (auto& ch : m_at_mm2s_chs) { // MM2S
      // std::cout << "RESET CHANNEL MM2S " << uint32_t(ch) << "\n";
      auto ch_r = m_aie_dma->getMM2SCtrl(ch);
      ch_r->reset()->write(at_location); // Reset
#ifdef USE_VISITOR
      components.push_back(ch_r->getUniqueId());
#else
      m_instr_counter +=
          m_instr_writer->write(ch_r, m_instr_buffer + m_instr_counter);
#endif
      // std::cout << "UNRESET CHANNEL MM2S " << uint32_t(ch) << "\n";
      auto ch_u = m_aie_dma->getMM2SCtrl(ch);
      ch_u->unreset()->write(at_location); // Unreset
#ifdef USE_VISITOR
      components.push_back(ch_u->getUniqueId());
#else
      m_instr_counter +=
          m_instr_writer->write(ch_u, m_instr_buffer + m_instr_counter);
#endif
    }

    // Set initial lock values for core-tile
    for (auto& lockId : m_at_locks) {
      // std::cout << "SET LOCK " << uint32_t(lock) << " VALUE 0\n";
      auto lock = m_aie_dma->getLock(lockId);
      lock->setValue(0)->write(at_location);
#ifdef USE_VISITOR
      components.push_back(lock->getUniqueId());
#else
      m_instr_counter +=
          m_instr_writer->write(lock, m_instr_buffer + m_instr_counter);
#endif
    }
  }
  // std::cout << "-- AIE DMA init done ...\n";
}

template <class Derived>
void InstructionCompiler<Derived>::generateAieDma(
    std::vector<uint32_t>& components) {
  // AIE Tile Dma Config
  // std::cout << "m_num_at : " << m_num_at << std::endl;
  for (uint32_t ct = 0; ct < m_num_at; ct++) {
    std::uint8_t ncols = 1, nrows = 1;

    AIE2::DmaTensor4D tensor;
    AIE2::DmaIteration iter;

    AIE2::Location at_location(0, 2 + ct);

    AIE2::LockConfig lock_conf(0, -1, 1, 1, 1);
    //// Input
    // Set new lock value for lock 0
    // std::cout << "SET LOCK " << uint32_t(0) << " VALUE 2\n";
    auto lock0 = m_aie_dma->getLock(0);
    lock0->setValue(2)->write(at_location);

    auto bd0 = m_aie_dma->getBD(0);
    auto bd1 = m_aie_dma->getBD(1);

    // Set Core tile BDs, (input, ping)
    bd0->updateAddr(m_ping_addr_in)
        ->updateLength(m_tile_window_size_in_total)
        ->updateTensorDims(tensor)
        ->updateIterationDims(iter)
        ->enableNextBD()
        ->updateNextBD(1)
        ->updateLockConfigs(lock_conf)
        ->write(at_location, ncols, nrows);
    // Set Core tile BDs, (input, pong)
    bd1->updateAddr(m_pong_addr_in)
        ->updateLength(m_tile_window_size_in_total)
        ->updateTensorDims(tensor)
        ->updateIterationDims(iter)
        ->enableNextBD()
        ->updateNextBD(0)
        ->updateLockConfigs(lock_conf)
        ->write(at_location, ncols, nrows);

    // Start Queue
    // std::cout << "START QUEUE S2MM " << uint32_t(0) << "\n";
    auto queue_s2mm = m_aie_dma->getS2MMQueue(0);
    queue_s2mm->setStartBd(0)->setRepeatCount(1)->write(at_location);

#ifdef USE_VISITOR
    components.push_back(lock0->getUniqueId());
    components.push_back(bd0->getUniqueId());
    components.push_back(bd1->getUniqueId());
    components.push_back(queue_s2mm->getUniqueId());
#else
    m_instr_counter +=
        m_instr_writer->write(lock0, m_instr_buffer + m_instr_counter);
    m_instr_counter +=
        m_instr_writer->write(bd0, m_instr_buffer + m_instr_counter);
    m_instr_counter +=
        m_instr_writer->write(bd1, m_instr_buffer + m_instr_counter);
    m_instr_counter +=
        m_instr_writer->write(queue_s2mm, m_instr_buffer + m_instr_counter);
#endif

    lock_conf = AIE2::LockConfig(3, -1, 1, 2, 1);
    //// Output
    // Set new lock value for lock 2
    // std::cout << "SET LOCK " << uint32_t(2) << " VALUE 2\n";
    auto lock2 = m_aie_dma->getLock(2);
    lock2->setValue(2)->write(at_location);

    auto bd2 = m_aie_dma->getBD(2);
    auto bd3 = m_aie_dma->getBD(3);
    // Set Core tile BDs, (output, ping)
    bd2->updateAddr(m_ping_addr_out)
        ->updateLength(m_tile_window_size_out_total)
        ->updateTensorDims(tensor)
        ->updateIterationDims(iter)
        ->enableNextBD()
        ->updateNextBD(3)
        ->updateLockConfigs(lock_conf)
        ->write(at_location, ncols, nrows);
    // Set Core tile BDs, (output, pong)
    bd3->updateAddr(m_pong_addr_out)
        ->updateLength(m_tile_window_size_out_total)
        ->updateTensorDims(tensor)
        ->updateIterationDims(iter)
        ->enableNextBD()
        ->updateNextBD(2)
        ->updateLockConfigs(lock_conf)
        ->write(at_location, ncols, nrows);

    // Start Queue
    // std::cout << "START QUEUE MM2S " << uint32_t(0) << "\n";
    auto queue_mm2s = m_aie_dma->getMM2SQueue(0);
    queue_mm2s->setStartBd(2)->setRepeatCount(1)->write(at_location);

#ifdef USE_VISITOR
    components.push_back(lock2->getUniqueId());
    components.push_back(bd2->getUniqueId());
    components.push_back(bd3->getUniqueId());
    components.push_back(queue_mm2s->getUniqueId());
#else
    m_instr_counter +=
        m_instr_writer->write(lock2, m_instr_buffer + m_instr_counter);
    m_instr_counter +=
        m_instr_writer->write(bd2, m_instr_buffer + m_instr_counter);
    m_instr_counter +=
        m_instr_writer->write(bd3, m_instr_buffer + m_instr_counter);
    m_instr_counter +=
        m_instr_writer->write(queue_mm2s, m_instr_buffer + m_instr_counter);
#endif

    // RTP write
    // std::cout << (m_rtp_size/ sizeof(int32_t)) << std::endl;
    for (int i = 0; i < m_rtp_size / sizeof(int32_t); ++i) {
      auto word = m_aie_dma->getIndexWord((m_rtp_addr + (i * 4)), i + 1);
      word->write(at_location);
#ifdef USE_VISITOR
      components.push_back(word->getUniqueId());
#else
      m_instr_counter +=
          m_instr_writer->write(word, m_instr_buffer + m_instr_counter);
#endif
    }

    std::uint32_t addr = 0, data = 0;
    // Offset write
    for (int i = 0; i < 8; i++) {
      auto word = m_aie_dma->getWord((m_rtp_addr + m_in_ind) + (i * 4),
                                     ((uint32_t*)(m_in_offsets))[i]);
      word->write(at_location);
#ifdef USE_VISITOR
      components.push_back(word->getUniqueId());
#else
      m_instr_counter +=
          m_instr_writer->write(word, m_instr_buffer + m_instr_counter);
#endif
    }
    for (int i = 0; i < 8; i++) {
      auto word = m_aie_dma->getWord((m_rtp_addr + m_out_ind) + (i * 4),
                                     ((uint32_t*)(m_out_offsets))[i]);
      word->write(at_location);
#ifdef USE_VISITOR
      components.push_back(word->getUniqueId());
#else
      m_instr_counter +=
          m_instr_writer->write(word, m_instr_buffer + m_instr_counter);
#endif
    }
    for (int i = 0; i < 8; i++) {
      auto word = m_aie_dma->getWord((m_rtp_addr + m_buf_ind) + (i * 4),
                                     ((uint32_t*)(m_buf_offsets))[i]);
      word->write(at_location);
#ifdef USE_VISITOR
      components.push_back(word->getUniqueId());
#else
      m_instr_counter +=
          m_instr_writer->write(word, m_instr_buffer + m_instr_counter);
#endif
    }
  }
  // std::cout << "-- AIE DMA config done ...\n";
}

template <class Derived>
void InstructionCompiler<Derived>::generateAieDma1(
    std::vector<uint32_t>& components) {
  // AIE Tile Dma Config
  // std::cout << "m_num_at : " << m_num_at << std::endl;
  for (uint32_t ct = 0; ct < m_num_at; ct++) {
    std::uint8_t ncols = 1, nrows = 1;

    AIE2::DmaTensor4D tensor;
    AIE2::DmaIteration iter;

    AIE2::Location at_location(0, 2 + ct);

    AIE2::LockConfig lock_conf(0, -1, 1, 1, 1);
    //// Input
    // Set new lock value for lock 0
    // std::cout << "SET LOCK " << uint32_t(0) << " VALUE 2\n";
    auto lock0 = m_aie_dma->getLock(0);
    lock0->setValue(2)->write(at_location);

    auto bd0 = m_aie_dma->getBD(0);
    auto bd1 = m_aie_dma->getBD(1);

    // Set Core tile BDs, (input, ping)
    bd0->updateAddr(m_ping_addr_in)
        ->updateLength(m_tile_window_size_in_total / 2)
        ->updateTensorDims(tensor)
        ->updateIterationDims(iter)
        ->enableNextBD()
        ->updateNextBD(1)
        ->updateLockConfigs(lock_conf)
        ->write(at_location, ncols, nrows);
    // Set Core tile BDs, (input, pong)
    bd1->updateAddr(m_pong_addr_in)
        ->updateLength(m_tile_window_size_in_total / 2)
        ->updateTensorDims(tensor)
        ->updateIterationDims(iter)
        ->enableNextBD()
        ->updateNextBD(0)
        ->updateLockConfigs(lock_conf)
        ->write(at_location, ncols, nrows);

    // Start Queue
    // std::cout << "START QUEUE S2MM " << uint32_t(0) << "\n";
    auto queue_s2mm = m_aie_dma->getS2MMQueue(0);
    queue_s2mm->setStartBd(0)->setRepeatCount(1)->write(at_location);

#ifdef USE_VISITOR
    components.push_back(lock0->getUniqueId());
    components.push_back(bd0->getUniqueId());
    components.push_back(bd1->getUniqueId());
    components.push_back(queue_s2mm->getUniqueId());
#else
    m_instr_counter +=
        m_instr_writer->write(lock0, m_instr_buffer + m_instr_counter);
    m_instr_counter +=
        m_instr_writer->write(bd0, m_instr_buffer + m_instr_counter);
    m_instr_counter +=
        m_instr_writer->write(bd1, m_instr_buffer + m_instr_counter);
    m_instr_counter +=
        m_instr_writer->write(queue_s2mm, m_instr_buffer + m_instr_counter);
#endif

    lock_conf = AIE2::LockConfig(3, -1, 1, 2, 1);
    //// Output
    // Set new lock value for lock 2
    // std::cout << "SET LOCK " << uint32_t(2) << " VALUE 2\n";
    auto lock2 = m_aie_dma->getLock(2);
    lock2->setValue(2)->write(at_location);

    auto bd2 = m_aie_dma->getBD(2);
    auto bd3 = m_aie_dma->getBD(3);
    // Set Core tile BDs, (output, ping)
    bd2->updateAddr(m_ping_addr_out)
        ->updateLength(m_tile_window_size_out_total / 2)
        ->updateTensorDims(tensor)
        ->updateIterationDims(iter)
        ->enableNextBD()
        ->updateNextBD(3)
        ->updateLockConfigs(lock_conf)
        ->write(at_location, ncols, nrows);
    // Set Core tile BDs, (output, pong)
    bd3->updateAddr(m_pong_addr_out)
        ->updateLength(m_tile_window_size_out_total / 2)
        ->updateTensorDims(tensor)
        ->updateIterationDims(iter)
        ->enableNextBD()
        ->updateNextBD(2)
        ->updateLockConfigs(lock_conf)
        ->write(at_location, ncols, nrows);

    // Start Queue
    // std::cout << "START QUEUE MM2S " << uint32_t(0) << "\n";
    auto queue_mm2s = m_aie_dma->getMM2SQueue(0);
    queue_mm2s->setStartBd(2)->setRepeatCount(1)->write(at_location);

#ifdef USE_VISITOR
    components.push_back(lock2->getUniqueId());
    components.push_back(bd2->getUniqueId());
    components.push_back(bd3->getUniqueId());
    components.push_back(queue_mm2s->getUniqueId());
#else
    m_instr_counter +=
        m_instr_writer->write(lock2, m_instr_buffer + m_instr_counter);
    m_instr_counter +=
        m_instr_writer->write(bd2, m_instr_buffer + m_instr_counter);
    m_instr_counter +=
        m_instr_writer->write(bd3, m_instr_buffer + m_instr_counter);
    m_instr_counter +=
        m_instr_writer->write(queue_mm2s, m_instr_buffer + m_instr_counter);
#endif

    // RTP write
    // std::cout << (m_rtp_size/ sizeof(int32_t)) << std::endl;
    for (int i = 0; i < m_rtp_size / sizeof(int32_t); ++i) {
      auto word = m_aie_dma->getIndexWord((m_rtp_addr + (i * 4)), i + 1);
      word->write(at_location);
#ifdef USE_VISITOR
      components.push_back(word->getUniqueId());
#else
      m_instr_counter +=
          m_instr_writer->write(word, m_instr_buffer + m_instr_counter);
#endif
    }

    std::uint32_t addr = 0, data = 0;
    // Offset write
    for (int i = 0; i < 8; i++) {
      auto word = m_aie_dma->getWord((m_rtp_addr + m_in_ind) + (i * 4),
                                     ((uint32_t*)(m_in_offsets))[i]);
      word->write(at_location);
#ifdef USE_VISITOR
      components.push_back(word->getUniqueId());
#else
      m_instr_counter +=
          m_instr_writer->write(word, m_instr_buffer + m_instr_counter);
#endif
    }
    for (int i = 0; i < 8; i++) {
      // std::cout << ((uint32_t*)(m_out_offsets))[i] << std::endl;
      auto word = m_aie_dma->getWord((m_rtp_addr + m_out_ind) + (i * 4),
                                     ((uint32_t*)(m_out_offsets))[i]);
      word->write(at_location);
#ifdef USE_VISITOR
      components.push_back(word->getUniqueId());
#else
      m_instr_counter +=
          m_instr_writer->write(word, m_instr_buffer + m_instr_counter);
#endif
    }
    for (int i = 0; i < 8; i++) {
      auto word = m_aie_dma->getWord((m_rtp_addr + m_buf_ind) + (i * 4),
                                     ((uint32_t*)(m_buf_offsets))[i]);
      word->write(at_location);
#ifdef USE_VISITOR
      components.push_back(word->getUniqueId());
#else
      m_instr_counter +=
          m_instr_writer->write(word, m_instr_buffer + m_instr_counter);
#endif
    }
  }
  // std::cout << "-- AIE DMA config done ...\n";
}

template <class Derived>
void InstructionCompiler<Derived>::generateInitMemDma(
    std::vector<uint32_t>& components) {
  // Mem Tile Dma Init
  AIE2::Location mt_location(0, 1);
  // DMA channel Reset/Un-reset
  for (auto& ch : m_mt_s2mm_chs) { // S2MM
    // std::cout << "RESET CHANNEL S2MM " << uint32_t(ch) << "\n";
    auto ch_r = m_mem_dma->getS2MMCtrl(ch);
    ch_r->reset()->write(mt_location); // reset
#ifdef USE_VISITOR
    components.push_back(ch_r->getUniqueId());
#else
    m_instr_counter +=
        m_instr_writer->write(ch_r, m_instr_buffer + m_instr_counter);
#endif

    // std::cout << "UNRESET CHANNEL S2MM " << uint32_t(ch) << "\n";
    auto ch_u = m_mem_dma->getS2MMCtrl(ch);
    ch_u->unreset()->write(mt_location); // unreset

#ifdef USE_VISITOR
    components.push_back(ch_u->getUniqueId());
#else
    m_instr_counter +=
        m_instr_writer->write(ch_u, m_instr_buffer + m_instr_counter);
#endif
  }
  for (auto& ch : m_mt_mm2s_chs) { // MM2S
    // std::cout << "RESET CHANNEL MM2S " << uint32_t(ch) << "\n";
    auto ch_r = m_mem_dma->getMM2SCtrl(ch);
    ch_r->reset()->write(mt_location); // reset
#ifdef USE_VISITOR
    components.push_back(ch_r->getUniqueId());
#else
    m_instr_counter +=
        m_instr_writer->write(ch_r, m_instr_buffer + m_instr_counter);
#endif

    // std::cout << "UNRESET CHANNEL MM2S " << uint32_t(ch) << "\n";
    auto ch_u = m_mem_dma->getMM2SCtrl(ch);
    ch_u->unreset()->write(mt_location); // unreset

#ifdef USE_VISITOR
    components.push_back(ch_u->getUniqueId());
#else
    m_instr_counter +=
        m_instr_writer->write(ch_u, m_instr_buffer + m_instr_counter);
#endif
  }

  // Set initial lock values for core-tile
  for (auto& lockId : m_mt_locks) {
    // std::cout << "SET LOCK " << uint32_t(lock) << " VALUE 0\n";
    auto lock = m_mem_dma->getLock(lockId);
    lock->setValue(0)->write(mt_location);
#ifdef USE_VISITOR
    components.push_back(lock->getUniqueId());
#else
    m_instr_counter +=
        m_instr_writer->write(lock, m_instr_buffer + m_instr_counter);
#endif
  }
  // std::cout << "-- Mem DMA init done ...\n";
}

template <class Derived>
void InstructionCompiler<Derived>::generateMemDma(
    std::vector<uint32_t>& components) {
  // Mem Tile Dma Init
  AIE2::Location mt_location(0, 1);

  // Mem Tile Dma Config
  m_ping_mem_in = 0;
  m_pong_mem_in = m_ping_mem_in + (m_tile_window_size_in_total * 4);
  m_ping_mem_out = m_pong_mem_in + (m_tile_window_size_in_total * 4);
  m_pong_mem_out = m_ping_mem_out + (m_tile_window_size_out_total * 4);

  uint8_t start_bds[] = {0, 24, 4, 28, 8, 32, 12, 36, 16, 40, 20, 44};
  uint8_t channels[] = {0, 1, 0, 1, 2, 3, 2, 3, 4, 5, 4, 5};
  uint8_t start_locks[] = {64, 64, 64, 64, 64, 64, 68, 68, 68, 68, 68, 68};
  uint8_t lock_value[] = {4, 4, 2, 2, 2, 2, 1, 1, 1, 1, 4, 4};
  uint8_t repeat = 1;

  // Input/Output ports for mem
  int s2mm_input_shm = 2;
  int s2mm_input_aie = 4;
  int mm2s_output_shm = m_out_ports;
  int mm2s_output_aie = 4;

  int idx = 0;

  // Input coming from Shim
  for (int i = 0; i < s2mm_input_shm; ++i) {
    uint32_t offset = i * m_tile_window_size_in[0];
    AIE2::DmaTensor4D tensor = {{{1, (m_tile_window_size_in[i] >> 2)},
                                 {(m_tile_window_size_in_total >> 2), 4},
                                 {1, 0},
                                 {1, 0}}};
    generateMTPingPongConfigs(
        components, m_ping_mem_in + offset, m_pong_mem_in + offset,
        4 * m_tile_window_size_in[i], tensor, start_bds[idx], start_locks[idx],
        lock_value[idx], channels[idx], repeat, mt_location, true);
    idx++;
  }

  AIE2::DmaTensor4D tensor = {{{1, 0}, {1, 0}, {1, 0}, {1, 0}}};

  // Output going to AIE
  for (int i = 0; i < mm2s_output_aie; ++i) {
    uint32_t offset = i * m_tile_window_size_in_total;
    generateMTPingPongConfigs(
        components, m_ping_mem_in + offset, m_pong_mem_in + offset,
        m_tile_window_size_in_total, tensor, start_bds[idx], start_locks[idx],
        lock_value[idx], channels[idx], repeat, mt_location, false);
    idx++;
  }

  // Input coming from AIE
  for (int i = 0; i < s2mm_input_aie; ++i) {
    uint32_t offset = i * m_tile_window_size_out_total;
    generateMTPingPongConfigs(
        components, m_ping_mem_out + offset, m_pong_mem_out + offset,
        m_tile_window_size_out_total, tensor, start_bds[idx], start_locks[idx],
        lock_value[idx], channels[idx], repeat, mt_location, true);
    idx++;
  }

  // Output going to Shim
  for (int i = 0; i < mm2s_output_shm; ++i) {
    uint32_t offset = i * m_tile_window_size_out[0];
    generateMTPingPongConfigs(
        components, m_ping_mem_out + offset, m_pong_mem_out + offset,
        4 * m_tile_window_size_out[i], tensor, start_bds[idx], start_locks[idx],
        lock_value[idx], channels[idx], repeat, mt_location, false);
    idx++;
  }

  // std::cout << "-- Mem DMA config done ...\n";
}
template <class Derived>
void InstructionCompiler<Derived>::generateMemDma1(
    std::vector<uint32_t>& components) {
  // Mem Tile Dma Init
  AIE2::Location mt_location(0, 1);

  // Mem Tile Dma Config
  m_ping_mem_in = 0;
  m_pong_mem_in = m_ping_mem_in + (m_tile_window_size_in_total * 4);
  m_ping_mem_out = m_pong_mem_in + (m_tile_window_size_in_total * 4);
  m_pong_mem_out = m_ping_mem_out + (m_tile_window_size_out_total * 4);

  uint8_t start_bds[] = {0, 24, 4, 28, 8, 32, 12, 36, 16, 40, 20, 44};
  uint8_t channels[] = {0, 1, 0, 1, 2, 3, 2, 3, 4, 5, 4, 5};
  uint8_t start_locks[] = {64, 64, 64, 64, 64, 64, 68, 68, 68, 68, 68, 68};
  uint8_t lock_value[] = {4, 4, 2, 2, 2, 2, 1, 1, 1, 1, 4, 4};
  uint8_t repeat = 1;

  // Input/Output ports for mem
  int s2mm_input_shm = 2;
  int s2mm_input_aie = 4;
  int mm2s_output_shm = m_out_ports;
  int mm2s_output_aie = 4;

  int idx = 0;

  // Input coming from Shim
  for (int i = 0; i < s2mm_input_shm; ++i) {
    uint32_t offset = i * m_tile_window_size_in[0];
    AIE2::DmaTensor4D tensor = {{{1, (m_tile_window_size_in[i] >> 3)},
                                 {(m_tile_window_size_in[i] >> 3), 2},
                                 {(m_tile_window_size_in_total >> 2), 4},
                                 {1, 0}}};
    generateMTPingPongConfigs(
        components, m_ping_mem_in + offset, m_pong_mem_in + offset,
        4 * m_tile_window_size_in[i], tensor, start_bds[idx], start_locks[idx],
        lock_value[idx], channels[idx], repeat, mt_location, true);
    idx++;
  }

  AIE2::DmaTensor4D tensor = {{{1, 0}, {1, 0}, {1, 0}, {1, 0}}};

  // Output going to AIE
  for (int i = 0; i < mm2s_output_aie; ++i) {
    uint32_t offset = i * m_tile_window_size_in_total;
    generateMTPingPongConfigs(
        components, m_ping_mem_in + offset, m_pong_mem_in + offset,
        m_tile_window_size_in_total, tensor, start_bds[idx], start_locks[idx],
        lock_value[idx], channels[idx], repeat, mt_location, false);
    idx++;
  }

  // Input coming from AIE
  for (int i = 0; i < s2mm_input_aie; ++i) {
    uint32_t offset = i * m_tile_window_size_out_total;
    generateMTPingPongConfigs(
        components, m_ping_mem_out + offset, m_pong_mem_out + offset,
        m_tile_window_size_out_total, tensor, start_bds[idx], start_locks[idx],
        lock_value[idx], channels[idx], repeat, mt_location, true);
    idx++;
  }

  // Output going to Shim
  for (int i = 0; i < mm2s_output_shm; ++i) {
    uint32_t offset = i * m_tile_window_size_out[0];
    generateMTPingPongConfigs(
        components, m_ping_mem_out + offset, m_pong_mem_out + offset,
        4 * m_tile_window_size_out[i], tensor, start_bds[idx], start_locks[idx],
        lock_value[idx], channels[idx], repeat, mt_location, false);
    idx++;
  }

  // std::cout << "-- Mem DMA config done ...\n";
}
template <class Derived>
void InstructionCompiler<Derived>::generateMemDma2(
    std::vector<uint32_t>& components) {
  // Mem Tile Dma Init
  AIE2::Location mt_location(0, 1);

  // Mem Tile Dma Config
  m_ping_mem_in = 0;
  m_pong_mem_in = m_ping_mem_in + (m_tile_window_size_in_total * 2);
  m_ping_mem_out = m_pong_mem_in + (m_tile_window_size_in_total * 2);
  m_pong_mem_out = m_ping_mem_out + (m_tile_window_size_out_total * 2);

  uint8_t start_bds[] = {0, 24, 4, 28, 8, 32, 12, 36, 16, 40, 20, 44};
  uint8_t channels[] = {0, 1, 0, 1, 2, 3, 2, 3, 4, 5, 4, 5};
  uint8_t start_locks[] = {64, 64, 64, 64, 64, 64, 68, 68, 68, 68, 68, 68};
  uint8_t lock_value[] = {4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4};
  uint8_t repeat = 1;

  // Input/Output ports for mem
  int s2mm_input_shm = 2;
  int s2mm_input_aie = 4;
  int mm2s_output_shm = 2;
  int mm2s_output_aie = 4;

  int idx = 0;

  // Input coming from Shim
  AIE2::DmaTensor4D tensor = {{{1, 0}, {1, 0}, {1, 0}, {1, 0}}};
  for (int i = 0; i < s2mm_input_shm; ++i) {
    uint32_t offset = i * m_tile_window_size_in_total;
    generateMTPingPongConfigs(
        components, m_ping_mem_in + offset, m_pong_mem_in + offset,
        m_tile_window_size_in_total, tensor, start_bds[idx], start_locks[idx],
        lock_value[idx], channels[idx], repeat, mt_location, true);
    idx++;
  }

  // Output going to AIE
  for (int i = 0; i < mm2s_output_aie; ++i) {
    uint32_t offset = i * (m_tile_window_size_in_total / 2);
    generateMTPingPongConfigs(components, m_ping_mem_in + offset,
                              m_pong_mem_in + offset,
                              (m_tile_window_size_in_total / 2), tensor,
                              start_bds[idx], start_locks[idx], lock_value[idx],
                              channels[idx], repeat, mt_location, false);
    idx++;
  }

  // Input coming from AIE
  for (int i = 0; i < s2mm_input_aie; ++i) {
    uint32_t offset = i * (m_tile_window_size_out_total / 2);
    generateMTPingPongConfigs(components, m_ping_mem_out + offset,
                              m_pong_mem_out + offset,
                              (m_tile_window_size_out_total / 2), tensor,
                              start_bds[idx], start_locks[idx], lock_value[idx],
                              channels[idx], repeat, mt_location, true);
    idx++;
  }

  // Output going to Shim
  for (int i = 0; i < mm2s_output_shm; ++i) {
    uint32_t offset = i * m_tile_window_size_out_total;
    generateMTPingPongConfigs(
        components, m_ping_mem_out + offset, m_pong_mem_out + offset,
        m_tile_window_size_out_total, tensor, start_bds[idx], start_locks[idx],
        lock_value[idx], channels[idx], repeat, mt_location, false);
    idx++;
  }

  // std::cout << "-- Mem DMA config done ...\n";
}

template <class Derived>
void InstructionCompiler<Derived>::generateMemDma_singlecore(
    std::vector<uint32_t>& components) {
  // Mem Tile Dma Init
  AIE2::Location mt_location(0, 1);

  // Mem Tile Dma Config
  m_ping_mem_in = 0;
  m_pong_mem_in = m_ping_mem_in + (m_tile_window_size_in_total);
  m_ping_mem_out = m_pong_mem_in + (m_tile_window_size_in_total);
  m_pong_mem_out = m_ping_mem_out + (m_tile_window_size_out_total);

  uint8_t start_bds[] = {0, 24, 4, 28, 8, 32, 12, 36, 16, 40, 20, 44};
  uint8_t channels[] = {0, 1, 0, 1, 2, 3, 2, 3, 4, 5, 4, 5};
  uint8_t start_locks[] = {64, 64, 64, 64, 64, 64, 68, 68, 68, 68, 68, 68};
  uint8_t lock_value[] = {4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4};
  uint8_t repeat = 1;

  // Input/Output ports for mem
  int s2mm_input_shm = 2;
  int s2mm_input_aie = 4;
  int mm2s_output_shm = 2;
  int mm2s_output_aie = 4;

  int idx = 0;

  // Input coming from Shim
  for (int i = 0; i < s2mm_input_shm; ++i) {
    uint32_t offset = i * (m_tile_window_size_in[0] / 2);
    AIE2::DmaTensor4D tensor = {{{1, 0}, {1, 0}, {1, 0}, {1, 0}}};

    generateMTPingPongConfigs(
        components, m_ping_mem_in + offset, m_pong_mem_in + offset,
        m_tile_window_size_in[0] / 2, tensor, start_bds[idx], start_locks[idx],
        lock_value[idx], channels[idx], repeat, mt_location, true);
    idx++;
  }

  AIE2::DmaTensor4D tensor = {{{1, 0}, {1, 0}, {1, 0}, {1, 0}}};

  // Output going to AIE
  for (int i = 0; i < mm2s_output_aie; ++i) {
    uint32_t offset = 0;
    generateMTPingPongConfigs(
        components, m_ping_mem_in + offset, m_pong_mem_in + offset,
        m_tile_window_size_in_total, tensor, start_bds[idx], start_locks[idx],
        lock_value[idx], channels[idx], repeat, mt_location, false);
    idx++;
  }

  // Input coming from AIE
  for (int i = 0; i < s2mm_input_aie; ++i) {
    uint32_t offset = 0;
    generateMTPingPongConfigs(
        components, m_ping_mem_out + offset, m_pong_mem_out + offset,
        m_tile_window_size_out_total, tensor, start_bds[idx], start_locks[idx],
        lock_value[idx], channels[idx], repeat, mt_location, true);
    idx++;
  }

  // Output going to Shim
  for (int i = 0; i < mm2s_output_shm; ++i) {
    uint32_t offset = i * (m_tile_window_size_out[0] / 2);
    generateMTPingPongConfigs(
        components, m_ping_mem_out + offset, m_pong_mem_out + offset,
        m_tile_window_size_out[0] / 2, tensor, start_bds[idx], start_locks[idx],
        lock_value[idx], channels[idx], repeat, mt_location, false);
    idx++;
  }

  // std::cout << "-- Mem DMA config done ...\n";
}

template <class Derived>
void InstructionCompiler<Derived>::generateMemDma_singlecore_singlechannel(
    std::vector<uint32_t>& components) {
  // Mem Tile Dma Init
  AIE2::Location mt_location(0, 1);

  // Mem Tile Dma Config
  m_ping_mem_in = 0;
  m_pong_mem_in = m_ping_mem_in + (m_tile_window_size_in_total);
  m_ping_mem_out = m_pong_mem_in + (m_tile_window_size_in_total);
  m_pong_mem_out = m_ping_mem_out + (m_tile_window_size_out_total);

  uint8_t start_bds[] = {0, 4, 28, 8, 32, 12, 36, 16, 40, 20};
  uint8_t channels[] = {0, 0, 1, 2, 3, 2, 3, 4, 5, 4};
  uint8_t start_locks[] = {64, 64, 64, 64, 64, 68, 68, 68, 68, 68};
  uint8_t lock_value[] = {4, 1, 1, 1, 1, 1, 1, 1, 1, 4};
  uint8_t repeat = 1;

  // Input/Output ports for mem
  int s2mm_input_shm = 1;
  int s2mm_input_aie = 4;
  int mm2s_output_shm = 1;
  int mm2s_output_aie = 4;

  int idx = 0;

  // Input coming from Shim
  for (int i = 0; i < s2mm_input_shm; ++i) {
    uint32_t offset = i * (m_tile_window_size_in[0] / 2);
    AIE2::DmaTensor4D tensor = {{{1, 0}, {1, 0}, {1, 0}, {1, 0}}};

    generateMTPingPongConfigs(
        components, m_ping_mem_in + offset, m_pong_mem_in + offset,
        m_tile_window_size_in_total, tensor, start_bds[idx], start_locks[idx],
        lock_value[idx], channels[idx], repeat, mt_location, true);
    idx++;
  }

  AIE2::DmaTensor4D tensor = {{{1, 0}, {1, 0}, {1, 0}, {1, 0}}};

  // Output going to AIE
  for (int i = 0; i < mm2s_output_aie; ++i) {
    uint32_t offset = 0;
    generateMTPingPongConfigs(
        components, m_ping_mem_in + offset, m_pong_mem_in + offset,
        m_tile_window_size_in_total, tensor, start_bds[idx], start_locks[idx],
        lock_value[idx], channels[idx], repeat, mt_location, false);
    idx++;
  }

  // Input coming from AIE
  for (int i = 0; i < s2mm_input_aie; ++i) {
    uint32_t offset = 0;
    generateMTPingPongConfigs(
        components, m_ping_mem_out + offset, m_pong_mem_out + offset,
        m_tile_window_size_out_total, tensor, start_bds[idx], start_locks[idx],
        lock_value[idx], channels[idx], repeat, mt_location, true);
    idx++;
  }

  // Output going to Shim
  for (int i = 0; i < mm2s_output_shm; ++i) {
    uint32_t offset = i * (m_tile_window_size_out[0] / 2);
    generateMTPingPongConfigs(
        components, m_ping_mem_out + offset, m_pong_mem_out + offset,
        m_tile_window_size_out_total, tensor, start_bds[idx], start_locks[idx],
        lock_value[idx], channels[idx], repeat, mt_location, false);
    idx++;
  }

  // std::cout << "-- Mem DMA config done ...\n";
}
