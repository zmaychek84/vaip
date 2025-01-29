/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */
#pragma once

#include "aie2_instr_ir_writer.hpp"
#include "aie2_ipu_debug_instr_writer.hpp"
#include "aie2_ipu_instr_writer.hpp"
#include "pp_instr_compiler_impl.hpp"

class TopkInstrCompiler : public InstructionCompiler<TopkInstrCompiler> {
public:
  // Ctor
  TopkInstrCompiler(std::uint32_t* instr_buffer)
      : InstructionCompiler(instr_buffer) {}
  // TODO:: Add const for the rtps buffer.
  int generate(std::uint16_t* rtps);
};

// TODO: create a separate copy of the rtps and must not change the one coming
// from app.
int TopkInstrCompiler::generate(std::uint16_t* rtps) {
  // Data from RTPs
  int input_elements = rtps[PP_TOPK_RTP_NUM_ELEM];
  int output_elements = rtps[PP_TOPK_RTP_K];
  int start_idx = rtps[PP_TOPK_RTP_START_IDX];

  // # input window size in one core (bytes)
  uint32_t TILE_WINDOW_SIZE_IN = (input_elements * sizeof(uint16_t));
  // # Total input size and we are using two input ports so div by 2
  uint32_t TILE_WINDOW_SIZE_IN1 = (input_elements * sizeof(uint16_t)) / 2;
  // # output window size in one core, multiply by 2 for indices and values
  // output
  uint32_t TILE_WINDOW_SIZE_OUT = (2 * output_elements * sizeof(uint16_t));
  // # Total output size and we are using two output ports so div by 2
  uint32_t TILE_WINDOW_SIZE_OUT1 = (2 * output_elements * sizeof(uint16_t)) / 2;

  int RTP_ADDR = 0x2000;

  // DMA components
  std::vector<uint32_t> dma_components;
  dma_components.reserve(1000);

  AIE2::TileMetaData shape_in = {TILE_WINDOW_SIZE_IN, 1, 1};
  // Fill this
  std::vector<AIE2::TileMetaData> tiles_in;

  tiles_in.emplace_back(shape_in);

  AIE2::TileMetaData shape_out = {TILE_WINDOW_SIZE_OUT, 1, 1};
  // Fill this
  std::vector<AIE2::TileMetaData> tiles_out;

  tiles_out.emplace_back(shape_out);

  // Initialize compiler
  init(tiles_in, tiles_out, TILE_WINDOW_SIZE_OUT1);
  ////
  // AIE and Mem Tile config
  ////
  generateInitMemDma(dma_components);
  generateInitAieDma(dma_components);
  generateAieDma(dma_components);
  generateMemDma_singlecore(dma_components);

  ////
  // shim tile config
  ////
  int num_tiles = 1;
  // TODO:: check if the param_buffer is really necessary
  // param_buffer[PP_RTP_NEXT_OPCODE_POS] = 0;
  rtps[PP_RTP_NUM_TILES_PER_CORE] = (num_tiles + 3) / 4;
  rtps[PP_RTP_NEXT_OPCODE_POS] = 0;
  for (int i = 0; i < 4; i++) {
    AIE2::Location at_location(0, 2 + i);
    auto word = m_aie_dma->getWord((m_rtp_addr + 2 * PP_RTP_NUM_TILES_PER_CORE),
                                   rtps[PP_RTP_NUM_TILES_PER_CORE]);
    word->write(at_location);
#ifdef USE_VISITOR
    dma_components.push_back(word->getUniqueId());
#else
    m_instr_counter +=
        m_instr_writer->write(word, m_instr_buffer + m_instr_counter);
#endif
  }

  // Set ShimDMA for input and output
  uint32_t s_tile_window_size_in_total = (input_elements * sizeof(uint16_t));
  uint32_t s_tile_window_size_out_total =
      (2 * output_elements * sizeof(uint16_t));
  uint16_t in_out_rows[2];
  uint32_t* io_row_ptr = (uint32_t*)(in_out_rows);

  AIE2::DmaIteration nt_iteration;
  AIE2::Location nt_location(0, 0);
  uint8_t nt_ncols = 1, ddr_type;

  AIE2::DmaTensor4D nt_out_tensor = {{{1, 0}, {1, 0}, {1, 0}, {1, 0}}};

  nt_iteration.m_current = 0;
  nt_iteration.m_step = 1;
  nt_iteration.m_wrap = 1;

  ddr_type = 1;

  // output BD
  // -------------------------------------------------------------------------
  uint32_t start_out_bd = 2;
  uint32_t repeat_out_bd = 1;

  auto outBD = m_noc_dma->getBD(start_out_bd);
  outBD->updateAddr(0, 0)->updateLength(TILE_WINDOW_SIZE_OUT1);
  outBD->updateTensorDims(nt_out_tensor)->updateIterationDims(nt_iteration);
  outBD->write(nt_location, nt_ncols, ddr_type);

  // task queue
  auto outTaskQ = m_noc_dma->getS2MMQueue(0);
  outTaskQ->setStartBd(start_out_bd)->setRepeatCount(repeat_out_bd);
  outTaskQ->enableTokenIssue()->write(nt_location);

  // output BD1
  // -------------------------------------------------------------------------
  start_out_bd = 3;
  repeat_out_bd = 1;

  auto outBD1 = m_noc_dma->getBD(start_out_bd);
  outBD1->updateAddr(TILE_WINDOW_SIZE_OUT1, 0)
      ->updateLength(TILE_WINDOW_SIZE_OUT1);
  outBD1->updateTensorDims(nt_out_tensor)->updateIterationDims(nt_iteration);
  outBD1->write(nt_location, nt_ncols, ddr_type);

  // task queue
  auto outTaskQ1 = m_noc_dma->getS2MMQueue(1);
  outTaskQ1->setStartBd(start_out_bd)->setRepeatCount(repeat_out_bd);
  outTaskQ1->enableTokenIssue()->write(nt_location);

#ifdef USE_VISITOR
  // Push to vector
  dma_components.push_back(outBD->getUniqueId());
  dma_components.push_back(outTaskQ->getUniqueId());
  dma_components.push_back(outBD1->getUniqueId());
  dma_components.push_back(outTaskQ1->getUniqueId());
#else
  m_instr_counter +=
      m_instr_writer->write(outBD, m_instr_buffer + m_instr_counter);
  m_instr_counter +=
      m_instr_writer->write(outTaskQ, m_instr_buffer + m_instr_counter);
#endif

  nt_iteration.m_current = 0;
  nt_iteration.m_step = 1;
  nt_iteration.m_wrap = 1;

  ddr_type = 0;

  // input BD
  // -------------------------------------------------------------------------
  uint32_t start_in_bd = 0;
  uint32_t repeat_in_bd = 1;

  auto inBD = m_noc_dma->getBD(start_in_bd);
  inBD->updateAddr(0, 0)->updateLength(TILE_WINDOW_SIZE_IN1);
  inBD->updateTensorDims(nt_out_tensor)->updateIterationDims(nt_iteration);
  inBD->write(nt_location, nt_ncols, ddr_type);

  // input BD1
  // -------------------------------------------------------------------------
  start_in_bd = 1;
  repeat_in_bd = 1;

  auto inBD1 = m_noc_dma->getBD(start_in_bd);
  inBD1->updateAddr(TILE_WINDOW_SIZE_IN1, 0)
      ->updateLength(TILE_WINDOW_SIZE_IN1);
  inBD1->updateTensorDims(nt_out_tensor)->updateIterationDims(nt_iteration);
  inBD1->write(nt_location, nt_ncols, ddr_type);

  // task queue
  auto inTaskQ = m_noc_dma->getMM2SQueue(0);
  inTaskQ->setStartBd(0)->setRepeatCount(repeat_in_bd);
  inTaskQ->write(nt_location);

  // task queue
  auto inTaskQ1 = m_noc_dma->getMM2SQueue(1);
  inTaskQ1->setStartBd(start_in_bd)->setRepeatCount(repeat_in_bd);
  inTaskQ1->write(nt_location);

#ifdef USE_VISITOR
  // Push to vector
  dma_components.push_back(inBD->getUniqueId());
  dma_components.push_back(inBD1->getUniqueId());
  dma_components.push_back(inTaskQ->getUniqueId());
  dma_components.push_back(inTaskQ1->getUniqueId());
#else
  m_instr_counter +=
      m_instr_writer->write(outBD, m_instr_buffer + m_instr_counter);
  m_instr_counter +=
      m_instr_writer->write(outTaskQ, m_instr_buffer + m_instr_counter);
#endif

  auto sync_word_1 = m_noc_dma->getSyncWord(AIE2::DmaDirection::S2MM, 0, 1,
                                            1); // (direction, ch, ncol, nrow)
  auto sync_word_2 = m_noc_dma->getSyncWord(AIE2::DmaDirection::S2MM, 1, 1,
                                            1); // (direction, ch, ncol, nrow)
  sync_word_1->write(nt_location);
  sync_word_1->write(nt_location);

#ifdef USE_VISITOR
  dma_components.push_back(sync_word_1->getUniqueId());
  dma_components.push_back(sync_word_2->getUniqueId());
#else
  m_instr_counter +=
      m_instr_writer->write(sync_word_1, m_instr_buffer + m_instr_counter);
  m_instr_counter +=
      m_instr_writer->write(sync_word_2, m_instr_buffer + m_instr_counter);
#endif

#ifdef USE_VISITOR
  // Visitors
  auto instrWriter = std::make_unique<IPUInstructionWriter>();
  // auto irWriter = std::make_unique<InstructionIRWriter>();
  auto dbgWriter = std::make_unique<IPUDebugInstrWriter>();
  // std::cout << "-- DMA Components: " << dma_components.size() << "\n";
  m_instr_counter = 0;
  for (auto uid : dma_components) {
    auto component = getDmaElement(uid);
    auto ncount =
        component->accept(instrWriter.get(), m_instr_buffer + m_instr_counter);
    auto ncount1 =
        component->accept(dbgWriter.get(), m_instr_buffer + m_instr_counter);
    // auto rval1 = component->accept(irWriter.get(), m_instr_buffer +
    // m_instr_counter);
    m_instr_counter += ncount;
  }
  // Write to console
  // dbgWriter->writeToConsole();
  dbgWriter->writeToFile("output_dbg.txt");
#endif
  // Return number of instructions generated
  return m_instr_counter;
}
