/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */
#pragma once

#include "aie2_instr_ir_writer.hpp"
#include "aie2_ipu_debug_instr_writer.hpp"
#include "aie2_ipu_instr_writer.hpp"
#include "pp_instr_compiler_impl.hpp"

class EGCPostInstrCompiler : public InstructionCompiler<EGCPostInstrCompiler> {
public:
  // Ctor
  EGCPostInstrCompiler(std::uint32_t* instr_buffer)
      : InstructionCompiler(instr_buffer) {}

  int generate(const std::uint16_t* rtps);
};

int EGCPostInstrCompiler::generate(const std::uint16_t* rtps) {
  // Data from RTPs
  uint16_t img_width_in = rtps[PP_RTP_IMG_WIDTH_IN];
  uint16_t img_width_out = rtps[PP_RTP_IMG_WIDTH_OUT];
  uint16_t img_height_out = rtps[PP_RTP_IMG_HEIGHT_OUT];
  uint16_t img_height_in = rtps[PP_RTP_IMG_HEIGHT_IN];

  uint16_t patch_x_out = rtps[PP_EGCPOST_RTP_PATCH_X_OUT];
  uint16_t patch_y_out = rtps[PP_EGCPOST_RTP_PATCH_Y_OUT];
  uint16_t patch_width_out = rtps[PP_EGCPOST_RTP_PATCH_WIDTH_OUT];
  uint16_t patch_height_out = rtps[PP_EGCPOST_RTP_PATCH_HEIGHT_OUT];

  uint32_t tile_width_in = img_width_in;
  uint32_t tile_height_in = 2;
  uint32_t tile_width_out = patch_width_out;
  uint32_t tile_height_out = 1;

  std::cout << "Tile IN " << tile_width_in << " x " << tile_height_in << "\n"
            << "Tile OUT " << tile_width_out << " x " << tile_height_out
            << "\n";

  uint32_t RTP_ADDR = 0x2000;

  // DMA components
  std::vector<uint32_t> dma_components;
  dma_components.reserve(1000);

  AIE2::TileMetaData shape_in = {tile_height_in, tile_width_in, 2};
  // Fill this
  std::vector<AIE2::TileMetaData> tiles_in;

  tiles_in.emplace_back(shape_in);
  tiles_in.emplace_back(shape_in);

  AIE2::TileMetaData shape_out_y = {tile_height_out, tile_width_out, 1};
  AIE2::TileMetaData shape_out_uv = {tile_height_out, tile_width_out, 1};
  // Fill this
  std::vector<AIE2::TileMetaData> tiles_out;

  tiles_out.emplace_back(shape_out_y);
  tiles_out.emplace_back(shape_out_uv);

  // Initialize compiler
  init(tiles_in, tiles_out);

  ////
  // AIE and Mem Tile config
  ////
  generateInitMemDma(dma_components);
  generateInitAieDma(dma_components);
  generateAieDma(dma_components);
  generateMemDma(dma_components);

  ////
  // shim tile config
  ////

  // Set ShimDMA for input and output
  float y_scale = (float)img_height_in / (float)patch_height_out;
  int num_tiles = patch_height_out / tile_height_out;

  constexpr int maxBDsPerBatch = 2;
  constexpr int numBDsPerBatch = 2;
  constexpr int numTilesPerBD = 2;
  constexpr int numTilesPerBatch = numBDsPerBatch * numTilesPerBD;

  uint16_t in_out_rows[2];
  uint32_t* io_row_ptr = (uint32_t*)(in_out_rows);

  AIE2::DmaIteration nt_iteration;
  AIE2::Location nt_location(0, 0);
  uint8_t nt_ncols = 1, ddr_type;

  for (int tile = 0; tile < num_tiles;) {
    // std::cout << "-- Tile: " << tile << "\n";
    int batch = tile / 4;
    tile = tile - (4 + tile - num_tiles) * ((num_tiles - tile) < 4);
    int StartBd = 7 * (batch % 2);

    AIE2::DmaTensor4D nt_out_tensor = {{{1, (uint32_t)(patch_width_out / 4)},
                                        {(uint32_t)(img_width_out / 4), 4},
                                        {1, 0},
                                        {}}};

    nt_iteration.m_current = 0;
    nt_iteration.m_step = 0; //(m_tile_window_size_out_total >> 2);
    nt_iteration.m_wrap = 1; //(numBDsPerBatch * numTilesPerBD);

    ddr_type = 1;

    // output BD
    // -------------------------------------------------------------------------
    uint32_t start_out_bd = StartBd + 2 * numBDsPerBatch;
    uint32_t repeat_out_bd = 1; // numBDsPerBatch * numTilesPerBD;
    uint32_t out_addr = (((patch_y_out + tile) * img_width_out) + patch_x_out);

    auto outBD = m_noc_dma->getBD(start_out_bd);
    outBD->updateAddr(out_addr, 0)
        ->updateLength(numTilesPerBatch * m_tile_window_size_out[0]);
    outBD->updateTensorDims(nt_out_tensor)->updateIterationDims(nt_iteration);
    outBD->write(nt_location, nt_ncols, ddr_type);

    // task queue
    auto outTaskQ = m_noc_dma->getS2MMQueue(0);
    outTaskQ->setStartBd(start_out_bd)->setRepeatCount(repeat_out_bd);
    outTaskQ->write(nt_location);

#ifdef USE_VISITOR
    // Push to vector
    dma_components.push_back(outBD->getUniqueId());
    dma_components.push_back(outTaskQ->getUniqueId());
#else
    m_instr_counter +=
        m_instr_writer->write(outBD, m_instr_buffer + m_instr_counter);
    m_instr_counter +=
        m_instr_writer->write(outTaskQ, m_instr_buffer + m_instr_counter);
#endif

    int offset_uv = tile / 2;
    int patch_uv_out = patch_y_out / 2;

    uint32_t out_addr_uv =
        (((patch_uv_out + (patch_y_out % 2) + offset_uv) * img_width_out) +
         patch_x_out);
    uint32_t out_addr_uv_dummy =
        (((patch_uv_out + offset_uv) * img_width_out) + patch_x_out);

    nt_iteration.m_current = 0;
    nt_iteration.m_step =
        (img_width_out >> 2); //(m_tile_window_size_out_total >> 2);
    nt_iteration.m_wrap = 2;  //(numBDsPerBatch * numTilesPerBD);
    AIE2::DmaTensor4D nt_out_tensor_yuv = {{{1, 0}, {1, 0}, {1, 0}, {1, 0}}};

    if ((patch_y_out % 2) == 0) {
      auto outBDuv1 = m_noc_dma->getBD(start_out_bd + 1);
      outBDuv1->updateAddr(out_addr_uv, 0)
          ->updateLength(m_tile_window_size_out[0]);
      outBDuv1->updateTensorDims(nt_out_tensor_yuv)
          ->updateIterationDims(nt_iteration);
      outBDuv1->enableNextBD()->updateNextBD(start_out_bd + 2);
      outBDuv1->write(nt_location, nt_ncols, 2);

#ifdef USE_VISITOR
      // Push to vector
      dma_components.push_back(outBDuv1->getUniqueId());
#else
      m_instr_counter +=
          m_instr_writer->write(outBDuv1, m_instr_buffer + m_instr_counter);
#endif
      auto outBDuv2 = m_noc_dma->getBD(start_out_bd + 2);
      outBDuv2->updateAddr(out_addr_uv_dummy, 0)
          ->updateLength(m_tile_window_size_out[0]);
      outBDuv2->updateTensorDims(nt_out_tensor_yuv)
          ->updateIterationDims(nt_iteration);
      outBDuv2->write(nt_location, nt_ncols, 3);

#ifdef USE_VISITOR
      // Push to vector
      dma_components.push_back(outBDuv2->getUniqueId());
#else
      m_instr_counter +=
          m_instr_writer->write(outBDuv2, m_instr_buffer + m_instr_counter);
#endif

    } else {
      auto outBDuv1 = m_noc_dma->getBD(start_out_bd + 1);
      outBDuv1->updateAddr(out_addr_uv_dummy, 0)
          ->updateLength(m_tile_window_size_out[0]);
      outBDuv1->updateTensorDims(nt_out_tensor_yuv)
          ->updateIterationDims(nt_iteration);
      outBDuv1->enableNextBD()->updateNextBD(start_out_bd + 2);
      outBDuv1->write(nt_location, nt_ncols, 3);

#ifdef USE_VISITOR
      // Push to vector
      dma_components.push_back(outBDuv1->getUniqueId());
#else
      m_instr_counter +=
          m_instr_writer->write(outBDuv1, m_instr_buffer + m_instr_counter);
#endif
      auto outBDuv2 = m_noc_dma->getBD(start_out_bd + 2);
      outBDuv2->updateAddr(out_addr_uv, 0)
          ->updateLength(m_tile_window_size_out[0]);
      outBDuv2->updateTensorDims(nt_out_tensor_yuv)
          ->updateIterationDims(nt_iteration);
      outBDuv2->write(nt_location, nt_ncols, 2);

#ifdef USE_VISITOR
      // Push to vector
      dma_components.push_back(outBDuv2->getUniqueId());
#else
      m_instr_counter +=
          m_instr_writer->write(outBDuv2, m_instr_buffer + m_instr_counter);
#endif
    }

    // task queue
    auto outTaskQuv = m_noc_dma->getS2MMQueue(0);
    outTaskQuv->setStartBd(start_out_bd + 1)->setRepeatCount(numTilesPerBD);
    outTaskQuv->enableTokenIssue()->write(nt_location);

#ifdef USE_VISITOR
    // Push to vector
    dma_components.push_back(outTaskQuv->getUniqueId());
#else
    m_instr_counter +=
        m_instr_writer->write(outTaskQuv, m_instr_buffer + m_instr_counter);
#endif

    for (int j = 0; j < numBDsPerBatch; tile++, j++) {
      // Metadata compute done here
      float idx_y = ((tile + j + 0.5f) * y_scale) - 0.5f;
      idx_y = PP_MIN(PP_MAX(idx_y, 0.0f), (float)(img_height_in - 2));
      int offset_y = (int)idx_y;
      // int offset_uv = (int)(offset_y / 2);

      float idx_y1 = ((tile + 1 + j + 0.5f) * y_scale) - 0.5f;
      idx_y1 = PP_MIN(PP_MAX(idx_y1, 0.0f), (float)(img_height_in - 2));
      int offset_y1 = (int)idx_y1;

      // printf("  Tile: %d, Rows: y(%d,%d)\n", tile + j, offset_y, offset_y +
      // 1); printf("  Tile: %d, Rows: y(%d,%d)\n", tile + j + 1, offset_y1,
      // offset_y1 + 1);

      in_out_rows[0] = offset_y;
      in_out_rows[1] = tile + j;

      AIE2::Location at_loc_1(0, 2 + 2 * (j % 2));
      auto word_1 =
          m_aie_dma->getWord((RTP_ADDR + 2 * (PP_EGCPOST_RTP_IN_ROW_0 +
                                              2 * ((j / 2) + (batch % 2)))),
                             io_row_ptr[0]);
      word_1->write(at_loc_1);

#ifdef USE_VISITOR
      dma_components.push_back(word_1->getUniqueId());
#else
      m_instr_counter +=
          m_instr_writer->write(word_1, m_instr_buffer + m_instr_counter);
#endif

      in_out_rows[0] = offset_y1;
      in_out_rows[1] = tile + 1 + j;

      AIE2::Location at_loc_2(0, 2 + 2 * (j % 2) + 1);
      auto word_2 =
          m_aie_dma->getWord((RTP_ADDR + 2 * (PP_EGCPOST_RTP_IN_ROW_0 +
                                              2 * ((j / 2) + (batch % 2)))),
                             io_row_ptr[0]);
      word_2->write(at_loc_2);

#ifdef USE_VISITOR
      dma_components.push_back(word_2->getUniqueId());
#else
      m_instr_counter +=
          m_instr_writer->write(word_2, m_instr_buffer + m_instr_counter);
#endif

      idx_y = ((tile + 0.5f) * y_scale) - 0.5f;
      idx_y = PP_MIN(PP_MAX(idx_y, 0.0f), (float)(img_height_in - 2));
      offset_y = (int)idx_y;

      idx_y1 = ((tile + numBDsPerBatch + 0.5f) * y_scale) - 0.5f;
      idx_y1 = PP_MIN(PP_MAX(idx_y1, 0.0f), (float)(img_height_in - 2));
      offset_y1 = (int)idx_y1;

      uint32_t addr_offset =
          (offset_y1 * img_width_in * 4) - (offset_y * img_width_in * 4);
      uint32_t addr_low, addr_high;
      uint8_t use_next_bd, bd_id, next_bd, valid_bd;

      // Input-0 (y)
      // -------------------------------------------------------------------------
      nt_iteration.m_current = 0;
      nt_iteration.m_step = addr_offset >> 2;
      nt_iteration.m_wrap = (addr_offset < 4) ? 1 : (numTilesPerBD);

      use_next_bd = (j < (numBDsPerBatch - 1)) ? 1 : 0;
      bd_id = StartBd + 0 + j;
      next_bd = (j < (numBDsPerBatch - 1)) ? (StartBd + 0 + j + 1) : 0;
      valid_bd = 1;
      ddr_type = 0;

      AIE2::DmaTensor4D nt_y0_tensor = {{{1, 0}, {1, 0}, {1, 0}, {}}};

      addr_low = (offset_y * img_width_in * 4);
      addr_high = 0;

      auto y0BD = m_noc_dma->getBD(bd_id);
      y0BD->updateAddr(addr_low, addr_high)
          ->updateLength(m_tile_window_size_in[0]);
      y0BD->updateTensorDims(nt_y0_tensor)->updateIterationDims(nt_iteration);
      if (use_next_bd)
        y0BD->enableNextBD()->updateNextBD(next_bd);
      y0BD->write(nt_location, nt_ncols, ddr_type);

#ifdef USE_VISITOR
      dma_components.push_back(y0BD->getUniqueId());
#else
      m_instr_counter +=
          m_instr_writer->write(y0BD, m_instr_buffer + m_instr_counter);
#endif

      use_next_bd = (j < (numBDsPerBatch - 1)) ? 1 : 0;
      bd_id = StartBd + numBDsPerBatch + j;
      next_bd =
          (j < (numBDsPerBatch - 1)) ? (StartBd + numBDsPerBatch + j + 1) : 0;
      valid_bd = 1;
      ddr_type = 0;

      addr_low = addr_low + (img_width_in * 4);
      addr_high = 0;

      auto y1BD = m_noc_dma->getBD(bd_id);
      y1BD->updateAddr(addr_low, addr_high)
          ->updateLength(m_tile_window_size_in[1]);
      y1BD->updateTensorDims(nt_y0_tensor)->updateIterationDims(nt_iteration);
      if (use_next_bd)
        y1BD->enableNextBD()->updateNextBD(next_bd);
      y1BD->write(nt_location, nt_ncols, ddr_type);

#ifdef USE_VISITOR
      dma_components.push_back(y1BD->getUniqueId());
#else
      m_instr_counter +=
          m_instr_writer->write(y1BD, m_instr_buffer + m_instr_counter);
#endif
    }

    // Start Q word
    auto in1TaskQ = m_noc_dma->getMM2SQueue(0);
    in1TaskQ->setStartBd(StartBd)
        ->setRepeatCount(numTilesPerBD)
        ->write(nt_location);

#ifdef USE_VISITOR
    dma_components.push_back(in1TaskQ->getUniqueId());
#else
    m_instr_counter +=
        m_instr_writer->write(in1TaskQ, m_instr_buffer + m_instr_counter);
#endif

    // Start Q word
    auto in2TaskQ = m_noc_dma->getMM2SQueue(1);
    in2TaskQ->setStartBd(StartBd + numBDsPerBatch)
        ->setRepeatCount(numTilesPerBD)
        ->write(nt_location);

#ifdef USE_VISITOR
    dma_components.push_back(in2TaskQ->getUniqueId());
#else
    m_instr_counter +=
        m_instr_writer->write(in2TaskQ, m_instr_buffer + m_instr_counter);
#endif

    // Update tile counter
    tile += numBDsPerBatch;
    if (batch >= 1) {
      // Sync
      auto sync_word = m_noc_dma->getSyncWord(AIE2::DmaDirection::S2MM, 0, 1,
                                              1); // (direction, ch, ncol, nrow)
      sync_word->write(nt_location);
#ifdef USE_VISITOR
      dma_components.push_back(sync_word->getUniqueId());
#else
      m_instr_counter +=
          m_instr_writer->write(sync_word, m_instr_buffer + m_instr_counter);
#endif
    }
  }

  auto sync_word = m_noc_dma->getSyncWord(AIE2::DmaDirection::S2MM, 0, 1,
                                          1); // (direction, ch, ncol, nrow)
  sync_word->write(nt_location);
#ifdef USE_VISITOR
  dma_components.push_back(sync_word->getUniqueId());
#else
  m_instr_counter +=
      m_instr_writer->write(sync_word, m_instr_buffer + m_instr_counter);
#endif

#ifdef USE_VISITOR
  // Visitors
  auto instrWriter = std::make_unique<IPUInstructionWriter>();
  auto irWriter = std::make_unique<InstructionIRWriter>();
  auto dbgWriter = std::make_unique<IPUDebugInstrWriter>();
  m_instr_counter = 0;
  for (auto uid : dma_components) {
    auto component = getDmaElement(uid);
    auto ncount =
        component->accept(instrWriter.get(), m_instr_buffer + m_instr_counter);
    auto ncount1 =
        component->accept(dbgWriter.get(), m_instr_buffer + m_instr_counter);
    auto rval1 =
        component->accept(irWriter.get(), m_instr_buffer + m_instr_counter);
    m_instr_counter += ncount;
  }

  // Write IR to console
  // dbgInstrWriter->writeToConsole();
  dbgWriter->writeToFile("output_dbg.txt");
  irWriter->writeToFile("output_ir.txt");
#endif
  // Return number of instructions generated
  return m_instr_counter;
}