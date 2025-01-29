/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */
/**
 * @file  aie2_types.hpp
 * @brief File containing basic data structures for abstacting
 *        different types/data such as Tile Indices, data shape,
 *        lock settings, etc.
 */
#pragma once
#ifdef __GNUC__
#  pragma GCC diagnostic ignored "-Wpedantic"
#endif
#include <array>
#include <cstdint>

namespace AIE2 {

/**
 * @struct  Location
 * @brief   Represents location of a Tile (AIE/MEM/NoC) as (Column, Row)
 *          in AI Engine array.
 */
struct Location {
  /**
   * Constructor.
   * @param col Column index (default: 0)
   * @param row Row index (default: 0)
   */
  Location(std::uint32_t col = 0, std::uint32_t row = 0)
      : m_col(col), m_row(row) {}
  std::uint32_t m_col;
  std::uint32_t m_row;
};

/**
 * @struct  LockConfig
 * @brief   Represents Lock settings for a Buffer Descriptor.
 */
struct LockConfig {
  /**
   * Default Constructor
   */
  LockConfig()
      : m_acq_id(0), m_acq_val(0), m_acq_en(0), m_rel_id(0), m_rel_val(0) {}
  /**
   * Constructor with params
   *
   * @param acq_id  Acquire lock ID
   * @param acq_val Acquire value for the lock
   * @param acq_en  Enable acquire
   * @param rel_id  Release lock ID
   * @param rel_val Release value for the lock
   */
  LockConfig(std::uint8_t acq_id, std::int8_t acq_val, std::uint8_t acq_en,
             std::uint8_t rel_id, std::int8_t rel_val)
      : m_acq_id(acq_id), m_acq_val(acq_val), m_acq_en(acq_en),
        m_rel_id(rel_id), m_rel_val(rel_val) {}
  std::uint8_t m_acq_id;
  std::int8_t m_acq_val;
  std::uint8_t m_acq_en;
  std::uint8_t m_rel_id;
  std::int8_t m_rel_val;
};

/**
 * @struct  DmaTensor
 * @brief   Represents DMA Tensor for data shape being transferred.
 */
struct DmaTensor {
  /**
   * Constructor.
   *
   * @param step  Offset for each step
   * @param wrap  Wrap after these many bytes
   */
  DmaTensor(std::uint32_t step = 1, std::uint32_t wrap = 0)
      : m_step(step), m_wrap(wrap) {}
  std::uint32_t m_step;
  std::uint32_t m_wrap;
};

/**
 * @var   DmaTensor4D
 * @brief Represents a 4-dimentional DMA Tensor
 */
using DmaTensor4D = std::array<DmaTensor, 4>;

/**
 * @struct  DmaIteration
 * @brief   Represents iteration step size and wrap each execution of a BD.
 */
struct DmaIteration {
  /**
   * Constructor.
   *
   * @param step    Offset for each step
   * @param wrap    Wrap after these many bytes
   * @param current Current iteration step
   */
  DmaIteration(std::uint32_t step = 1, std::uint32_t wrap = 1,
               std::uint32_t current = 0)
      : m_step(step), m_wrap(wrap), m_current(current) {}
  std::uint32_t m_step; /*@var DmaIteration::m_step (actual-1) */
  std::uint32_t m_wrap; /*@var DmaIteration::m_wrap (actual-1) */
  std::uint32_t m_current;
};

/**
 * @enum  DmaDirection
 * @brief Represents data movement direction.
 */
enum class DmaDirection : uint32_t { S2MM = 0, MM2S = 1 };

/**
 * @enum  DmaChannelCtrlFlags
 * @brief Represents state of time channel control.
 */
enum class DmaChannelCtrlFlags : uint32_t { UNRESET = 0, RESET = 1 };

/**
 * @struct  TileMetaData
 * @brief   Represents tile meta data.
 */
struct TileMetaData {
  struct {
    uint32_t height;
    uint32_t width;
    uint32_t channel;
  };
};

} // namespace AIE2

#ifndef PP_MAX
#  define PP_MAX(x, y) (((x) > (y)) ? (x) : (y))
#endif

#ifndef PP_MIN
#  define PP_MIN(x, y) (((x) < (y)) ? (x) : (y))
#endif
