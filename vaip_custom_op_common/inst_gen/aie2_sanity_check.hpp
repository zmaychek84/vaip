/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */
/**
 * @file  aie2_sanity_check.hpp
 * @brief File containing Class methods to check sanity checks for bit-fields of
 * AIE2 registers.
 */
#pragma once

#include <iostream>

namespace AIE2 {

namespace Sanity {

/**
 * @brief   Returns maximum unsigned value represented by number of bits.
 * @returns unsigned max value
 */
template <int BITS> constexpr uint32_t numeric_max_u(void) {
  return static_cast<uint32_t>(((static_cast<uint64_t>(1) << BITS) - 1) &
                               0xFFFFFFFF);
}

/**
 * @brief   Returns minimum unsigned value represented by number of bits.
 * @returns unsigned min value (0 always)
 */
template <int BITS> constexpr uint32_t numeric_min_u(void) { return 0; }

/**
 * @brief   Returns maximum signed value represented by number of bits.
 * @returns signed max value
 */
template <int BITS> constexpr int32_t numeric_max_s(void) {
  return static_cast<int32_t>((1 << (BITS - 1)) - 1);
}

/**
 * @brief   Returns minimum signed value represented by number of bits.
 * @returns signed min value
 */
template <int BITS> constexpr int32_t numeric_min_s(void) {
  return static_cast<int32_t>(-(1 << (BITS - 1)));
}

/**
 * @class AieTile
 * @brief Class containing methods to check sanity of AIE-tile registers.
 */
class AieTile {
public:
  /**
   * @class BD
   * @brief Sanity checker for AIE-tile BD registers.
   */
  class BD {
  public:
    template <int BITS = 14> // Length: 14b
    static bool checkLength(uint32_t len) {
      return ((len >> 2) > numeric_max_u<BITS>()) ? false : true;
    }
    template <int BITS = 14> // Addr: 14b
    static bool checkBaseAddr(uint32_t addr) {
      return ((addr >> 2) > numeric_max_u<BITS>()) ? false : true;
    }
    template <int BITS = 3> // PacketType: 3b
    static bool checkPacketType(uint32_t ptype) {
      return (ptype > numeric_max_u<BITS>()) ? false : true;
    }
    template <int BITS = 5> // PacketId: 5b
    static bool checkPacketId(uint32_t id) {
      return (id > numeric_max_u<BITS>()) ? false : true;
    }
    template <int BITS = 6> // OutOfOrderBdId: 6b
    static bool checkOutOfOrderBdId(uint32_t id) {
      return (id > numeric_max_u<BITS>()) ? false : true;
    }
    template <int BITS = 13> // D0/D1/D2 step: 13b
    static bool checkStep(uint32_t step) {
      return ((step - 1) > numeric_max_u<BITS>()) ? false : true;
    }
    template <int BITS = 8> // D0/D1 wrap: 8b
    static bool checkWrap(uint32_t wrap) {
      return (wrap > numeric_max_u<BITS>()) ? false : true;
    }
    template <int BITS = 13> // Iter step: 13b
    static bool checkIterStep(uint32_t step) {
      return ((step - 1) > numeric_max_u<BITS>()) ? false : true;
    }
    template <int BITS = 6> // Iter wrap: 6b
    static bool checkIterWrap(uint32_t wrap) {
      return ((wrap - 1) > numeric_max_u<BITS>()) ? false : true;
    }
    template <int BITS = 7> // Lock Value(signed): 7b
    static bool checkLockVal(int32_t val) {
      return ((val > numeric_max_s<BITS>()) || (val < numeric_min_s<BITS>()))
                 ? false
                 : true;
    }
    template <int BITS = 4> // Lock ID: 4b
    static bool checkLockId(uint32_t id) {
      return (id > numeric_max_u<BITS>()) ? false : true;
    }
    template <int BITS = 4> // NextBD: 4b
    static bool checkNextBd(uint32_t id) {
      return (id > numeric_max_u<BITS>()) ? false : true;
    }
  };

  /**
   * @class Queue
   * @brief Sanity checker for AIE-tile Channel Start Queue registers.
   */
  class Queue {
  public:
    template <int BITS = 4> // StartBD: 4b
    static bool checkStartBd(uint32_t bd) {
      return (bd > numeric_max_u<BITS>()) ? false : true;
    }
    template <int BITS = 8> // RepeatCount: 8b
    static bool checkRepeatCount(uint32_t count) {
      return ((count - 1) > numeric_max_u<BITS>()) ? false : true;
    }
  };
};

/**
 * @class MemTile
 * @brief Class containing methods to check sanity of Mem-tile registers.
 */
class MemTile {
public:
  /**
   * @class BD
   * @brief Sanity checker for Mem-tile BD registers.
   */
  class BD {
  public:
    template <int BITS = 17> // Length: 17b
    static bool checkLength(uint32_t len) {
      return ((len >> 2) > numeric_max_u<BITS>()) ? false : true;
    }
    template <int BITS = 19> // Addr: 19b
    static bool checkBaseAddr(uint32_t addr) {
      return ((addr >> 2) > numeric_max_u<BITS>()) ? false : true;
    }
    template <int BITS = 3> // PacketType: 3b
    static bool checkPacketType(uint32_t ptype) {
      return (ptype > numeric_max_u<BITS>()) ? false : true;
    }
    template <int BITS = 5> // PacketId: 5b
    static bool checkPacketId(uint32_t id) {
      return (id > numeric_max_u<BITS>()) ? false : true;
    }
    template <int BITS = 6> // OutOfOrderBdId: 6b
    static bool checkOutOfOrderBdId(uint32_t id) {
      return (id > numeric_max_u<BITS>()) ? false : true;
    }
    template <int BITS = 17> // D0/D1/D2/D3 step: 17b
    static bool checkStep(uint32_t step) {
      return ((step - 1) > numeric_max_u<BITS>()) ? false : true;
    }
    template <int BITS = 10> // D0/D1 wrap: 10b
    static bool checkWrap(uint32_t wrap) {
      return (wrap > numeric_max_u<BITS>()) ? false : true;
    }
    template <int BITS = 17> // Iter step: 17b
    static bool checkIterStep(uint32_t step) {
      return ((step - 1) > numeric_max_u<BITS>()) ? false : true;
    }
    template <int BITS = 6> // Iter wrap: 6b
    static bool checkIterWrap(uint32_t wrap) {
      return ((wrap - 1) > numeric_max_u<BITS>()) ? false : true;
    }
    template <int BITS = 7> // Lock Value(signed): 7b
    static bool checkLockVal(int32_t val) {
      // Remove the last check if
      return ((val > numeric_max_s<BITS>()) || (val < numeric_min_s<BITS>()))
                 ? false
                 : true;
    }
    template <int BITS = 8> // Lock ID: 8b
    static bool checkLockId(uint32_t id) {
      return (id > numeric_max_u<BITS>()) ? false : true;
    }
    template <int BITS = 6> // Next BD: 6b
    static bool checkNextBd(uint32_t id) {
      return (id > numeric_max_u<BITS>()) ? false : true;
    }
  };

  /**
   * @class Queue
   * @brief Sanity checker for Mem-tile Channel Start Queue registers.
   */
  class Queue {
  public:
    template <int BITS = 7> static bool checkStartBd(uint32_t bd) {
      return (bd > numeric_max_u<6>()) ? false : true; // StartBD: 6b
    }
    template <int BITS = 7> static bool checkRepeatCount(uint32_t count) {
      return ((count - 1) > numeric_max_u<8>()) ? false
                                                : true; // RepeatCount: 8b
    }
  };
};

/**
 * @class NoCTile
 * @brief Class containing methods to check sanity of NoC-tile registers.
 */
class NoCTile {
public:
  /**
   * @class BD
   * @brief Sanity checker for NoC-tile BD registers.
   */
  class BD {
  public:
    template <int BITS = 32> // Length: 32b
    static bool checkLength(uint32_t len) {
      return ((len >> 2) > numeric_max_u<BITS>()) ? false : true;
    }
    template <int BITS = 30> // AddrLow: 30b
    static bool checkBaseAddrLow(uint32_t addr) {
      return ((addr >> 2) > numeric_max_u<BITS>()) ? false : true;
    }
    template <int BITS = 16> // Addrhigh: 16b
    static bool checkBaseAddrHigh(uint32_t addr) {
      return ((addr >> 2) > numeric_max_u<BITS>()) ? false : true;
    }
    template <int BITS = 3> // PacketType: 3b
    static bool checkPacketType(uint32_t ptype) {
      return (ptype > numeric_max_u<BITS>()) ? false : true;
    }
    template <int BITS = 5> // PacketId: 5b
    static bool checkPacketId(uint32_t id) {
      return (id > numeric_max_u<BITS>()) ? false : true;
    }
    template <int BITS = 6> // OutOfOrderBdId: 6b
    static bool checkOutOfOrderBdId(uint32_t id) {
      return (id > numeric_max_u<BITS>()) ? false : true;
    }
    template <int BITS = 20> // D0/D1/D2 step: 20b
    static bool checkStep(uint32_t step) {
      return ((step - 1) > numeric_max_u<BITS>()) ? false : true;
    }
    template <int BITS = 10> // D0/D1 wrap: 10b
    static bool checkWrap(uint32_t wrap) {
      return (wrap > numeric_max_u<BITS>()) ? false : true;
    }
    template <int BITS = 20> // Iter step: 20b
    static bool checkIterStep(uint32_t step) {
      return ((step - 1) > numeric_max_u<BITS>()) ? false : true;
    }
    template <int BITS = 6> // Iter wrap: 6b
    static bool checkIterWrap(uint32_t wrap) {
      return ((wrap - 1) > numeric_max_u<BITS>()) ? false : true;
    }
    template <int BITS = 7> // Lock Value(signed): 7b
    static bool checkLockVal(int32_t val) {
      return ((val > numeric_max_s<BITS>()) || (val < numeric_min_s<BITS>()))
                 ? false
                 : true;
    }
    template <int BITS = 4> // Lock ID: 4b
    static bool checkLockId(uint32_t id) {
      return (id > numeric_max_u<BITS>()) ? false : true;
    }
    template <int BITS = 4> // Next BD: 4b
    static bool checkNextBd(uint32_t id) {
      return (id > numeric_max_u<BITS>()) ? false : true;
    }
    template <int BITS = 4> // AxQos: 4b
    static bool checkAxQos(uint32_t axqos) {
      return (axqos > numeric_max_u<BITS>()) ? false : true;
    }
    template <int BITS = 4> // AxCache: 4b
    static bool checkAxCache(uint32_t axcache) {
      return (axcache > numeric_max_u<BITS>()) ? false : true;
    }
    template <int BITS = 4> // SMID: 4b
    static bool checkSmid(uint32_t id) {
      return (id > numeric_max_u<BITS>()) ? false : true;
    }
  };

  /**
   * @class Queue
   * @brief Sanity checker for NoC-tile Channel Start Queue registers.
   */
  class Queue {
  public:
    template <int BITS = 4> // StartBD: 4b
    static bool checkStartBd(uint32_t bd) {
      return (bd > numeric_max_u<BITS>()) ? false : true;
    }
    template <int BITS = 8> // RepeatCount: 8b
    static bool checkRepeatCount(uint32_t count) {
      return ((count - 1) > numeric_max_u<BITS>()) ? false : true;
    }
  };
};

class Lock {
public:
  template <int BITS = 6> // Value: 6b
  static bool checkValue(uint32_t val) {
    return (val > numeric_max_u<BITS>()) ? false : true;
  }
};

} // namespace Sanity

} // namespace AIE2
