/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */
#pragma once
#include <string>
#include <vector>

namespace vaip_vaiml_custom_op {

// Taken from dod/src/txn/txn_utils.hpp
class transaction_op {
public:
  /**
   * @brief create txn op created by aie_controller locally.
   * Format :
   *     | TRANSACTION_OP | SIZE | txn |
   */
  transaction_op(const std::vector<uint8_t>& txn);
  transaction_op(const std::string& txn);

  std::vector<uint8_t> get_txn_op();
  size_t get_txn_instr_size();

  static size_t getInstrBufSize(const std::string& txn_str);
  static void addTxnOp(const std::string& txn_str, void* instr_buf);

  // size of txn op header in bytes
  // this is the wrapper header around txn format supported by aie-rt
  constexpr static size_t TXN_OP_SIZE = 8;
  constexpr static uint32_t TXN_OP_CODE = 0;

private:
  std::vector<uint8_t> txn_op_;
};
} // namespace vaip_vaiml_custom_op
