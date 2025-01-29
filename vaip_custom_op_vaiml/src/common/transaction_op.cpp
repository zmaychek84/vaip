/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

#include "transaction_op.h"
#include <xaiengine.h>

namespace vaip_vaiml_custom_op {

transaction_op::transaction_op(const std::vector<uint8_t>& txn) {
  txn_op_.resize(txn.size() + TXN_OP_SIZE);

  XAie_TxnHeader* hdr = (XAie_TxnHeader*)txn.data();

  uint32_t* ptr = (uint32_t*)txn_op_.data();
  // set op code
  *ptr = TXN_OP_CODE;
  ptr++;
  *ptr = hdr->TxnSize + TXN_OP_SIZE;

  memcpy(txn_op_.data() + TXN_OP_SIZE, txn.data(), txn.size());
}

transaction_op::transaction_op(const std::string& txn) {
  txn_op_.resize(txn.size() + TXN_OP_SIZE);

  XAie_TxnHeader* hdr = (XAie_TxnHeader*)txn.data();

  uint32_t* ptr = (uint32_t*)txn_op_.data();
  // set op code
  *ptr = TXN_OP_CODE;
  ptr++;
  *ptr = hdr->TxnSize + TXN_OP_SIZE;

  memcpy(txn_op_.data() + TXN_OP_SIZE, txn.data(), txn.size());
}

size_t transaction_op::get_txn_instr_size() {
  uint32_t* ptr = (uint32_t*)txn_op_.data();
  return *(++ptr);
}

std::vector<uint8_t> transaction_op::get_txn_op() { return txn_op_; }

size_t transaction_op::getInstrBufSize(const std::string& txn_str) {
  return TXN_OP_SIZE + txn_str.size();
}
void transaction_op::addTxnOp(const std::string& txn_str, void* instr_buf) {

  XAie_TxnHeader* hdr = (XAie_TxnHeader*)txn_str.data();

  uint32_t* ptr = (uint32_t*)instr_buf;
  // set op code
  *ptr = TXN_OP_CODE;
  ptr++;
  *ptr = hdr->TxnSize + TXN_OP_SIZE;

  uint8_t* instr_ptr = (uint8_t*)instr_buf;

  memcpy(instr_ptr + TXN_OP_SIZE, txn_str.data(), txn_str.size());
}

} // namespace vaip_vaiml_custom_op
