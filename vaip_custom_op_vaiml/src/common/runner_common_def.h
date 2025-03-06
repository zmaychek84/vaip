/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

#pragma once
#include <cstdlib>
#include <filesystem>
#include <fstream>

#include <sstream>
namespace vaip_vaiml_custom_op {

enum KERNEL_NM { GT_CONV = 0, GT_MM = 1, HT = 2 };
enum BO_ORDER {
  WTS_IFM_OFM,
  WTS_IFM_TMP_OFM,
  WTS_OFM_TMP_IFM,
  ODR_GT_CONV,
  ODR_GT_HEAD,
  ODR_GT_TRANSFORMER,
  ODR_GT_TAIL,
  ODR_HT
};

struct XRTRunOffset {
  size_t ifm_offset = 0;
  size_t wts_offset = 0;
  size_t ofm_offset = 0;
  size_t tmp_offset = 0;
  XRTRunOffset(size_t i_off, size_t w_off, size_t o_off, size_t t_off)
      : ifm_offset(i_off), wts_offset(w_off), ofm_offset(o_off),
        tmp_offset(t_off) {}
  XRTRunOffset() {}
};

class hw_runner_base {
public:
  hw_runner_base(){};
  // Constructor for hw_runner

  // Destructor
  ~hw_runner_base(){};
  virtual void set_bo_order(BO_ORDER order) = 0;
  virtual void set_bo_order_vec(std::vector<BO_ORDER> order) = 0;
  virtual void hw_runner_init(uint32_t ifm_size, uint32_t wts_size,
                              uint32_t ofm_size, uint32_t tmp_size,
                              bool gt_mode,
                              const std::vector<XRTRunOffset>& run_offets,
                              const std::vector<KERNEL_NM>& kernel_index) = 0;

  virtual void load_xclbin(const std::vector<char>& xclbin) = 0;
  virtual void load_txn_bin(const std::string& txnbin_filename) {}
  virtual void load_txn_bin(const std::vector<std::string>& txnbin_filenames) {}
  virtual void
  load_ctrl_pkt_bin(const std::vector<std::string>& ctrlpkt_filenames) {}
  virtual void load_elf(const std::vector<std::string>& elf_paths) {}
  virtual void load_elf(std::vector<std::stringstream>& elf_stream) {}
  // Performs the run on hardware
  virtual int run(void* ifm, void* wts, void* ofm) = 0;

  virtual void get_bo_ptrs(int8_t*& ifm_ptr, int8_t*& wts_ptr,
                           int8_t*& ofm_ptr) = 0;
  virtual void pre_run_bo_sync() = 0;
  virtual void create_ifm_and_update_run_obj() = 0;
};
} // namespace vaip_vaiml_custom_op
