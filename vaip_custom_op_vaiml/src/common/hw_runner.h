/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */
#ifndef HW_RUNNER_H
#  define HW_RUNNER_H
#  pragma once
#  include <cstdlib>
#  include <filesystem>
#  include <fstream>

// AIE Driver headers
#  include "xaiengine.h"

#  include "experimental/xrt_kernel.h"
#  include "runner_common_def.h"
#  include "xrt/xrt_bo.h"
#  include "xrt/xrt_device.h"
#  include "xrt/xrt_kernel.h"
using namespace vaip_vaiml_custom_op;

#  define HW_RUNNER_USE_CMDLIST
constexpr std::uint64_t DDR_AIE_ADDR_OFFSET = std::uint64_t{0x80000000};
constexpr std::uint64_t OPCODE = std::uint64_t{2};
// namespace vaip_vaiml_custom_op {

class xrt_context {
protected:
  xrt::device device_;
  xrt::xclbin xclbin_;
  xrt::hw_context context_;
  // xrt::kernel kernel_;
  // xrt::kernel kernel0_;
  std::vector<xrt::kernel> kernel_vec_;

protected:
  xrt_context(const std::vector<char>& xclbin) {
    unsigned int device_index = 0;
    device_ = xrt::device(device_index);
    xclbin_ = xrt::xclbin(xclbin);

    device_.register_xclbin(xclbin_);
    context_ = xrt::hw_context(device_, xclbin_.get_uuid());
    kernel_vec_.emplace_back(xrt::kernel(context_, "GT_CONV"));
    kernel_vec_.emplace_back(xrt::kernel(context_, "GT_MM"));
    kernel_vec_.emplace_back(xrt::kernel(context_, "HT"));
  }

public:
  static xrt_context& get_instance(const std::vector<char>& xclbin) {
    static xrt_context ctx_(xclbin);
    return ctx_;
  }

  xrt_context(const xrt_context&) = delete;
  xrt_context(const xrt_context&&) = delete;
  xrt_context& operator=(const xrt_context&) = delete;
  xrt_context& operator=(const xrt_context&&) = delete;

  xrt::device& get_device() { return device_; }
  xrt::hw_context& get_context() { return context_; }
  xrt::kernel& get_kernel(uint32_t idx = 0) { return kernel_vec_[idx]; }
};

class hw_runner : public hw_runner_base {
public:
  hw_runner();
  hw_runner(bool lazy_ifm_creation);
  // Constructor for hw_runner

  // Destructor
  ~hw_runner(){};
  void set_bo_order(BO_ORDER order);
  void set_bo_order_vec(std::vector<BO_ORDER> order);
  //  void hw_runner_init(const std::string& xclbin_filename, uint32_t ifm_size,
  //                      uint32_t wts_size, uint32_t ofm_size, uint32_t
  //                      tmp_size, bool gt_mode);
  void hw_runner_init(uint32_t ifm_size, uint32_t wts_size, uint32_t ofm_size,
                      uint32_t tmp_size, bool gt_mode,
                      const std::vector<XRTRunOffset>& run_offets,
                      const std::vector<KERNEL_NM>& kernel_index);

  void load_xclbin(const std::vector<char>& xclbin);
  void load_txn_bin(const std::string& txnbin_filename);
  void load_txn_bin(const std::vector<std::string>& txnbin_filenames);

  void load_ctrl_pkt_bin(const std::vector<std::string>& ctrlpkt_filenames);

  // Performs the run on hardware
  int run(void* ifm, void* wts, void* ofm);

  void get_bo_ptrs(int8_t*& ifm_ptr, int8_t*& wts_ptr, int8_t*& ofm_ptr);
  void pre_run_bo_sync();
  void create_ifm_and_update_run_obj();

private:
  std::vector<XRTRunOffset> run_offsets_;
  std::vector<KERNEL_NM> kernel_index_;
  bool lazy_ifm_creation_ = false;
  BO_ORDER bo_order_ = BO_ORDER::WTS_IFM_TMP_OFM;
  std::vector<BO_ORDER> bo_order_vec_ = {};
  std::vector<xrt::run> run_obj_vec_;
  xrt::bo* logical_ofm_bo_ = nullptr;
#  ifdef HW_RUNNER_USE_CMDLIST
  std::vector<xrt::runlist> runlist_wrapped_;
#  endif
  xrt_context* context_;
  uint32_t ifm_size_;
  uint32_t wts_size_;
  uint32_t ofm_size_;
  uint32_t tmp_size_;

  std::string ofm_filename_;

  std::vector<xrt::bo> instr_bo_vec_;
  std::vector<xrt::bo> ctrl_pkt_bo_vec_;

  int8_t* ifm_ptr_ = nullptr;
  int8_t* wts_ptr_ = nullptr;
  int8_t* ofm_ptr_ = nullptr;
  int8_t* tmp_ptr_ = nullptr;
  xrt::bo ifm_bo_;
  xrt::bo wts_bo_;
  xrt::bo ofm_bo_;
  xrt::bo tmp_bo_;
};

#endif // HW_RUNNER_H

       // } // namespace