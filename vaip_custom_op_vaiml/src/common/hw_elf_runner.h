/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */
#pragma once
#include <cstdlib>
#include <filesystem>
#include <fstream>

// AIE Driver headers
#include "xaiengine.h"

#include "experimental/xrt_elf.h"
#include "experimental/xrt_ext.h"
#include "experimental/xrt_kernel.h"
#include "experimental/xrt_module.h"
#include "runner_common_def.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"
#include <sstream>
#define HW_RUNNER_USE_CMDLIST
namespace vaiml_elf_runner {
using namespace vaip_vaiml_custom_op;

class xrt_context {
protected:
  xrt::device device_;
  xrt::xclbin xclbin_;
  xrt::hw_context context_;
  std::vector<xrt::kernel> kernel_vec_;
  std::map<std::string, std::uint32_t> qos = {{"is_preemptible", true}};
  xrt_context(const std::vector<char>& xclbin) {
    unsigned int device_index = 0;
    device_ = xrt::device(device_index);
    xclbin_ = xrt::xclbin(xclbin);

    device_.register_xclbin(xclbin_);
    context_ = xrt::hw_context(device_, xclbin_.get_uuid(), qos);
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
};

class hw_elf_runner : public hw_runner_base {
public:
  hw_elf_runner();
  hw_elf_runner(bool lazy_ifm_creation);

  // Constructor for hw_runner

  // Destructor
  ~hw_elf_runner(){};
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

  // Performs the run on hardware
  int run(void* ifm, void* wts, void* ofm);

  void get_bo_ptrs(int8_t*& ifm_ptr, int8_t*& wts_ptr, int8_t*& ofm_ptr);
  void pre_run_bo_sync();
  void load_elf(const std::vector<std::string>& elf_paths);
  void load_elf(std::vector<std::stringstream>& elf_stream);

  void create_ert_kernel(const std::vector<xrt::module>& mod_vec,
                         const std::vector<KERNEL_NM>& kernel_index) {
    for (int i = 0; i < mod_vec.size(); i++) {
      kernel_vec_.push_back(xrt::ext::kernel(context_->get_context(),
                                             mod_vec[i],
                                             kernel_name_map[kernel_index[i]]));
    }
  }
  xrt::kernel& get_kernel(uint32_t idx = 0) { return kernel_vec_[idx]; }
  void create_ifm_and_update_run_obj();

private:
  std::vector<XRTRunOffset> run_offsets_;
  std::vector<KERNEL_NM> kernel_index_;
  bool lazy_ifm_creation_ = false;
  std::map<KERNEL_NM, std::string> kernel_name_map = {
      {KERNEL_NM::GT_CONV, "GT"},
      {KERNEL_NM::GT_MM, "GT"},
      {KERNEL_NM::HT, "HT"}};
  std::vector<std::vector<xrt::bo>> sub_bo_;
  std::vector<xrt::elf> elf_vec_;
  std::vector<xrt::module> mod_vec_;
  std::vector<xrt::kernel> kernel_vec_;

  BO_ORDER bo_order_ = BO_ORDER::WTS_IFM_TMP_OFM;
  std::vector<BO_ORDER> bo_order_vec_ = {};
  std::vector<xrt::run> run_obj_vec_;
  xrt::bo* logical_ofm_bo_ = nullptr;
#ifdef HW_RUNNER_USE_CMDLIST
  std::vector<xrt::runlist> runlist_wrapped_;
#endif
  xrt_context* context_;
  uint32_t ifm_size_;
  uint32_t wts_size_;
  uint32_t ofm_size_;
  uint32_t tmp_size_;

  std::string ofm_filename_;

  int8_t* ifm_ptr_;
  int8_t* wts_ptr_;
  int8_t* ofm_ptr_;
  int8_t* tmp_ptr_;
  xrt::bo ifm_bo_;
  xrt::bo wts_bo_;
  xrt::bo ofm_bo_;
  xrt::bo tmp_bo_;
};

} // namespace vaiml_elf_runner
