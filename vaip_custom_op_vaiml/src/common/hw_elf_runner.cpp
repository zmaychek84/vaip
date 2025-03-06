/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */
#include "hw_elf_runner.h"
#include "experimental/xrt_ini.h"
#include "timer.h"
#include "utils.h"
#include <iostream>
#include <windows.h>
// #define __DEBUG_AIE_RUNNER_HOST__

// namespace vaip_vaiml_custom_op {
namespace vaiml_elf_runner {

hw_elf_runner::hw_elf_runner() : lazy_ifm_creation_(false) {}
hw_elf_runner::hw_elf_runner(bool lazy_ifm_creation)
    : lazy_ifm_creation_(lazy_ifm_creation) {}

void hw_elf_runner::load_xclbin(const std::vector<char>& xclbin) {
  context_ = &xrt_context::get_instance(xclbin);
}

void hw_elf_runner::hw_runner_init(uint32_t ifm_size, uint32_t wts_size,
                                   uint32_t ofm_size, uint32_t tmp_size,
                                   bool gt_mode_,
                                   const std::vector<XRTRunOffset>& run_offsets,
                                   const std::vector<KERNEL_NM>& kernel_index) {
  xrt::ini::set("verbosity", 5); // for halo compatible?
  ifm_size_ = ifm_size;
  wts_size_ = wts_size;
  ofm_size_ = ofm_size;
  tmp_size_ = tmp_size;
  VAIML_DEBUG_PRINT("sizes passed to hw runner", ifm_size, ",", wts_size, ",",
                    ofm_size, ",", tmp_size);
  if (!gt_mode_) { // legacy mode
    wts_bo_ = xrt::ext::bo{context_->get_device(), wts_size_};
    ofm_bo_ = xrt::ext::bo{context_->get_device(), ofm_size_};
    tmp_bo_ = xrt::ext::bo{context_->get_device(), tmp_size_};
    logical_ofm_bo_ = &ofm_bo_;
    if (!lazy_ifm_creation_) {
      ifm_bo_ = xrt::ext::bo{context_->get_device(), ifm_size_};
      ifm_ptr_ = ifm_bo_.map<int8_t*>();
    }
    wts_ptr_ = wts_bo_.map<int8_t*>();
    ofm_ptr_ = logical_ofm_bo_->map<int8_t*>();
  } else {
    wts_bo_ = xrt::ext::bo{context_->get_device(), wts_size_};
    tmp_bo_ = xrt::ext::bo{context_->get_device(), tmp_size_};
    logical_ofm_bo_ = &ifm_bo_;
    if (!lazy_ifm_creation_) {
      ifm_bo_ = xrt::ext::bo{context_->get_device(), ifm_size_};
      ifm_ptr_ = ifm_bo_.map<int8_t*>();
      ofm_ptr_ = logical_ofm_bo_->map<int8_t*>();
    }
    wts_ptr_ = wts_bo_.map<int8_t*>();
  }
  this->create_ert_kernel(mod_vec_, kernel_index);

  run_offsets_ = run_offsets;
  kernel_index_ = kernel_index;

  runlist_wrapped_.emplace_back(context_->get_context());
  xrt::runlist& cmdq = runlist_wrapped_[0];

  for (int sg_id = 0; sg_id < run_offsets_.size(); sg_id++) {
    // create xrt run obj in advance
    // uint32_t idx = kernel_index[sg_id];
    run_obj_vec_.emplace_back(this->get_kernel()); // always use kernel at idx-0
    cmdq.add(run_obj_vec_.back());
  }
  if (!lazy_ifm_creation_) {
    create_ifm_and_update_run_obj();
  }
}
void hw_elf_runner::create_ifm_and_update_run_obj() {
  if (lazy_ifm_creation_) {
    ifm_bo_ = xrt::ext::bo{context_->get_device(), ifm_size_};
    ifm_ptr_ = ifm_bo_.map<int8_t*>();
    ofm_ptr_ = logical_ofm_bo_->map<int8_t*>();
  }
  for (int sg_id = 0; sg_id < run_offsets_.size(); sg_id++) {
    auto& run = run_obj_vec_[sg_id];

    if (bo_order_vec_[sg_id] == BO_ORDER::ODR_GT_CONV) {
      sub_bo_.push_back(
          {xrt::bo(wts_bo_, 2501056, 0), xrt::bo(ifm_bo_, 16480, 0),
           xrt::bo(tmp_bo_, 2129920, 0), xrt::bo(ifm_bo_, 552960, 16480)});
      run.set_arg(0, 3);
      run.set_arg(1, 0);
      run.set_arg(2, 0);
      run.set_arg(3, sub_bo_.back()[0]);
      run.set_arg(4, sub_bo_.back()[1]);
      run.set_arg(5, sub_bo_.back()[2]);
      run.set_arg(6, 0);
      run.set_arg(7, sub_bo_.back()[3]);
    }
    if (bo_order_vec_[sg_id] == BO_ORDER::ODR_GT_HEAD) {
      sub_bo_.push_back({xrt::bo(wts_bo_, 5418432, 0),
                         xrt::bo(ifm_bo_, 552960, 16480),
                         xrt::bo(ifm_bo_, 25600, 16480 + 552960)});
      run.set_arg(0, 3);
      run.set_arg(1, 0);
      run.set_arg(2, 0);
      run.set_arg(3, sub_bo_.back()[0]);
      run.set_arg(4, sub_bo_.back()[1]);
      run.set_arg(5, sub_bo_.back()[2]);
      run.set_arg(6, 0);
      run.set_arg(7, 0);
    }
    if (bo_order_vec_[sg_id] == BO_ORDER::ODR_GT_TAIL) {
      sub_bo_.push_back({xrt::bo(wts_bo_, 302656, 0),
                         xrt::bo(ifm_bo_, 25600, 0), xrt::bo(tmp_bo_, 51200, 0),
                         xrt::bo(ifm_bo_, 25600, 0)});
      run.set_arg(0, 3);
      run.set_arg(1, 0);
      run.set_arg(2, 0);
      run.set_arg(3, sub_bo_.back()[0]);
      run.set_arg(4, sub_bo_.back()[1]);
      run.set_arg(5, sub_bo_.back()[2]);
      run.set_arg(6, 0);
      run.set_arg(7, sub_bo_.back()[3]);
    }
    if (bo_order_vec_[sg_id] == BO_ORDER::ODR_GT_TRANSFORMER) {
      sub_bo_.push_back(
          {xrt::bo(wts_bo_, 6032384, run_offsets_[sg_id].wts_offset),
           xrt::bo(ifm_bo_, 4300800, run_offsets_[sg_id].ifm_offset),
           xrt::bo(tmp_bo_, 1536000, 0),
           xrt::bo(ifm_bo_, 4300800, run_offsets_[sg_id].ofm_offset)});
      run.set_arg(0, 3);
      run.set_arg(1, 0);
      run.set_arg(2, 0);
      run.set_arg(3, sub_bo_.back()[0]);
      run.set_arg(4, sub_bo_.back()[1]);
      run.set_arg(5, sub_bo_.back()[2]);
      run.set_arg(6, 0);
      run.set_arg(7, sub_bo_.back()[3]);
    }
    if (bo_order_vec_[sg_id] == BO_ORDER::ODR_HT) {
      sub_bo_.push_back(
          {xrt::bo(wts_bo_, 17868672, 0), xrt::bo(*logical_ofm_bo_, 9344, 0),
           xrt::bo(tmp_bo_, 11904, 0), xrt::bo(ifm_bo_, 15104, 0)});
      run.set_arg(0, 3);
      run.set_arg(1, 0);
      run.set_arg(2, 0);
      run.set_arg(3, sub_bo_.back()[0]);
      run.set_arg(4, sub_bo_.back()[1]);
      run.set_arg(5, sub_bo_.back()[2]);
      run.set_arg(6, 0);
      run.set_arg(7, sub_bo_.back()[3]);
    }
  }
}

void hw_elf_runner::set_bo_order(BO_ORDER order) { bo_order_ = order; }

void hw_elf_runner::set_bo_order_vec(std::vector<BO_ORDER> order) {
  bo_order_vec_ = order;
}

void hw_elf_runner::get_bo_ptrs(int8_t*& ifm_ptr, int8_t*& wts_ptr,
                                int8_t*& ofm_ptr) {
  ifm_ptr = ifm_ptr_;
  wts_ptr = wts_ptr_;
  ofm_ptr = ofm_ptr_;
}
void hw_elf_runner::pre_run_bo_sync() {
  tmp_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  wts_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // init bo space, require sync agin in run body
  // ifm_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  // ofm_bo_.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
}

void hw_elf_runner::load_elf(const std::vector<std::string>& elf_paths) {
  for (auto& p : elf_paths) {
    elf_vec_.emplace_back(p);
    mod_vec_.emplace_back(elf_vec_.back());
  }
}
void hw_elf_runner::load_elf(std::vector<std::stringstream>& elf_stream) {
  for (auto& stream : elf_stream) {
    elf_vec_.emplace_back(stream);
    mod_vec_.emplace_back(elf_vec_.back());
  }
}

// Performs the run on hardware
int hw_elf_runner::run(void* ifm, void* wts, void* ofm) {
  {
    TIMER(SYNC_IFM, "sync ifm bo ")
    ifm_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  }

  {
    TIMER(XRT_RUN, "xrt run ")

    runlist_wrapped_[0].execute();
    runlist_wrapped_[0].wait();
  }

  // Sleep(1);

  // auto run_state = run_obj_.state();

  // if (run_state != 4) {
  //   return -2;
  // }

  {
    TIMER(SYNC_OFM, "ofm bo sync ")
    logical_ofm_bo_->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  }

  return 0;
}

} // namespace vaiml_elf_runner
