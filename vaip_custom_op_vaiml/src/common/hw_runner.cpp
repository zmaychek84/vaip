/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */
#include "hw_runner.h"
#include "experimental/xrt_ini.h"
#include "timer.h"
#include "transaction_op.h"
#include "utils.h"
#include <iostream>
#include <windows.h>
// #define __DEBUG_AIE_RUNNER_HOST__

// namespace vaip_vaiml_custom_op {

namespace hw_runner_helper {
std::uint64_t to_aie_addr(std::uint64_t addr) {
  return addr + DDR_AIE_ADDR_OFFSET;
}
void set_fm(xrt::bo& ifm_bo_, xrt::bo& wts_bo_, xrt::bo& tmp_bo_,
            xrt::bo& ofm_bo_, BO_ORDER bo_order_,
            const XRTRunOffset& run_offset, std::uint64_t& fm_0,
            std::uint64_t& fm_1, std::uint64_t& fm_2, std::uint64_t& fm_3,
            std::uint64_t& fm_4, xrt::bo* ctrl_pkt_bo) {
  switch (bo_order_) {
  case BO_ORDER::ODR_GT_HEAD:
    fm_0 = to_aie_addr(wts_bo_.address()) + run_offset.wts_offset;
    fm_1 = to_aie_addr(ifm_bo_.address()) + run_offset.ifm_offset;
    fm_2 = to_aie_addr(ofm_bo_.address()) + run_offset.ofm_offset;
    fm_4 = 0;
    fm_3 = to_aie_addr(ctrl_pkt_bo->address());
    break;
  case BO_ORDER::ODR_GT_CONV:
    fm_0 = to_aie_addr(wts_bo_.address()) + run_offset.wts_offset;
    fm_1 = to_aie_addr(ifm_bo_.address()) + run_offset.ifm_offset;
    fm_2 = to_aie_addr(tmp_bo_.address()) + run_offset.tmp_offset;
    fm_3 = to_aie_addr(ofm_bo_.address()) + run_offset.ofm_offset;
    fm_4 = 0;
    break;
  case BO_ORDER::ODR_GT_TRANSFORMER:
    fm_0 = to_aie_addr(wts_bo_.address()) + run_offset.wts_offset;
    fm_1 = to_aie_addr(ifm_bo_.address()) + run_offset.ifm_offset;
    fm_2 = to_aie_addr(tmp_bo_.address()) + run_offset.tmp_offset;
    fm_4 = to_aie_addr(ofm_bo_.address()) + run_offset.ofm_offset;
    fm_3 = to_aie_addr(ctrl_pkt_bo->address());
    break;
  case BO_ORDER::ODR_GT_TAIL:
    fm_0 = to_aie_addr(wts_bo_.address());
    fm_1 = to_aie_addr(ifm_bo_.address());
    fm_2 = to_aie_addr(tmp_bo_.address());
    fm_4 = to_aie_addr(ofm_bo_.address());
    fm_3 = to_aie_addr(ctrl_pkt_bo->address());
    break;
  case BO_ORDER::ODR_HT:
    fm_0 = to_aie_addr(wts_bo_.address());
    fm_1 = to_aie_addr(ofm_bo_.address());
    fm_2 = to_aie_addr(tmp_bo_.address());
    fm_3 = to_aie_addr(ifm_bo_.address());
    fm_4 = 0;
    break;
  default:
    printf("unrecognized bo_order_: %d\n", bo_order_);
    break;
  }
}
}; // namespace hw_runner_helper

// Constructor for hw_runner
void hw_runner::load_xclbin(const std::vector<char>& xclbin) {
  context_ = &xrt_context::get_instance(xclbin);
}

// void hw_runner::hw_runner_init(const std::string& xclbin_filename,
//                                uint32_t ifm_size, uint32_t wts_size,
//                                uint32_t ofm_size, uint32_t tmp_size,
//                                bool gt_mode_) {
//   this->load_xclbin(xclbin_filename);
//   this->hw_runner_init(ifm_size, wts_size, ofm_size, tmp_size, gt_mode_, {},
//   {});
// }

void hw_runner::hw_runner_init(uint32_t ifm_size, uint32_t wts_size,
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
    ifm_bo_ = xrt::bo(context_->get_device(), ifm_size_, XRT_BO_FLAGS_HOST_ONLY,
                      context_->get_kernel().group_id(0));
    wts_bo_ = xrt::bo(context_->get_device(), wts_size_, XRT_BO_FLAGS_HOST_ONLY,
                      context_->get_kernel().group_id(0));
    ofm_bo_ = xrt::bo(context_->get_device(), ofm_size_, XRT_BO_FLAGS_HOST_ONLY,
                      context_->get_kernel().group_id(0));
    tmp_bo_ = xrt::bo(context_->get_device(), tmp_size_, XRT_BO_FLAGS_HOST_ONLY,
                      context_->get_kernel().group_id(0));
    logical_ofm_bo_ = &ofm_bo_;
  } else {
    ifm_bo_ =
        xrt::bo(context_->get_device(), ifm_size_ + ofm_size_,
                XRT_BO_FLAGS_HOST_ONLY, context_->get_kernel().group_id(0));
    wts_bo_ = xrt::bo(context_->get_device(), wts_size_, XRT_BO_FLAGS_HOST_ONLY,
                      context_->get_kernel().group_id(0));
    tmp_bo_ = xrt::bo(context_->get_device(), tmp_size_, XRT_BO_FLAGS_HOST_ONLY,
                      context_->get_kernel().group_id(0));
    logical_ofm_bo_ = &ifm_bo_;

    ifm_ptr_ = ifm_bo_.map<int8_t*>();
    memset(ifm_ptr_, 0, ifm_size_ + ofm_size_);
    wts_ptr_ = wts_bo_.map<int8_t*>();
    memset(wts_ptr_, 0, wts_size_);
    ofm_ptr_ = logical_ofm_bo_->map<int8_t*>();
    int8_t* tmp_buff_ptr_ = tmp_bo_.map<int8_t*>();
    memset(tmp_buff_ptr_, 0, tmp_size_);
  }
#ifdef HW_RUNNER_USE_CMDLIST
  runlist_wrapped_.emplace_back(context_->get_context());
  xrt::runlist& cmdq = runlist_wrapped_[0];
#endif
  std::uint64_t fm_0, fm_1, fm_2, fm_3, fm_4;
  XRTRunOffset dummy_run_offset;
  for (int sg_id = 0; sg_id < run_offsets.size(); sg_id++) {
    hw_runner_helper::set_fm(
        ifm_bo_, wts_bo_, tmp_bo_, *logical_ofm_bo_, bo_order_vec_[sg_id],
        run_offsets[sg_id], fm_0, fm_1, fm_2, fm_3, fm_4,
        ctrl_pkt_bo_vec_.empty() ? nullptr : &ctrl_pkt_bo_vec_[sg_id]);
    // create xrt run obj in advance
#ifdef HW_RUNNER_USE_CMDLIST
    uint32_t idx = kernel_index[sg_id];
    auto run = xrt::run(context_->get_kernel(idx));
    run.set_arg(0, OPCODE);
    run.set_arg(1, instr_bo_vec_[sg_id]);
    run.set_arg(2, instr_bo_vec_[sg_id].size() / sizeof(uint32_t));
    run.set_arg(3, fm_0);
    run.set_arg(4, fm_1);
    run.set_arg(5, fm_2);
    run.set_arg(6, fm_3);
    run.set_arg(7, fm_4);
    cmdq.add(run);
#else
    run_obj_vec_.emplace_back(context_->get_kernel(sg_id));
    run_obj_vec_.back().set_arg(0, OPCODE);
    run_obj_vec_.back().set_arg(1, instr_bo_vec_[sg_id]);
    run_obj_vec_.back().set_arg(2,
                                instr_bo_vec_[sg_id].size() / sizeof(uint32_t));
    run_obj_vec_.back().set_arg(3, fm_0);
    run_obj_vec_.back().set_arg(4, fm_1);
    run_obj_vec_.back().set_arg(5, fm_2);
    run_obj_vec_.back().set_arg(6, fm_3);
    run_obj_vec_.back().set_arg(7, fm_4);
#endif
  }
  ifm_ptr_ = ifm_bo_.map<int8_t*>();
  wts_ptr_ = wts_bo_.map<int8_t*>();
  ofm_ptr_ = logical_ofm_bo_->map<int8_t*>();
}

void hw_runner::set_bo_order(BO_ORDER order) { bo_order_ = order; }

void hw_runner::set_bo_order_vec(std::vector<BO_ORDER> order) {
  bo_order_vec_ = order;
}

void hw_runner::get_bo_ptrs(int8_t*& ifm_ptr, int8_t*& wts_ptr,
                            int8_t*& ofm_ptr) {
  ifm_ptr = ifm_ptr_;
  wts_ptr = wts_ptr_;
  ofm_ptr = ofm_ptr_;
}
void hw_runner::pre_run_bo_sync() {
  tmp_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  wts_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // init bo space, require sync agin in run body
  // ifm_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  // ofm_bo_.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
}

void hw_runner::load_ctrl_pkt_bin(const std::vector<std::string>& ctrlPktbins) {
  for (const auto& ctrlPktbin : ctrlPktbins) {
    // std::cout << "ctrl pkt size: " << ctrlPktbin.size() << std::endl;
    ctrl_pkt_bo_vec_.emplace_back(context_->get_device(), ctrlPktbin.size(),
                                  XRT_BO_FLAGS_HOST_ONLY,
                                  context_->get_kernel().group_id(1));

    ctrl_pkt_bo_vec_.back().write(ctrlPktbin.data());
    ctrl_pkt_bo_vec_.back().sync(XCL_BO_SYNC_BO_TO_DEVICE);
  }
}

void hw_runner::load_txn_bin(const std::string& txnbin) {
  std::vector<std::string> txnbins = {txnbin};
  this->load_txn_bin(txnbins);
}

void hw_runner::load_txn_bin(const std::vector<std::string>& txnbins) {
  for (const auto& txnbin : txnbins) {
    auto instr_buf = vaip_vaiml_custom_op::transaction_op(txnbin);
    instr_bo_vec_.emplace_back(
        context_->get_device(), instr_buf.get_txn_instr_size(),
        XCL_BO_FLAGS_CACHEABLE, context_->get_kernel().group_id(1));

    instr_bo_vec_.back().write(instr_buf.get_txn_op().data());
    instr_bo_vec_.back().sync(XCL_BO_SYNC_BO_TO_DEVICE);
  }
}

// Performs the run on hardware
int hw_runner::run(void* ifm, void* wts, void* ofm) {
  {
    TIMER(SYNC_IFM, "sync ifm bo ")
    ifm_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  }

  {
    TIMER(XRT_RUN, "xrt run ")
#ifdef HW_RUNNER_USE_CMDLIST
    runlist_wrapped_[0].execute();
    runlist_wrapped_[0].wait();
#else
    for (int i = 0; i < run_obj_vec_.size(); i++) {
      run_obj_vec_[i].start();
      run_obj_vec_[i].wait2();
    }
#endif
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

// } // namespace