/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */
#pragma once

#include <numeric>

#include "pp_softmax_instr_compiler.hpp"
#include "xf_aie_const.hpp"

class SoftMax {
public:
  SoftMax(xrt::device& device, xrt::kernel& kernel,
          std::vector<int64_t>& in_shape, std::vector<int64_t>& out_shape) {
    // Input/output sizes
    int64_t in_size = 1;
    in_size = std::accumulate(in_shape.begin(), in_shape.end(), in_size,
                              std::multiplies());
    int64_t out_size = 1;
    out_size = std::accumulate(out_shape.begin(), out_shape.end(), out_size,
                               std::multiplies());

    rtpData = new uint16_t[RTP_SIZE >> 1];

    uint16_t opcode = PP_SOFTMAX;
    rtpData[PP_OPCODE] = opcode;
    rtpData[PP_SOFTMAX_RTP_IN_ELEM] = static_cast<uint16_t>(in_size);
    rtpData[PP_SOFTMAX_RTP_OUT_ELEM] = static_cast<uint16_t>(out_size);

    std::cout << "in, out sizes: " << in_size << ", " << out_size << "\n";

    size_t metadata_size = RTP_SIZE; // bytes
    size_t metadata_words = 1 + (metadata_size / sizeof(uint32_t));

    // Create a buffer to hold metadata + instructions
    instr_buffer_softmax = new uint32_t[30000];

    // Load metadata
    instr_buffer_softmax[0] = static_cast<uint32_t>(metadata_words);
    memcpy(instr_buffer_softmax + 1, rtpData, RTP_SIZE);

    // Create instance of compiler
    auto compiler = std::make_unique<SoftmaxInstrCompiler>(
        instr_buffer_softmax + metadata_words);
    instr_counter = compiler->generate(rtpData);
    instr_counter = static_cast<uint32_t>(instr_counter + metadata_words);

    // Create BOs
    bo_in = xrt::bo(device, in_size * sizeof(uint16_t), XRT_BO_FLAGS_HOST_ONLY,
                    kernel.group_id(2));
    bo_out = xrt::bo(device, out_size * sizeof(uint16_t),
                     XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  }

  ~SoftMax() {
    if (instr_buffer_softmax)
      delete[] instr_buffer_softmax;
    if (rtpData)
      delete[] rtpData;
  }

  uint16_t* get_host_buffer_in() { return bo_in.map<uint16_t*>(); }
  uint16_t* get_host_buffer_out() { return bo_out.map<uint16_t*>(); }

  size_t get_instr_size() { return instr_counter * sizeof(uint32_t); }

  void sync_instructions(xrt::bo& bo_instr) {
    memcpy(bo_instr.map<void*>(), instr_buffer_softmax,
           instr_counter * sizeof(uint32_t));
    bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE, instr_counter * sizeof(uint32_t),
                  0);
  }

  void exec(xrt::kernel& kernel, xrt::bo& bo_instr) {
    // Sync Input BOs
    bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // Set kernel argument and trigger it to run
    auto run = kernel(bo_instr, instr_counter, bo_in, bo_out);

    // Wait for kernel to be done
    run.wait();

    // Sync outputs(score and indices) back to host
    bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  }

public:
  xrt::bo bo_in, bo_out;
  uint32_t* instr_buffer_softmax;
  uint16_t* rtpData;
  uint32_t instr_counter;
};
