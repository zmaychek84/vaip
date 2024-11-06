/*
 *     The Xilinx Vitis AI Vaip in this distribution are provided under the
 * following free and permissive binary-only license, but are not provided in
 * source code form.  While the following free and permissive license is similar
 * to the BSD open source license, it is NOT the BSD open source license nor
 * other OSI-approved open source license.
 *
 *      Copyright (C) 2022 Xilinx, Inc. All rights reserved.
 *      Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights
 * reserved.
 *
 *      Redistribution and use in binary form only, without modification, is
 * permitted provided that the following conditions are met:
 *
 *      1. Redistributions must reproduce the above copyright notice, this list
 * of conditions and the following disclaimer in the documentation and/or other
 * materials provided with the distribution.
 *
 *      2. The name of Xilinx, Inc. may not be used to endorse or promote
 * products redistributed with this software without specific prior written
 * permission.
 *
 *      THIS SOFTWARE IS PROVIDED BY XILINX, INC. "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL XILINX, INC. BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 *      PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE
 */
#pragma once

#include <cmath>
#include <numeric>

#include "pp_resize_instr_compiler.hpp"
#include "xf_aie_const.hpp"

template <int FBITS = 16> uint32_t compute_scalefactor(int M, int N) {
  float x_scale = (float)M / (float)N;
  float scale = x_scale * (1 << FBITS);
  return (uint32_t)(std::roundf(scale));
}

class ResizeDown {
public:
  ResizeDown(xrt::device& device, xrt::kernel& kernel,
             std::vector<int64_t>& in_shape, std::vector<int64_t>& out_shape,
             std::vector<int64_t>& rsz_out_shape) {
    // Input/output sizes
    int64_t in_size = 1;
    in_size = std::accumulate(in_shape.begin(), in_shape.end(), in_size,
                              std::multiplies());

    int64_t out_size = 1;
    out_size = std::accumulate(rsz_out_shape.begin(), rsz_out_shape.end(),
                               out_size, std::multiplies());

    rtpData = new uint16_t[RTP_SIZE >> 1];

    uint16_t opcode = PP_RESIZE_DOWN;

    rtpData[PP_OPCODE] = opcode;
    rtpData[PP_RTP_IMG_WIDTH_IN] = static_cast<uint16_t>(in_shape[1]);
    rtpData[PP_RTP_IMG_HEIGHT_IN] = static_cast<uint16_t>(in_shape[0]);
    rtpData[PP_RTP_IMG_WIDTH_OUT] = static_cast<uint16_t>(rsz_out_shape[1]);
    rtpData[PP_RTP_IMG_HEIGHT_OUT] = static_cast<uint16_t>(rsz_out_shape[0]);

    uint32_t scale_x_fix = compute_scalefactor(
        static_cast<int>(in_shape[1]), static_cast<int>(rsz_out_shape[1]));
    uint32_t scale_y_fix = compute_scalefactor(
        static_cast<int>(in_shape[0]), static_cast<int>(rsz_out_shape[0]));

    rtpData[PP_RESIZE_DOWN_RTP_SCALE_X_LO] =
        static_cast<uint16_t>(scale_x_fix % 65536);
    rtpData[PP_RESIZE_DOWN_RTP_SCALE_X_HI] =
        static_cast<uint16_t>(scale_x_fix >> 16);
    rtpData[PP_RESIZE_DOWN_RTP_SCALE_Y_LO] =
        static_cast<uint16_t>(scale_y_fix % 65536);
    rtpData[PP_RESIZE_DOWN_RTP_SCALE_Y_HI] =
        static_cast<uint16_t>(scale_y_fix >> 16);
    rtpData[PP_RESIZE_DOWN_RTP_CHANNELS] =
        static_cast<uint16_t>(rsz_out_shape[2]);

    size_t metadata_size = RTP_SIZE; // bytes
    size_t metadata_words = 1 + (metadata_size / sizeof(uint32_t));

    // Create a buffer to hold metadata + instructions
    instr_buffer_resize = new uint32_t[30000];

    // Load metadata
    instr_buffer_resize[0] = static_cast<uint32_t>(metadata_words);
    memcpy(instr_buffer_resize + 1, rtpData, RTP_SIZE);

    // Create instance of compiler
    auto compiler = std::make_unique<ResizeInstrCompiler>(instr_buffer_resize +
                                                          metadata_words);
    instr_counter = compiler->generate(rtpData);
    instr_counter = static_cast<uint32_t>(instr_counter + metadata_words);
    // Create BOs
    bo_frame =
        xrt::bo(device, in_size, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(2));
    bo_out =
        xrt::bo(device, out_size, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  }

  ~ResizeDown() {
    if (instr_buffer_resize)
      delete[] instr_buffer_resize;
    if (rtpData)
      delete[] rtpData;
  }

  uint8_t* get_host_buffer_out() { return bo_out.map<uint8_t*>(); }
  uint8_t* get_host_buffer_in() { return bo_frame.map<uint8_t*>(); }
  xrt::bo& get_output_bo() { return bo_out; }
  size_t get_instr_size() { return instr_counter * sizeof(uint32_t); }

  void sync_instructions(xrt::bo& bo_instr) {
    memcpy(bo_instr.map<void*>(), instr_buffer_resize,
           instr_counter * sizeof(uint32_t));
    bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE, instr_counter * sizeof(uint32_t),
                  0);
  }

  void exec(xrt::kernel& kernel, xrt::bo& bo_instr) {
    // Sync Input BOs
    bo_frame.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // Set kernel argument and trigger it to run
    auto run = kernel(bo_instr, instr_counter, bo_frame, bo_out);

    // Wait for kernel to be done
    run.wait();

    // Sync Output BOs
    bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  }

public:
  xrt::bo bo_frame, bo_out;
  uint32_t* instr_buffer_resize;
  uint16_t* rtpData;
  uint32_t instr_counter;
};
