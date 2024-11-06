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

#include <numeric>

#include "pp_norm_instr_compiler.hpp"
#include "xf_aie_const.hpp"

static void get_alpha_beta(std::vector<float>& mean,
                           std::vector<float>& std_deviation,
                           std::array<unsigned char, 4>& alpha,
                           std::array<char, 4>& beta, int fbits_alpha = 0,
                           int fbits_beta = 4) {
  for (int i = 0; i < 4; i++) {
    float a_v = mean[i] * static_cast<float>((1 << fbits_alpha));
    float b_v;
    if (std_deviation[i] != 0)
      b_v = (1.0f / std_deviation[i]) * static_cast<float>((1 << fbits_beta));
    else
      b_v = 0.0f;

    alpha[i] = (unsigned char)a_v;
    beta[i] = (char)b_v;

    assert((a_v < (1 << 8)) && "alpha values exceeds 8 bit precison");
    assert((b_v < (1 << 8)) && "beta values exceeds 8 bit precison");
  }
}

class Normalize {
public:
  Normalize(xrt::device& device, xrt::kernel& kernel,
            std::vector<int64_t>& in_shape, std::vector<int64_t>& out_shape,
            std::vector<int64_t>& norm_out_shape, std::vector<int>& fl_bits,
            std::vector<float>& mean, std::vector<float>& std_deviation) {
    // Input/output sizes
    int64_t out_size = 1;
    out_size = std::accumulate(norm_out_shape.begin(), norm_out_shape.end(),
                               out_size, std::multiplies());

    // Precision bits
    int fbits_alpha = fl_bits[0];
    int fbits_beta = fl_bits[1];
    int fbits_out = fl_bits[2];

    std::array<unsigned char, 4> alpha;
    std::array<char, 4> beta;
    // Get alpha/beta values
    get_alpha_beta(mean, std_deviation, alpha, beta, fbits_alpha, fbits_beta);

    rtpData = new uint16_t[RTP_SIZE >> 1];

    uint16_t opcode = PP_NORM;
    rtpData[PP_OPCODE] = opcode;
    rtpData[PP_RTP_IMG_WIDTH_IN] = static_cast<uint16_t>(norm_out_shape[1]);
    rtpData[PP_RTP_IMG_HEIGHT_IN] = static_cast<uint16_t>(norm_out_shape[0]);
    rtpData[PP_RTP_IMG_WIDTH_OUT] = static_cast<uint16_t>(norm_out_shape[1]);
    rtpData[PP_RTP_IMG_HEIGHT_OUT] = static_cast<uint16_t>(norm_out_shape[0]);
    rtpData[PP_NORM_RTP_FBITS_ALPHA] = static_cast<uint16_t>(fbits_alpha);
    rtpData[PP_NORM_RTP_FBITS_BETA] = static_cast<uint16_t>(fbits_beta);
    rtpData[PP_NORM_RTP_FBITS_OUT] = static_cast<uint16_t>(fbits_out);

    for (int i = 0; i < 4; i++) {
      rtpData[PP_NORM_RTP_ALPHA_0 + i] = alpha[i];
      rtpData[PP_NORM_RTP_BETA_0 + i] = beta[i];
    }

    size_t metadata_size = RTP_SIZE; // bytes
    size_t metadata_words = 1 + (metadata_size / sizeof(uint32_t));

    // Create a buffer to hold metadata + instructions
    instr_buffer_norm = new uint32_t[30000];

    // Load metadata
    instr_buffer_norm[0] = static_cast<uint32_t>(metadata_words);
    memcpy(instr_buffer_norm + 1, rtpData, RTP_SIZE);

    // Create instance of compiler
    auto compiler =
        std::make_unique<NormInstrCompiler>(instr_buffer_norm + metadata_words);
    instr_counter = compiler->generate(rtpData);
    instr_counter = static_cast<uint32_t>(instr_counter + metadata_words);
    // Create BO
    bo_out =
        xrt::bo(device, out_size, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  }

  ~Normalize() {
    if (instr_buffer_norm)
      delete[] instr_buffer_norm;
    if (rtpData)
      delete[] rtpData;
  }

  int8_t* get_host_buffer_out() { return bo_out.map<int8_t*>(); }

  size_t get_instr_size() { return instr_counter * sizeof(uint32_t); }

  void sync_instructions(xrt::bo& bo_instr) {
    memcpy(bo_instr.map<void*>(), instr_buffer_norm,
           instr_counter * sizeof(uint32_t));
    bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE, instr_counter * sizeof(uint32_t),
                  0);
  }

  void exec(xrt::kernel& kernel, xrt::bo& bo_instr, xrt::bo& bo_rgba) {
    // Sync Input BOs
    bo_rgba.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // Set kernel argument and trigger it to run
    auto run = kernel(bo_instr, instr_counter, bo_rgba, bo_out);

    // Wait for kernel to be done
    run.wait();

    // Sync output frame back to host
    bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  }

public:
  xrt::bo bo_out;
  uint32_t* instr_buffer_norm;
  uint16_t* rtpData;
  uint32_t instr_counter;
};
