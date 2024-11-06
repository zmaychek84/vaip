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
#include <cstring>
#if defined(_WIN32)
#  include <intrin.h>
#else
#  include <x86intrin.h>
#endif
#include <iostream>
#include <memory>
#include <mutex>
#include <set>
#include <vector>
static uint16_t float_to_bfloat16(float x) {
  uint32_t i;
  uint8_t* src = (uint8_t*)&x;
  uint8_t* tmp = (uint8_t*)&i;
  std::memcpy(tmp, src, sizeof(float));
  uint32_t lsb = (i >> 16) & 0x1;
  uint32_t bias = 0x7fff + lsb;
  i += bias;
  uint16_t y = uint16_t(i >> 16);
  return y;
}

static void fill_attn_mask_impl(uint16_t* attn_mask, int S) {
  std::memset(attn_mask, 0, S * S * sizeof(uint16_t));
  const uint16_t neg_inf_ui16 = float_to_bfloat16(-3.389e38f);
  for (int i = 0; i < S; i++) {
    uint16_t* start_ptr = attn_mask + i * S + (i + 1);
    size_t count = S - (i + 1);
#ifdef _WIN32
    __stosw(start_ptr, neg_inf_ui16, count);
#else
    std::fill(start_ptr, start_ptr + count, neg_inf_ui16);
#endif
  }
}

/*
Note(chuanliang & ltp):
This is LUT constructor for attentation mask.
We will create as many LUT(128 as step) for different S as possble, it's up to
the invoker to decide which LUT to use.
*/
class AttnMaskLUTSingleton {

private:
  // Add more in the future if required.
  static std::set<int32_t> get_seqs() { return {128, 256, 512, 1024, 2048}; }

public:
  static AttnMaskLUTSingleton& getInstance() {
    static AttnMaskLUTSingleton instance;
    return instance;
  }

  bool hasLut(int32_t S) { return idx_map_.count(S) != 0; }

  /// Should use const.
  uint16_t* getLut(int32_t S) {
    if (!hasLut(S)) {
      throw std::invalid_argument("There is not LUT for " + std::to_string(S));
    }

    auto idx = idx_map_.at(S);
    return attn_masks_lut_[idx].data();
  }

private:
  AttnMaskLUTSingleton() {
    // construct the LUT
    auto seqs = get_seqs();
    int32_t idx = 0;
    for (auto& s : seqs) {
      std::vector<uint16_t> lut;
      lut.resize(s * s);
      fill_attn_mask_impl(lut.data(), s);
      attn_masks_lut_.emplace_back(std::move(lut));
      idx_map_[s] = idx++;
    }
  }

  // index for LUT given a S
  std::unordered_map<uint16_t, int32_t> idx_map_;
  std::vector<std::vector<uint16_t>> attn_masks_lut_;

  AttnMaskLUTSingleton(const AttnMaskLUTSingleton&) = delete;
  void operator=(const AttnMaskLUTSingleton&) = delete;
};