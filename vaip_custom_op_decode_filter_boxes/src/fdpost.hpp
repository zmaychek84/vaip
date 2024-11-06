/*
 *     The Xilinx Vitis AI Vaip in this distribution are provided under the
 * following free and permissive binary-only license, but are not provided in
 * source code form.  While the following free and permissive license is similar
 * to the BSD open source license, it is NOT the BSD open source license nor
 * other OSI-approved open source license.
 *
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

#include <fstream>
#include <string>

#include "config_fdpost.h"
#include "in_anchors_fixed.h"
#include "in_box_coord_fixed.h"
#include "in_scores_fixed.h"
#include "instructions.fdpost.h"
#include "xf_aie_const.hpp"

class FDPOST {
public:
  FDPOST(xrt::device& device, xrt::kernel& kernel) {
    int BLOCK_SIZE_inbox_Bytes = IN_BOX_SIZE;
    int BLOCK_SIZE_anchor_Bytes = IN_ANCHOR_SIZE;
    int BLOCK_SIZE_score_Bytes = IN_SCORE_SIZE;
    int BLOCK_SIZE_out_Bytes = OUT_SIZE;

    inboxData = (int8_t*)malloc(BLOCK_SIZE_inbox_Bytes);
    anchorData = (uint8_t*)malloc(BLOCK_SIZE_anchor_Bytes);
    scoreData = (uint8_t*)malloc(BLOCK_SIZE_score_Bytes);
    outData = (uint16_t*)malloc(BLOCK_SIZE_out_Bytes);
    outRef = (uint16_t*)malloc(BLOCK_SIZE_out_Bytes);

    int8_t* inbox = (int8_t*)inboxData;
    uint8_t* anchor = (uint8_t*)anchorData;
    uint8_t* score = (uint8_t*)scoreData;

    for (int i = 0; i < IN_BOX_ELEM; i++) {
      inbox[i] = in_box_coord_fixed[i];
    }
    for (int i = 0; i < IN_ANCHOR_ELEM; i++) {
      anchor[i] = in_anchors_fixed[i];
    }
    for (int i = 0; i < IN_SCORE_ELEM; i++) {
      score[i] = in_scores_fixed[i];
    }

    size_t instr_word_size = opcode_fd_post.size();
    size_t metadata_size = RTP_SIZE; // bytes
    size_t metadata_words = 1 + (metadata_size / sizeof(uint32_t));
    instr_words = static_cast<uint32_t>(instr_word_size + metadata_words);

    // Create a buffer to hold metadata + instructions
    instr_buffer = new uint32_t[instr_words];
    rtpData = new uint16_t[RTP_SIZE >> 1];

    uint16_t opcode = PP_FD_POST;

    rtpData[PP_OPCODE] = opcode;
    rtpData[PP_FDPOST_RTP_SCORE_THRESH] = SCORE_THRESH;
    rtpData[PP_FDPOST_RTP_IOU_THRESH] = IOU_THRESH;
    rtpData[PP_FDPOST_RTP_MAX_DET] = MAX_DET;
    // TODO: Get this every frame?
    rtpData[PP_RTP_IMG_HEIGHT_IN] = 300;
    rtpData[PP_RTP_IMG_WIDTH_IN] = 300;

    // Load metadata
    instr_buffer[0] = static_cast<uint32_t>(metadata_words);
    memcpy(instr_buffer + 1, rtpData, RTP_SIZE);
    // Load instructions
    memcpy(instr_buffer + metadata_words, opcode_fd_post.data(),
           (opcode_fd_post.size() * sizeof(uint32_t)));

    // Create BOs
    bo_inbox = xrt::bo(device, BLOCK_SIZE_inbox_Bytes, XRT_BO_FLAGS_HOST_ONLY,
                       kernel.group_id(2));
    bo_anchor = xrt::bo(device, BLOCK_SIZE_anchor_Bytes, XRT_BO_FLAGS_HOST_ONLY,
                        kernel.group_id(3));
    bo_score = xrt::bo(device, BLOCK_SIZE_score_Bytes, XRT_BO_FLAGS_HOST_ONLY,
                       kernel.group_id(4));
    bo_out = xrt::bo(device, BLOCK_SIZE_out_Bytes, XRT_BO_FLAGS_HOST_ONLY,
                     kernel.group_id(5));

    // Init BOs (Testcase specific per config.h)
    DATA_TYPE_IN* bufInBox = bo_inbox.map<DATA_TYPE_IN*>();
    memcpy(bufInBox, inboxData, BLOCK_SIZE_inbox_Bytes);

    DATA_TYPE_IN* bufAnchor = bo_anchor.map<DATA_TYPE_IN*>();
    memcpy(bufAnchor, anchorData, BLOCK_SIZE_anchor_Bytes);

    DATA_TYPE_IN* bufScore = bo_score.map<DATA_TYPE_IN*>();
    memcpy(bufScore, scoreData, BLOCK_SIZE_score_Bytes);
  }

  ~FDPOST() {
    if (instr_buffer)
      delete[] instr_buffer;
    if (rtpData)
      delete[] rtpData;
    if (inboxData)
      free(inboxData);
    if (anchorData)
      free(anchorData);
    if (scoreData)
      free(scoreData);
    if (outData)
      free(outData);
    if (outRef)
      free(outRef);
  }

  uint8_t* get_host_buffer_boxes() { return bo_inbox.map<uint8_t*>(); }
  uint8_t* get_host_buffer_scores() { return bo_score.map<uint8_t*>(); }
  uint16_t* get_host_buffer_out_box() { return bo_out.map<uint16_t*>(); }
  size_t get_instr_size() { return instr_words * sizeof(uint32_t); }

  void sync_instructions(xrt::bo& bo_instr) {
    memcpy(bo_instr.map<void*>(), instr_buffer, instr_words * sizeof(int));
    bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE, instr_words * sizeof(uint32_t), 0);
  }

  void exec(xrt::kernel& kernel, xrt::bo& bo_instr) {
    // Sync Input BOs
    bo_inbox.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_anchor.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_score.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // Set kernel argument and trigger it to run
    auto run =
        kernel(bo_instr, instr_words, bo_inbox, bo_anchor, bo_score, bo_out);

    // Wait for kernel to be done
    run.wait();

    // Sync output frame back to host
    bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  }

public:
  xrt::bo bo_inbox, bo_anchor, bo_score, bo_out;
  uint32_t* instr_buffer;
  uint16_t* rtpData;
  uint32_t instr_words;
  int8_t* inboxData;
  uint8_t* anchorData;
  uint8_t* scoreData;
  uint16_t* outData;
  uint16_t* outRef;
};
