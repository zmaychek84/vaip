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

#include <chrono>
#include <iostream>

// #define TIMER(obj_name, str) vaip_vaiml_custom_op::Timer
// _timer_##obj_name(str);
#define TIMER(obj_name, str)
namespace vaip_vaiml_custom_op {

class Timer {
public:
  Timer(const std::string& str) : str_(str) {
    start_point = std::chrono::high_resolution_clock::now();
  }

  ~Timer() { Stop(); }

  void Stop() {
    auto end_point = std::chrono::high_resolution_clock::now();
    auto start =
        std::chrono::time_point_cast<std::chrono::microseconds>(start_point)
            .time_since_epoch()
            .count();
    auto end =
        std::chrono::time_point_cast<std::chrono::microseconds>(end_point)
            .time_since_epoch()
            .count();

    auto duration = end - start;

    std::cout << str_ << duration << "us\n";
  }

private:
  std::chrono::time_point<std::chrono::high_resolution_clock> start_point;
  const std::string str_;
};

} // namespace vaip_vaiml_custom_op