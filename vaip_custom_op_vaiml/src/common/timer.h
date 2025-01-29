/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
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