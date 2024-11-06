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
#include <iostream>
#include <string>
#include <unordered_map>

class ReportInventory {
private:
  // Private constructor to prevent direct instantiation
  ReportInventory() {}

  // Static unordered_map that is shared across all instances
  static std::unordered_map<std::string, std::string> dataMap;

public:
  // Delete copy constructor and assignment operator to prevent copying
  ReportInventory(const ReportInventory&) = delete;
  ReportInventory& operator=(const ReportInventory&) = delete;

  // Public method to get the single instance of the class
  static ReportInventory& getInstance();

  // Method to add key-value pairs to the unordered_map
  void addData(const std::string& key, const std::string& value);

  // Method to get the value associated with a key
  std::string getData(const std::string& key);

  // Method to print all the data in the map
  void printData();

  // Destructor that prints the op summary upon destruction
  ~ReportInventory() {
    // std::cout << "Destructing ReportInventory instance and printing map:\n";
    printData();
    // cnt++;
  }
};

// Define the macro to handle method execution and conditional logging on error
#define TRY_EXECUTE_WITH_LOG(method_call, dry_run, log_method, ...)            \
  try {                                                                        \
    method_call;                                                               \
  } catch (const std::exception& e) {                                          \
    if (dry_run) {                                                             \
      log_method(__VA_ARGS__);                                                 \
    } else {                                                                   \
      std::cerr << "Exception " << e.what() << std::endl;                      \
      throw e;                                                                 \
    }                                                                          \
  }
