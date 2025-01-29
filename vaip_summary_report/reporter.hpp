/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
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
