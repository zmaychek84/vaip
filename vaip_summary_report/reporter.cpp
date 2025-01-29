/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

#include "reporter.hpp"

// Initialize the static unordered_map
std::unordered_map<std::string, std::string> ReportInventory::dataMap;

// ReportInventory instance method
ReportInventory& ReportInventory::getInstance() {
  static ReportInventory instance; // Guaranteed to be created only once
  return instance;
}

// Add data to the map
void ReportInventory::addData(const std::string& key,
                              const std::string& value) {
  dataMap[key] = value;
}

// Get data from the map
std::string ReportInventory::getData(const std::string& key) {
  if (dataMap.find(key) != dataMap.end()) {
    return dataMap[key];
  }
  return "none"; // Default value if the key doesn't exist
}

// Print all the data in the map
void ReportInventory::printData() {
  std::cout << std::endl << "Unsupported op summary:" << std::endl << std::endl;
  for (const auto& pair : dataMap) {
    std::string str = pair.second;
    std::string::size_type pos = 0;
    // Replace underscores with commas
    while ((pos = str.find('_', pos)) != std::string::npos) {
      str.replace(pos, 1, ",");
      ++pos; // Move past the replaced comma
    }
    std::cout << pair.first << ": " << str << std::endl;
    // std::cout << pair.first << ": " << pair.second << std::endl;
  }
}
