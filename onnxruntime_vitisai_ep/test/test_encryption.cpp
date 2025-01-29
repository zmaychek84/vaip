/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

#include "../../encryption/src/encryption.hpp"
#include <iostream>

int main(int argc, char* argv[]) {
  std::string key =
      "89703f950ed9f738d956f6769d7e45a385d3c988ca753838b5afbc569ebf35b2";
  std::string str = "A tall white fountain played.";
  auto ciphertext = vaip_encryption::aes_encryption(str, key);
  std::cout << str << std::endl;
  std::cout << ciphertext << std::endl;
  std::cout << vaip_encryption::aes_decryption(ciphertext, key);
  return 0;
}
