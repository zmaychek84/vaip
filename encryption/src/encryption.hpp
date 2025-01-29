/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

#include <string>
namespace vaip_encryption {
std::string aes_encryption(const std::string& str, const std::string& key);
std::string aes_decryption(const std::string& str, const std::string& key);
} // namespace vaip_encryption
