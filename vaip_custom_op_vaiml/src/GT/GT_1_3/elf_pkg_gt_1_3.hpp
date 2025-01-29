/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */
#pragma once
#include <string>
#include <tuple>
const std::string& getElf_front_sub(std::string& model_version);
const std::string& getElf_ln_matmul_bias_ln(std::string& model_version);
const std::string& getElf_out_matmul_bias(std::string& model_version);
const std::string& getElf_transformer_layers(std::string& model_version);
