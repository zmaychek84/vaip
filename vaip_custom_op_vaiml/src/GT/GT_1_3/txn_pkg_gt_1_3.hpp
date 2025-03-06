/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */
#pragma once
#include <string>
#include <tuple>
const std::string& getBins_front_sub_ml_txn_1_3(std::string& model_version);
const std::string&
getBins_ln_matmul_bias_ln_ctrl_pkt_1_3(std::string& model_version);
const std::string&
getBins_ln_matmul_bias_ln_ml_txn_1_3(std::string& model_version);
const std::string&
getBins_out_matmul_bias_ctrl_pkt_1_3(std::string& model_version);
const std::string&
getBins_out_matmul_bias_ml_txn_1_3(std::string& model_version);
const std::string&
getBins_transformer_layers_ctrl_pkt_1_3(std::string& model_version);
const std::string&
getBins_transformer_layers_ml_txn_1_3(std::string& model_version);
