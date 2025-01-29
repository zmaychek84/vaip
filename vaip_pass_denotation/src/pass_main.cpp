/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

#include <glog/logging.h>

#include "./denotation_pass2.hpp"
#include "vaip/vaip.hpp"
#include "vitis/ai/env_config.hpp"
using namespace vaip_core;
using namespace vaip_pass_denotation;
DEFINE_VAIP_PASS(DenotationPass, vaip_pass_denotation)
