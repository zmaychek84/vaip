/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */
#ifdef __GNUC__
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wpedantic"
#  pragma GCC diagnostic ignored "-Wconversion"
#endif

#include "vaip/dd/coeffs.hpp"
#include "vaip/dd/dd_utils.hpp"

#ifdef __GNUC__
#  pragma GCC diagnostic pop
#endif
#include "vaip/vaip.hpp"
#include "vitis/ai/env_config.hpp"
#include <glog/logging.h>

#include "vaip/pattern_zoo.hpp"
#include "vaip/vaip_ort.hpp"

DEF_ENV_PARAM(DEBUG_DD_PATTERN, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_DD_PATTERN) >= n)
using namespace vaip_core;

struct Merge_top_fuse {
  Merge_top_fuse(IPass& self) : self_{self} {}

  void process(IPass& self, Graph& graph) {

    auto& pass_proto =
        const_cast<vaip_core::PassProto&>(self_.get_pass_proto());

    MY_LOG(1) << "Fuse Pattern in top_fuse"
              << pass_proto.pass_fusion_param().pattern().pattern_name();

    pass_proto.set_name("vaip_pass_dd_merge_fuse_general");
    pass_proto.set_plugin("vaip-pass_dd_merge_fuse_general");

    auto passes = std::vector<std::shared_ptr<vaip_core::IPass>>{};
    passes.push_back(IPass::create_pass(self_.get_context(), pass_proto));

    IPass::run_passes(passes, graph);
  }

public:
  IPass& self_;
};

DEFINE_VAIP_PASS(Merge_top_fuse, vaip_pass_dd_merge_top_fuse)
