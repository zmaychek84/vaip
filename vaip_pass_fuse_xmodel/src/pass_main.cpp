/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

#include <glog/logging.h>

#include "./fuse_xmodel.hpp"

using namespace vaip_core;
namespace {
using namespace vaip_pass_fuse_xmodel;
struct FuseXmodel {
  FuseXmodel(IPass& self) {}
  void process(const IPass& self, Graph& graph) { fuse_xmodel(graph); }
};
} // namespace

DEFINE_VAIP_PASS(FuseXmodel, vaip_pass_fuse_xmodel)
