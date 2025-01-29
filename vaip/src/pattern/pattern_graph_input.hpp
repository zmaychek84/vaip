/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

#pragma once

#include "vaip/pattern.hpp"
namespace vaip_core {

class PatternGraphInput : public Pattern {
public:
  explicit PatternGraphInput(int id);
  ~PatternGraphInput();

private:
  virtual BinderBuilderPtr
  match_uncached(const onnxruntime::Graph& graph, const NodeInput& node_input,
                 const BinderBuilder& binder) const override final;
  virtual std::string debug_string() const override;
  virtual std::string virtualize_label() const override;
  virtual void dump_to_proto_imp(RootPatternProto& pattern_proto,
                                 PatternProto& this_proto) const override final;
};
} // namespace vaip_core
