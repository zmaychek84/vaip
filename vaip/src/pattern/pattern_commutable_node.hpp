/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */
#include "vaip/pattern.hpp"

namespace vaip_core {

class PatternCommutableNode : public Pattern {
public:
  explicit PatternCommutableNode(int id, const std::string& op_type,
                                 const std::shared_ptr<Pattern>& arg1,
                                 const std::shared_ptr<Pattern>& arg2);
  ~PatternCommutableNode();

private:
  virtual BinderBuilderPtr
  match_uncached(const onnxruntime::Graph& graph, const NodeInput& node_input,
                 const BinderBuilder& binder) const override final;
  virtual std::string debug_string() const final;
  virtual void dump_to_proto_imp(RootPatternProto& pattern_proto,
                                 PatternProto& this_proto) const override;

private:
  const std::string op_type_;
  const std::shared_ptr<Pattern> arg1_;
  const std::shared_ptr<Pattern> arg2_;
};
} // namespace vaip_core
