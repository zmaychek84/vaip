/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

#include "vaip/anchor_point.hpp"

namespace vaip_core_imp {
using namespace vaip_core;
class AnchorPointImp : public AnchorPoint {
public:
  AnchorPointImp(const NodeArg& node_arg, const Description& desciption);
  virtual ~AnchorPointImp();
  AnchorPointImp(const AnchorPointProto& proto);

private:
  virtual const AnchorPointProto& get_proto() const override final;

public:
  AnchorPointImp();

private:
  AnchorPointProto merge_proto(const AnchorPointImp* other) const;

private:
  const AnchorPointProto proto_;
};
} // namespace vaip_core_imp
