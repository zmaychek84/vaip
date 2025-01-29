/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

#include "broadcast_op.hpp"
template <typename T> static auto Mul_tmpl() {
  return [](IPass& self, const Node& node, GTensorView<T> output,
            GTensorView<T> input_a, GTensorView<T> input_b) -> bool {
    return Broadcast_op_tmpl(output, input_a, input_b,
                             [](T a, T b) -> T { return a * b; });
  };
}

template <typename... T> static std::unique_ptr<BaseRule> Mul(IPass& pass) {
  return std::make_unique<ConstantFoldRule>(pass, "Mul", Mul_tmpl<T>()...);
}
