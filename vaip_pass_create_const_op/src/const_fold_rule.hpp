/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

#pragma once
#include <glog/logging.h>
#include <optional>
#include <type_traits>
#include <unordered_map>
#include <vaip/vaip.hpp>
#include <vitis/ai/env_config.hpp>
DEF_ENV_PARAM(DEBUG_CONST_FOLDING, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_CONST_FOLDING) >= n)
namespace vaip_pass_create_const_op {
using namespace vaip_core;
struct TensorView {
  gsl::span<char> data;
  int data_type;
  std::vector<int64_t> shape;
};

template <typename T> struct GTensorView {
  gsl::span<T> data;
  std::vector<int64_t> shape;
};

class ConstantFoldRule : public BaseRule {
public:
  using internal_flag_t = std::nullptr_t;
  using action_t =
      std::function<bool(IPass& self, const Node& node, TensorView output,
                         const std::vector<TensorView>& inputs)>;
  template <typename... T>
  explicit ConstantFoldRule(IPass& pass, const std::string& op_type,
                            T&&... op_implentation);

private:
  ConstantFoldRule(std::nullptr_t, IPass& pass, const std::string& op_type,
                   std::vector<action_t>&& action);

public:
  virtual ~ConstantFoldRule();
  bool compute(const Node& node, TensorView output,
               const std::vector<TensorView>& inputs);

private:
  virtual bool apply_once(onnxruntime::Graph* graph,
                          const onnxruntime::Node* node) override final;

private:
  IPass& pass_;
  const std::string op_type_;
  const std::vector<action_t> action_;
};

template <typename T, class = void>
struct arg_converter_t : public std::false_type {};
template <typename T, class = void>
struct is_type_supported_t : public std::false_type {};
template <> struct is_type_supported_t<float> : public std::true_type {
  static constexpr int expected_data_type = onnx::TensorProto_DataType_FLOAT;
};
template <> struct is_type_supported_t<int8_t> : public std::true_type {
  static constexpr int expected_data_type = onnx::TensorProto_DataType_INT8;
};
template <> struct is_type_supported_t<int32_t> : public std::true_type {
  static constexpr int expected_data_type = onnx::TensorProto_DataType_INT32;
};
template <> struct is_type_supported_t<int64_t> : public std::true_type {
  static constexpr int expected_data_type = onnx::TensorProto_DataType_INT64;
};

template <typename T>
struct arg_converter_t<GTensorView<T>,
                       std::enable_if_t<is_type_supported_t<T>::value>>
    : public std::true_type {
  static constexpr int expected_data_type =
      is_type_supported_t<T>::expected_data_type;
  static constexpr bool is_required = true;
  static GTensorView<T> convert(const std::vector<TensorView>& args,
                                size_t index, int& convert_ok) {
    if (!convert_ok) {
      return {};
    }
    if (index >= args.size()) {
      LOG(WARNING) << "required arg missing "
                   << "index : " << index;
      convert_ok = 0;
      return {};
    }
    auto& arg = args[index];
#define MY_CHECK_TYPE(type, onnx_type)                                         \
  do {                                                                         \
    if (std::is_same_v<T, type> && arg.data_type != onnx_type) {               \
      LOG_IF(WARNING, false)                                                   \
          << " data type mismatch. " #type " expected but element type is "    \
          << arg.data_type << " index " << index << " "                        \
          << " type = " << typeid(T).name();                                   \
      convert_ok = 0;                                                          \
      return {};                                                               \
    }                                                                          \
  } while (0)
    MY_CHECK_TYPE(float, onnx::TensorProto_DataType_FLOAT);
    MY_CHECK_TYPE(int8_t, onnx::TensorProto_DataType_INT8);
    MY_CHECK_TYPE(int32_t, onnx::TensorProto_DataType_INT32);
    return GTensorView<T>{gsl::span<T>(reinterpret_cast<T*>(arg.data.data()),
                                       arg.data.size_bytes() / sizeof(T)),
                          arg.shape};
  }
};

template <typename T>
struct arg_converter_t<T, std::enable_if_t<is_type_supported_t<T>::value>>
    : public std::true_type {
  static constexpr int expected_data_type =
      is_type_supported_t<T>::expected_data_type;
  static constexpr bool is_required = true;
  static T convert(const std::vector<TensorView>& args, size_t index,
                   int& convert_ok) {
    if (!convert_ok) {
      return T();
    }
    auto s = arg_converter_t<GTensorView<T>>::convert(args, index, convert_ok);
    if (s.data.size() != 1u) {
      MY_LOG(1) << " scalar expected : "
                << "index " << index << " "
                << "type= " << typeid(T).name();
      convert_ok = 0;
    }
    if (!convert_ok) {
      return T();
    }
    return s.data[0];
  }
};

template <typename T>
struct arg_converter_t<std::optional<T>,
                       std::enable_if_t<arg_converter_t<T>::value>>
    : public std::true_type {
  static constexpr int expected_data_type =
      is_type_supported_t<T>::expected_data_type;
  static constexpr bool is_required = false;
  static std::optional<T> convert(const std::vector<TensorView>& args,
                                  size_t index, int& convert_ok) {
    if (!convert_ok) {
      return std::nullopt;
    }
    if (index >= args.size()) {
      return std::nullopt;
    }
    return std::optional<T>(
        arg_converter_t<T>::convert(args, index, convert_ok));
  }
};

template <typename T, typename R, typename... Args>
bool calculate_proxy1(int& convert_ok, IPass& pass, const Node& node,
                      const T& self,
                      bool (T::*f)(IPass&, const Node&, R, Args...) const,
                      R&& r, Args&&... args) {
  auto ret = false;
  if (convert_ok) {
    ret = std::invoke(f, self, pass, node, std::forward<R>(r),
                      std::forward<Args>(args)...);
  }
  return ret;
}
template <typename T, typename R, typename... Args, size_t... Index>
bool calculate_proxy0(IPass& pass, const Node& node, const T& self,
                      bool (T::*f)(IPass&, const Node&, R, Args...) const,
                      TensorView output, const std::vector<TensorView>& inputs,
                      std::integer_sequence<size_t, Index...>) {
  auto convert_ok = 1;
  if (arg_converter_t<R>::expected_data_type != output.data_type) {
    LOG_IF(WARNING, false)
        << "cancel constant folding, return type mismatch: actual type= "
        << output.data_type << " but " << arg_converter_t<R>::expected_data_type
        << " expected. node=" << node_as_string(node);
    return false;
  }
  auto n_of_args = inputs.size();
  for (auto i : {Index...}) {
    if (i >= n_of_args) {
      if (arg_converter_t<R>::is_required) {
        LOG(WARNING)
            << "cancel constant folding, required argument missing: arg_index="
            << i << " num of args=" << n_of_args
            << ". node=" << node_as_string(node);
        return false;
      }
    }
  }
  return calculate_proxy1(
      convert_ok, pass, node, self, f,
      arg_converter_t<R>::convert(std::vector<TensorView>{output}, 0,
                                  convert_ok),
      (arg_converter_t<Args>::convert(inputs, Index, convert_ok))...);
}

template <typename T, typename R, typename... Args>
bool calculate_proxy(IPass& pass, const Node& node, const T& self,
                     bool (T::*f)(IPass&, const Node&, R, Args...) const,
                     TensorView output, const std::vector<TensorView>& inputs) {
  return calculate_proxy0(pass, node, self, f, output, inputs,
                          std::make_index_sequence<sizeof...(Args)>());
}
template <typename T>
ConstantFoldRule::action_t create_action(T&& op_implentation) {
  return [op_implentation](IPass& pass, const Node& node, TensorView output,
                           const std::vector<TensorView>& inputs) {
    return calculate_proxy(pass, node, op_implentation,
                           &std::remove_reference_t<T>::operator(), output,
                           inputs);
  };
}

template <typename... T>
ConstantFoldRule::ConstantFoldRule(IPass& pass, const std::string& op_type,
                                   T&&... op_implentation)
    : ConstantFoldRule(
          nullptr, pass, op_type,
          std::vector<action_t>{create_action(op_implentation)...}) {}

} // namespace vaip_pass_create_const_op
