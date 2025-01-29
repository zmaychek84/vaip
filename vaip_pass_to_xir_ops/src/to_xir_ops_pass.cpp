/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

#include "./to_xir_ops_pass.hpp"

#include <glog/logging.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <numeric>
#include <optional>
#include <unordered_map>

#include "vaip/node.hpp"
#include "vitis/ai/env_config.hpp"
DEF_ENV_PARAM(DEBUG_TO_XIR_PASS, "0")
DEF_ENV_PARAM(XLNX_ENABLE_CONV1D, "0")
DEF_ENV_PARAM(VAIP_ALIGNMENT_TO_ORT_SKIP_CONCAT, "0")
DEF_ENV_PARAM(VAIP_ALIGNMENT_TO_ORT_SKIP_RESIZE, "0")
DEF_ENV_PARAM(VAIP_SKIP_MATMUL_DIMS_NOT_EQUAL_4, "0")
DEF_ENV_PARAM(VAIP_SKIP_SLICE_DIMS_NOT_EQUAL_4, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_TO_XIR_PASS) >= n)
namespace vaip_pass_to_xir_ops {
using namespace vaip_core;
ToXirRule::ToXirRule(
    const std::string& onnx_op_type,
    // xir_op_type is actually the name of registered operators in Onnx
    // where `-` is converted into `_` to comply with onnx spec.
    const std::string xir_op_type, IPass& pass)
    : onnx_op_type_{onnx_op_type}, xir_op_type_{xir_op_type}, pass_{pass} {
  check_dynamic_shape();
  // In the past, xir op output tensor only supported int8 , uint8 & float32.
  // Now xir has removed this restriction, so remove this check.
  // check_xir_support_data_type();
}
ToXirRule& ToXirRule::set_xir_op_type(const std::string& xir_op_type) {
  this->xir_op_type_ = xir_op_type;
  return *this;
}

bool ToXirRule::apply(Graph& graph, const Node& node) {
  auto& op_type = node_op_type(node);
  auto& op_domain = node_op_domain(node);
  auto op_full_name = op_domain.empty() ? op_type : (op_domain + ":" + op_type);
  if (op_full_name != onnx_op_type_) {
    return false;
  }
  auto node_builder = NodeBuilder(graph, pass_);
  auto& attrs_builder = node_builder.get_attrs_builder();
  auto convert_ok = true;

  for (auto& action : actions_) {
    convert_ok = convert_ok && action(this, graph, node, attrs_builder);
  }
  if (convert_ok) {
    MY_LOG(1) << "xir op conversion : " << node_as_string(node);
    auto input_args = get_input_args(node);
    node_builder.set_input_node_args(input_args)
        .set_op_type(xir_op_type_)
        .set_anchor_point1(node)
        .build();
  } else {
  }
  return convert_ok;
}

std::vector<const NodeArg*> ToXirRule::get_input_args(const Node& node) {
  auto ret = std::vector<const NodeArg*>();
  int index = 0;
  for (auto node_arg : node_get_input_node_args(node)) {
    if (find(remove_inputs_.begin(), remove_inputs_.end(), index) ==
        remove_inputs_.end()) {
      ret.push_back(node_arg);
    }
    index = index + 1;
  }

  return ret;
}

std::vector<const NodeArg*> ToXirRule::get_output_args(const Node& node) {
  auto ret = std::vector<const NodeArg*>();
  for (auto node_arg : node_get_output_node_args(node)) {
    ret.push_back(node_arg);
  }
  return ret;
}

ToXirRule& ToXirRule::rename(
    const std::string from, const std::string to,
    std::function<AttributeProtoPtr(ToXirRule* self, const Node&)> or_else) {
  this->actions_.push_back(
      [from, to, or_else](ToXirRule* self, const Graph& /*graph*/,
                          const Node& node,
                          NodeAttributesBuilder& attrs) -> bool {
        AttributeProtoPtr to_attr;
        if (!node_has_attr(node, from)) {
          to_attr = or_else(self, node);
        } else {
          to_attr = attr_proto_clone(*node_get_attr(node, from));
        }
        auto ret = false;
        if (to_attr) {
          attrs.add(to, std::move(to_attr));
          ret = true;
        } else {
          LOG(WARNING) << "cancel xir conversion, no default attr "
                       << node_as_string(node);
          ret = false;
        }
        return ret;
      });
  return *this;
}

ToXirRule& ToXirRule::check_scalar() {
  this->actions_.push_back([=](ToXirRule* self, const Graph& /*graph*/,
                               const Node& node,
                               NodeAttributesBuilder& attrs) -> bool {
    // test case: /home/public/bevdet/LS_int.onnx
    // check input must not be a scalar, xilinx op does not support scalar.
    auto inputs = node_get_inputs(node);
    bool is_scalar = false;
    bool all_inputs_is_scalar = true;
    for (auto ni : inputs) {
      all_inputs_is_scalar =
          all_inputs_is_scalar && node_arg_is_scalar(*ni.node_arg);
    }
    is_scalar = is_scalar || all_inputs_is_scalar;
    auto outputs = node_get_output_node_args(node);
    for (auto output : outputs) {
      is_scalar = is_scalar || node_arg_is_scalar(*output);
    }
    if (is_scalar) {
      LOG(WARNING) << "cancel xir conversion, scalar input/output found: "
                   << node_as_string(node);
      return false;
    }
    if (outputs.size() > 1) {
      LOG(WARNING)
          << "cancel xir conversion, multiple outputs are not supported"
          << node_as_string(node);
      return false;
    }
    return true;
  });
  return *this;
}
ToXirRule& ToXirRule::check_node_input_zero_shape() {
  this->actions_.push_back([=](ToXirRule* self, const Graph& /*graph*/,
                               const Node& node,
                               NodeAttributesBuilder& attrs) -> bool {
    // test case: pytorch/3d_detection/pointpillars/quantized/VoxelNet_int.onnx
    auto inputs = node_get_inputs(node);
    bool is_zero_shape = false;
    for (auto ni : inputs) {
      is_zero_shape = is_zero_shape || node_arg_is_zero_shape(*ni.node_arg) ||
                      node_arg_is_unknown_shape(*ni.node_arg);
      // Similiar to zero shape, i.e. scalar, XIR does not support unknown shape
      // so that we treat unknown shape as same as zero shape.
    }
    if (is_zero_shape) {
      LOG(WARNING) << "cancel xir conversion, zero shape(scalar) found: "
                   << node_as_string(node);
      return false;
    }
    return true;
  });
  return *this;
}
void ToXirRule::check_xir_support_data_type() {
  this->actions_.push_back([=](ToXirRule* self, const Graph& /*graph*/,
                               const Node& node,
                               NodeAttributesBuilder& attrs) -> bool {
    auto outputs = node_get_output_node_args(node);
    if (outputs.size() != 1) {
      LOG(WARNING) << "cancel xir conversion, xir not support multiple "
                      "outputs node : "
                   << node_as_string(node);
      return false;
    }
    auto ret = true;
    auto output = outputs[0];
    auto data_type = node_arg_get_element_type(*output);
    // xir only support int8 and float32 , The addition of uint8 support is
    // because of need support U8S8 onnx models, we need convert
    // QuantierLinear/DequantizeLinear op to float2fix/fix2float. The previous
    // solution involved adding a pass, combining QuantizerLinear(uint8) +
    // fix2float -> fix. Now there is a need to support models with int8/uint8
    // inputs and outputs. The fundamental reason lies in the quantization tool
    // performing rather intricate operations specifically for U8S8.
    // for int32 : test case A16W8 mmjbka : input(int32) -> unsqueeze (int32)
    if (data_type == onnx::TensorProto_DataType_INT8 ||
        data_type == onnx::TensorProto_DataType_UINT8 ||
        data_type == onnx::TensorProto_DataType_FLOAT ||
        data_type == onnx::TensorProto_DataType_INT32) {
      // do nothing
    } else {
      LOG(WARNING) << "cancel xir conversion, xir not support data type: "
                   << data_type << " node : " << node_as_string(node);
      ret = false;
    }

    return ret;
  });
}
void ToXirRule::check_dynamic_shape() {
  this->actions_.push_back([=](ToXirRule* self, const Graph& /*graph*/,
                               const Node& node,
                               NodeAttributesBuilder& attrs) -> bool {
    // test case: /home/public/bevdet/LS_int.onnx
    // check input must not be a scalar, xilinx op does not support scalar.
    auto inputs = node_get_inputs(node);

    bool is_unknown_shape = false;
    bool is_dynamic_shape = false;
    for (auto ni : inputs) {
      if (!node_arg_exists(*ni.node_arg)) {
        continue;
      }
      is_unknown_shape =
          is_unknown_shape || node_arg_is_unknown_shape(*ni.node_arg);
      is_dynamic_shape =
          is_dynamic_shape || node_arg_is_dynamic_shape(*ni.node_arg);
    }
    auto outputs = node_get_output_node_args(node);
    for (auto output : outputs) {
      is_unknown_shape = is_unknown_shape || node_arg_is_unknown_shape(*output);
      is_dynamic_shape = is_dynamic_shape || node_arg_is_dynamic_shape(*output);
    }
    if (is_unknown_shape) {
      LOG(WARNING) << "cancel xir conversion, unknown shape found: "
                   << node_as_string(node);
      return false;
    }
    if (is_dynamic_shape) {
      LOG(WARNING) << "cancel xir conversion, dynamic shape found: "
                   << node_as_string(node);
      return false;
    }
    if (outputs.size() > 1) {
      LOG(WARNING)
          << "cancel xir conversion, multiple outputs are not supported"
          << node_as_string(node);
      return false;
    }
    return true;
  });
}
ToXirRule& ToXirRule::rename_and_convert_HW_to_WH(const std::string from,
                                                  const std::string to) {
  this->actions_.push_back([=](ToXirRule* self, const Graph& /*graph*/,
                               const Node& node,
                               NodeAttributesBuilder& attrs) -> bool {
    auto has_from_attr = node_has_attr(node, from);
    if (!has_from_attr) {
      // some attributes might be optional
      return true;
    }
    auto origin_attr = node_get_attr_ints(node, from);
    if (origin_attr.size() == 2) {
      auto filp_attr = {origin_attr[1], origin_attr[0]};
      attrs.add(to, filp_attr);
    } else {
      LOG(WARNING) << "'" << from << "' only support 2-dims convert"
                   << "cancel xir conversion. " << node_as_string(node);
      return false;
    }
    return true;
  });
  return *this;
}

ToXirRule& ToXirRule::add_attr_s(const std::string name,
                                 const std::string value) {
  this->actions_.push_back([=](ToXirRule* self, const Graph& /*graph*/,
                               const Node& /*node*/,
                               NodeAttributesBuilder& attrs) -> bool {
    attrs.add(name, value);
    return true;
  });
  return *this;
}

ToXirRule& ToXirRule::add_attr_i(const std::string name, const int64_t value) {
  this->actions_.push_back([=](ToXirRule* self, const Graph& /*graph*/,
                               const Node& /*node*/,
                               NodeAttributesBuilder& attrs) -> bool {
    attrs.add(name, value);
    return true;
  });
  return *this;
}

ToXirRule& ToXirRule::constant_input_arg_to_attr(size_t arg_index,
                                                 const std::string& attr_name,
                                                 const std::string& fun) {
  return action([=](ToXirRule* self, const Graph& graph, const Node& node,
                    NodeAttributesBuilder& attrs) -> bool {
    auto inputs = node_get_inputs(node);
    self->remove_input(arg_index);
    if (arg_index < inputs.size()) {
      auto input_arg = inputs[arg_index];
      // we assume all constant initializer are converted to Constant
      // op and then convert to xilinx:const op.
      if (input_arg.node != nullptr &&
          node_op_type(*input_arg.node) == "const") {
        if (fun == "to_ints") {
          auto ints = self->pass_.get_const_data<int64_t>(*input_arg.node);
          auto attr_data = std::vector<int64_t>(ints.begin(), ints.end());
          attrs.add(attr_name, attr_data);

        } else {
          LOG(FATAL) << "unknown function " << fun;
        }
      }
    }
    return true;
  });
}

ToXirRule& ToXirRule::set_shape_1_for_scalar() {
  return action([=](ToXirRule* self, const Graph& graph, const Node& node,
                    NodeAttributesBuilder& attrs) -> bool {
    auto shape = node_get_output_shape(node, 0);
    if (shape.empty()) {
      attrs.add("shape", std::vector<int64_t>{1});
    }
    return true;
  });
}
static ToXirRule::action_t convert_quant_dequant() {
  return [](ToXirRule* self, const Graph& graph, const Node& node,
            NodeAttributesBuilder& attrs) -> bool {
    if (self->pass_.get_context()
            ->get_provider_option("xlnx_enable_old_qdq")
            .value() != "1") {
      return false;
    }
    auto builder = PatternBuilder();
    std::shared_ptr<Pattern> pat_input = builder.wildcard();
    std::shared_ptr<Pattern> pat_scale = builder.xir_const_op();
    // zero_point maybe xir_const_op or node_arg
    // for support U8S8 , zero_point data_type is uint8, so not const folding,
    // and not create const op, so zero_point still is a node_arg
    std::shared_ptr<Pattern> pat_zero_point = builder.wildcard();
    std::shared_ptr<Pattern> pat_quantizer = builder.node3(
        node_op_type(node), {pat_input, pat_scale, pat_zero_point},
        {false, false, true});
    auto binder = pat_quantizer->match(graph, node);
    CHECK(binder != nullptr) << node_as_string(node);
    // trigger this bug with bevdet, model path: ci bevdet
    auto ni_input = (*binder)[pat_input->get_id()];
    bool is_unknown_shape = node_arg_is_unknown_shape(*ni_input.node_arg);
    bool is_scalar = node_arg_is_scalar(*ni_input.node_arg);
    bool is_dynamic_shape = node_arg_is_dynamic_shape(*ni_input.node_arg);
    if (is_unknown_shape || is_scalar || is_dynamic_shape) {
      LOG(WARNING) << "cancel xir conversion, unknown shape or scalar "
                      "input or dynamic shape found: "
                   << node_as_string(node);
      return false;
    }
    auto ni_scale = (*binder)[pat_scale->get_id()];
    auto scale_val = 0.0f;
    scale_val = self->pass_.get_const_data<float>(*ni_scale.node)[0];
    auto fix_point_p = scale_to_fix_point(scale_val);
    if (fix_point_p.get() == nullptr) {
      return false;
    }
    auto fix_point = *fix_point_p;
    auto ni_zero_point = (*binder)[pat_zero_point->get_id()];
    if (ni_zero_point.node_arg != nullptr) {
      auto data_type = node_arg_get_element_type(*ni_zero_point.node_arg);
      if (data_type == onnx::TensorProto_DataType_INT8) {
        // if zero_point data_type is int8, it should be convert to xir_const_op
        CHECK(ni_zero_point.node != nullptr &&
              node_op_type(*ni_zero_point.node) == "const")
            << "int8 zero_point should be convert to xir_const_op node.";
        auto const_data =
            self->pass_.get_const_data<int8_t>(*ni_zero_point.node)[0];
        if (const_data != 0) {
          LOG(WARNING)
              << "cancel xir conversion ,  int8 data_type only support "
                 "zero_point is 0. this zero_point is: "
              << std::to_string(const_data) << ", " << node_as_string(node);
          return false;
        }
      } else if (data_type == onnx::TensorProto_DataType_UINT8) {
        // if zero_point data_type is uint8, it shouldn't convert to
        // xir_const_op, it will be Initializer node_arg
        auto& tensor =
            node_arg_get_const_data_as_tensor(graph, *ni_zero_point.node_arg);
        auto const_data = tensor_proto_as_u8(graph, tensor);
        if (const_data != 128u) {
          LOG(WARNING)
              << "cancel xir conversion ,  uint8 data_type only support "
                 "zero_point is 128. this zero_point is: "
              << std::to_string(const_data) << ", " << node_as_string(node);
          return false;
        }
      } else {
        LOG(FATAL) << "TODO... " << node_as_string(node);
      }
    }
    auto shape = node_get_output_shape(node, 0);
    if (shape.empty()) {
      LOG(WARNING) << "cancel xir conversion, output shape empty"
                   << node_as_string(node);
      return false;
    }
    auto round_mode = "DPU_ROUND";
    if (self->pass_.get_context()
            ->get_provider_option("xlnx_enable_py3_round")
            .value() == "1") {
      round_mode = "PY3_ROUND";
    }
    attrs.add("fix_point", static_cast<int64_t>(fix_point))
        .add("bit_width", static_cast<int64_t>(8))
        .add("if_signed", static_cast<int64_t>(1))
        .add("round_mode", round_mode);
    return true;
  };
}
static ToXirRule::action_t convert_qdq() {
  return [](ToXirRule* self, const Graph& graph, const Node& node,
            NodeAttributesBuilder& attrs) -> bool {
    if (self->pass_.get_context()
            ->get_provider_option("xlnx_enable_old_qdq")
            .value() == "1") {
      return false;
    }
    auto round_mode = "DPU_ROUND";
    if (self->pass_.get_context()
            ->get_provider_option("xlnx_enable_py3_round")
            .value() == "1") {
      round_mode = "PY3_ROUND";
    }
    attrs.add("round_mode", round_mode);
    return true;
  };
}
static ToXirRule::action_t convert_fixneuron() {
  return [](ToXirRule* self, const Graph& graph, const Node& node,
            NodeAttributesBuilder& attrs) -> bool {
    auto builder = PatternBuilder();
    std::shared_ptr<Pattern> pat_input = builder.wildcard();
    std::shared_ptr<Pattern> pat_scale = builder.xir_const_op();
    std::shared_ptr<Pattern> pat_zero_point = builder.xir_const_op();
    auto op_type = node_op_type(node);
    auto op_domain = node_op_domain(node);
    auto op_full_name =
        op_domain.empty() ? op_type : (op_domain + ":" + op_type);

    std::shared_ptr<Pattern> pat_quantizer =
        builder.node3(op_full_name, {pat_input, pat_scale, pat_zero_point},
                      {false, false, true});
    auto binder = pat_quantizer->match(graph, node);
    CHECK(binder != nullptr) << node_as_string(node);
    auto ni_input = (*binder)[pat_input->get_id()];
    bool is_scalar = node_arg_is_scalar(*ni_input.node_arg);
    bool is_dynamic_shape = node_arg_is_dynamic_shape(*ni_input.node_arg);
    if (is_scalar || is_dynamic_shape) {
      LOG(WARNING)
          << "cancel xir conversion, scalar input or dynamic shape found:"
          << node_as_string(node);
      return false;
    }
    auto ni_scale = (*binder)[pat_scale->get_id()];
    auto scale_val = 0.0f;
    scale_val = self->pass_.get_const_data<float>(*ni_scale.node)[0];
    auto fix_point_p = scale_to_fix_point(scale_val);
    if (fix_point_p.get() == nullptr) {
      return false;
    }
    auto fix_point = *fix_point_p;
    auto shape = node_get_output_shape(node, 0);
    if (shape.empty()) {
      return false;
    }
    auto round_mode = "DPU_ROUND";
    if (self->pass_.get_context()
            ->get_provider_option("xlnx_enable_py3_round")
            .value() == "1") {
      round_mode = "PY3_ROUND";
    }
    attrs.add("fix_point", static_cast<int64_t>(fix_point))
        .add("bit_width", static_cast<int64_t>(8))
        .add("if_signed", static_cast<int64_t>(1))
        .add("round_mode", round_mode);
    return true;
  };
}
static ToXirRule::action_t convert_instnorm(const char* onnx_op_type) {
  return [onnx_op_type](ToXirRule* self, const Graph& graph, const Node& node,
                        NodeAttributesBuilder& attrs) -> bool {
    PatternBuilder b;
    std::shared_ptr<Pattern> pat_x = b.wildcard();
    std::shared_ptr<Pattern> pat_scale = b.wildcard();
    std::shared_ptr<Pattern> pat_B = b.wildcard();
    std::shared_ptr<Pattern> pat_instnorm =
        b.node2(onnx_op_type, {pat_x, pat_scale, pat_B});
    auto binder = pat_instnorm->match(graph, node);
    if (binder == nullptr) {
      LOG(WARNING) << "cancel xir conversion , " << node_as_string(node);
      return false;
    }
    auto x_node = (*binder)[pat_x->get_id()];
    std::vector<std::int64_t> input_shape;
    input_shape = *node_arg_get_shape_i64(*x_node.node_arg);
    if (input_shape.size() == 4) {
      self->set_xir_op_type("instancenorm_nchw");
    } else if (input_shape.size() == 3) {
      self->set_xir_op_type("instancenorm_ncd"); // testcase PST
    } else {
      LOG(WARNING) << "cancel xir ops conversion, dim!=4/3  is not support yet "
                      ",current dim is "
                   << input_shape.size() << ", op is " << node_as_string(node);
      return false;
    }
    if (!node_has_attr(node, "epsilon")) {
      LOG(WARNING) << "cancel xir ops conversion, need attr: epsilon , op is "
                   << node_as_string(node);
      return false;
    }
    attrs.add("affine", static_cast<int64_t>(0));
    return true;
  };
}
// rename "pads" to "pad" , and convert to {left, right , top, bottom}
static ToXirRule::action_t convert_pad_attr() {
  return [](ToXirRule* self, const Graph& graph, const Node& node,
            NodeAttributesBuilder& attrs) -> bool {
    /**
     * onnx , pads {x1_begin, x2_begin,.., x1_end, x2_end,...}
     * xir , pad {left, right, top, bottom}
     *
     * onnx pads:
     * {x1_begin,x1_end} means H-dim pad
     * {x2_begin, x2_end} means W-dim pad
     *
     * xir pad:
     * {left, right} means W-dim pad
     * {top, bottom} means H-dim pad
     *
     * convert {x1_begin, x2_begin, x1_end, x2_end} to  {x2_begin,
     * x2_end, x1_begin, x1_end}
     */
    auto new_pads = std::vector<int64_t>();
    if (node_has_attr(node, "pads")) {
      auto pads = node_get_attr_ints(node, "pads");
      CHECK_EQ(pads.size(), 4) << "Only support pads is  4-dims.";
      new_pads = {pads[1], pads[3], pads[0], pads[2]};
    } else {
      new_pads = {0, 0, 0, 0};
    }
    attrs.add("pad", new_pads);
    return true;
  };
}
static ToXirRule::action_t convert_pad_mode_attr() {
  return [](ToXirRule* self, const Graph& graph, const Node& node,
            NodeAttributesBuilder& attrs) -> bool {
    auto auto_pad = node_get_attr_string_with_default(node, "auto_pad",
                                                      std::string("FLOOR"));
    auto pad_mode = std::string();
    if (auto_pad == std::string("SAME_UPPER")) {
      pad_mode = "SAME";
    } else if (auto_pad == std::string("SAME_LOWER")) {
      pad_mode = "SAME";
    } else if (auto_pad == std::string("VALID")) {
      pad_mode = "VALID";
    } else if (auto_pad == std::string("NOTSET")) {
      pad_mode = "FLOOR";
    } else {
      LOG(WARNING) << "cancel xir conversion, unknown pad_mode = " << pad_mode;
      return false;
    }
    attrs.add("pad_mode", pad_mode);
    return true;
  };
}
static ToXirRule::action_t convert_reduction(const char* onnx_op_type) {
  return [onnx_op_type](ToXirRule* self, const Graph& graph, const Node& node,
                        NodeAttributesBuilder& attrs) -> bool {
    std::vector<std::int64_t> input_shape;
    std::vector<std::int64_t> axes;

    PatternBuilder b;
    std::shared_ptr<Pattern> pat_data = b.wildcard();
    std::shared_ptr<Pattern> pat_axes = b.xir_const_op();
    std::shared_ptr<Pattern> pat_reduction =
        b.node3(onnx_op_type, {pat_data, pat_axes}, {false, true});
    auto binder = pat_reduction->match(graph, node);
    if (binder == nullptr) {
      LOG(WARNING) << "cancel xir conversion , " << node_as_string(node);
      return false;
    }
    auto axes_node = (*binder)[pat_axes->get_id()];
    auto data_node = (*binder)[pat_data->get_id()];
    input_shape = *node_arg_get_shape_i64(*data_node.node_arg);
    auto noop_with_empty_axes =
        node_get_attr_int_with_default(node, "noop_with_empty_axes", 0);

    if (axes_node.node != nullptr) {
      auto ints = self->pass_.get_const_data<int64_t>(*axes_node.node);
      axes = std::vector<int64_t>(ints.begin(), ints.end());
      self->remove_input(1);
    } else if (node_has_attr(node, "axes")) {
      auto span = node_get_attr_ints(node, "axes");
      axes = std::vector<std::int64_t>(span.begin(), span.end());
    } else if (noop_with_empty_axes == 0) {
      axes = std::vector<std::int64_t>(input_shape.size());
      std::iota(axes.begin(), axes.end(), 0);
    } else {
      axes = std::vector<std::int64_t>{};
    }

    auto axis = std::vector<int64_t>();
    std::transform(axes.begin(), axes.end(), std::back_inserter(axis),
                   [input_shape](std::int64_t data) {
                     return (data + input_shape.size()) % input_shape.size();
                   });
    attrs.add("axis", axis);
    return true;
  };
}
static std::vector<int64_t>
gen_shape_attr_with_reshape(std::vector<int64_t> input_shape,
                            std::vector<int64_t> data_attr) {
  auto shape = std::vector<int64_t>();
  shape.reserve(data_attr.size());
  int64_t element_size = 1;
  for (auto dim : input_shape) {
    element_size *= dim;
  }
  auto automatic_dim = element_size;
  auto count_minus_1 = 0;
  auto i = 0;
  for (auto dim : data_attr) {
    if (dim == -1) {
      count_minus_1++;
    } else if (dim == 0) {
      automatic_dim = automatic_dim / input_shape[i];
    } else {
      automatic_dim = automatic_dim / dim;
    }
    i++;
  }
  CHECK_LE(count_minus_1, 1)
      << "invalid shape, contains multiple -1 , unable to deduce shape.";
  i = 0;
  for (auto dim : data_attr) {
    if (dim == -1) {
      shape.push_back(automatic_dim);
    } else if (dim == 0) {
      shape.push_back(input_shape[i]);
    } else {
      shape.push_back(dim);
    }
    i++;
  }
  return shape;
}

void to_xir_ops_pass(IPass& pass, Graph& graph) {
  std::vector<std::unique_ptr<ToXirRule>> rules;
  rules.push_back(
      std::make_unique<ToXirRule>("QuantizeLinear", "float2fix", pass));
  (*rules.back())
      .remove_input(1)
      .remove_input(2)
      .action(convert_quant_dequant());
  rules.push_back(
      std::make_unique<ToXirRule>("DequantizeLinear", "fix2float", pass));
  (*rules.back())
      .remove_input(1)
      .remove_input(2)
      .action(convert_quant_dequant());
  rules.push_back(
      std::make_unique<ToXirRule>("QuantizeLinear", "quantize_linear", pass));
  (*rules.back()).action(convert_qdq());
  rules.push_back(std::make_unique<ToXirRule>("DequantizeLinear",
                                              "dequantize_linear", pass));
  (*rules.back()).action(convert_qdq());
  rules.push_back(std::make_unique<ToXirRule>("com.microsoft:QuantizeLinear",
                                              "quantize_linear", pass));
  (*rules.back()).action(convert_qdq());
  rules.push_back(std::make_unique<ToXirRule>("com.microsoft:DequantizeLinear",
                                              "dequantize_linear", pass));
  (*rules.back()).action(convert_qdq());
  rules.push_back(std::make_unique<ToXirRule>(
      "com.vai.quantize:VitisQuantizeLinear", "quantize_linear", pass));
  (*rules.back()).action(convert_qdq());
  rules.push_back(std::make_unique<ToXirRule>(
      "com.vai.quantize:VitisDequantizeLinear", "dequantize_linear", pass));
  (*rules.back()).action(convert_qdq());
  rules.push_back(
      std::make_unique<ToXirRule>("ai.onnx.contrib:FixNeuron", "fix", pass));
  (*rules.back()).remove_input(1).remove_input(2).action(convert_fixneuron());
  rules.push_back(std::make_unique<ToXirRule>("Relu", "relu", pass));
  (*rules.back()).check_scalar();
  rules.push_back(std::make_unique<ToXirRule>("Round", "round", pass));
  (*rules.back()).check_scalar();
  rules.push_back(std::make_unique<ToXirRule>("Equal", "equal", pass));
  (*rules.back()).check_scalar();
  rules.push_back(std::make_unique<ToXirRule>("ArgMax", "argmax", pass));
  (*rules.back())
      .rename("axis", "axis")
      .rename("keepdims", "keepdims")
      .check_scalar();
  rules.push_back(
      std::make_unique<ToXirRule>("Reciprocal", "reciprocal", pass));
  (*rules.back()).check_scalar();
  rules.push_back(std::make_unique<ToXirRule>("Clip", "clamp", pass));
  (*rules.back())
      .remove_input(1)
      .remove_input(2)
      .action([](ToXirRule* self, const Graph& graph, const Node& node,
                 NodeAttributesBuilder& attrs) -> bool {
        PatternBuilder b;
        std::shared_ptr<Pattern> pat_input = b.wildcard();
        std::shared_ptr<Pattern> pat_min = b.xir_const_op();
        std::shared_ptr<Pattern> pat_max = b.xir_const_op();
        std::shared_ptr<Pattern> pat_clip =
            b.node3("Clip", {pat_input, pat_min, pat_max}, {false, true, true});
        auto binder = pat_clip->match(graph, node);
        CHECK(binder != nullptr)
            << "only support 1 input and 2 optional inputs. "
            << node_as_string(node);
        // For onnx Clip : The interval is specified by the inputs 'min' and
        // 'max'. They default to numeric_limits::lowest() and
        // numeric_limits::max(), respectively.
        // For xir clamp : min and max data type is  int32_t
        // so only support covert xir when min&max convert to int32_t without
        // loss of precision and can not support default value.
        auto ni_min = (*binder)[pat_min->get_id()];
        auto ni_max = (*binder)[pat_max->get_id()];
        if (ni_min.node_arg && ni_max.node_arg) {
          auto min = self->pass_.get_const_data<float>(*ni_min.node)[0];
          auto max = self->pass_.get_const_data<float>(*ni_max.node)[0];
          if (min == 0.0f && max == 6.0f) {
            self->set_xir_op_type("relu6");
          } else {
            self->set_xir_op_type("clamp");
            attrs.add("min", min);
            attrs.add("max", max);
          }
        } else {
          LOG(WARNING) << "cancel xir conversion ,  not support clip min&max "
                          "default value "
                       << node_as_string(node);
          return false;
        }
        return true;
      });
  rules.push_back(
      std::make_unique<ToXirRule>("com.ms.internal.nhwc:Conv", "conv2d", pass));
  (*rules.back())
      .add_attr_s("pad_mode", std::string("FLOOR"))
      .rename_and_convert_HW_to_WH("kernel_shape", "kernel")
      .rename_and_convert_HW_to_WH("strides", "stride")
      .rename_and_convert_HW_to_WH("dilations", "dilation")
      .action(convert_pad_attr())
      .action([](ToXirRule* self, const Graph& graph, const Node& node,
                 NodeAttributesBuilder& attrs) -> bool {
        // group!=1 && group == in_channels, conv to depthwise-conv2d,
        // else group = 1 or in_channels % group == 0 && out_channels %
        // group == 0 && weight_channels == in_channels / group, conv to
        // conv2d.
        auto output_shape = node_get_output_shape(node, 0);
        if (output_shape.size() != 4) {
          LOG(WARNING) << "cancel xir conversion,  conv2d output shape is "
                          "not equals 4 . "
                       << node_as_string(node);
          return false;
        }
        auto group = node_get_attr_int(node, "group");
        if (group == 1) {
          self->set_xir_op_type("conv2d");
          return true;
        }
        LOG_IF(INFO, ENV_PARAM(DEBUG_TO_XIR_PASS))
            << "convert conv to depthwise-conv2d : group " << group
            << ", node name " << VAIP_ORT_API(node_get_name)(node);

        // Match conv(*, fix(const),fix(const))
        PatternBuilder b;
        std::shared_ptr<Pattern> pat_input = b.wildcard();
        std::shared_ptr<Pattern> pat_weights = b.xir_const_op();
        std::shared_ptr<Pattern> pat_bias = b.xir_const_op();
        std::shared_ptr<Pattern> pat_conv =
            b.node3("com.ms.internal.nhwc:Conv",
                    {pat_input, pat_weights, pat_bias}, {false, false, true});
        auto binder = pat_conv->match(graph, node);
        if (binder == nullptr) {
          LOG(WARNING) << "cancel xir conversion , " << node_as_string(node);
          return false;
        }

        auto input = (*binder)[pat_input->get_id()];
        auto conv = (*binder)[pat_conv->get_id()];
        auto weights = (*binder)[pat_weights->get_id()];
        auto input_shape = node_get_output_shape(*input.node, 0);
        auto weights_shape = node_get_output_shape(*weights.node, 0);
        if (input_shape.size() != 4) {
          LOG(WARNING) << "cancel xir conversion, conv2d input shape is "
                          "not equals 4 . "
                       << node_as_string(node);
          return false;
        }
        if (weights_shape.size() != 4) {
          LOG(WARNING) << "cancel xir conversion, conv2d weights shape is "
                          "not equals 4 . "
                       << node_as_string(node);
          return false;
        }

        auto in_channels = input_shape[3];       // NHWC
        auto weight_channels = weights_shape[3]; // NHWC
        auto conv_shape = node_arg_get_shape_i64(*conv.node_arg);
        CHECK(conv_shape != nullptr)
            << node_arg_as_string(*conv.node_arg) << " shape absent";

        auto out_channels = (*conv_shape)[3]; // NHWC

        if (in_channels != group) {
          // test case: model # RetinaNet
          if (in_channels % group == 0 && out_channels % group == 0 &&
              weight_channels == in_channels / group) {
            self->set_xir_op_type("conv2d");
            attrs.add("group", group);
            return true;
          } else {
            LOG(WARNING) << "cancel xir conversion ,  conv2d only support "
                            "group==1 or in_channels == group or in_channels % "
                            "group == 0 && out_channels % group == 0 && "
                            "weight_channels == in_channels / group."
                         << node_as_string(node);
            return false;
          }
        }

        // channel_mutiplier = (int64_t)(in_channels / group);
        // because in_channels == group, so channel_mutiplier == 1
        // from shape info out_channels = channel_mutiplier * in_channles,
        // so out_channels == in_channles is true
        CHECK_EQ(in_channels, out_channels);

        self->set_xir_op_type("depthwise_conv2d_ihwo");
        // CONV convert to com.ms.internal.nhwc:Conv had convert weights
        // from (M, Cin/group, H, W) to (M, H, W, Cin/group),
        // com.ms.internal.nhwc:Conv convert to depthwise_conv2d weights
        // need layout transform {3, 1, 2, 0}, we can compared
        // to the next ToXirRule CONV convert to depthwise_conv2d_nchw
        // directly. {0, 2, 3, 1} + {3, 1, 2, 0} = {1, 2, 3, 0}. do this in
        // vaip_pass_layout_transform_via_adding_transpose/src/pass_main.cpp

        /**
         onnx Conv weights shape : (M, Cin/group , kH, kW)
         M is out channels;

         xir depthwise_conv weigths shape :
         (channel_mutiplier, Cin, kH, kW)

         channel_mutiplier = Cin/group
         Cout = channel_mutiplier * Cin
         becasue Cin == group
         so channel_mutiplier == 1  is true
         so Cin == Cout is true
        */
        return true;
      });
  rules.push_back(std::make_unique<ToXirRule>("Conv", "conv2d_nchw", pass));
  (*rules.back())
      .action(convert_pad_mode_attr())
      .rename_and_convert_HW_to_WH("kernel_shape", "kernel")
      .rename_and_convert_HW_to_WH("strides", "stride")
      .rename_and_convert_HW_to_WH("dilations", "dilation")
      .action(convert_pad_attr())
      .action([](ToXirRule* self, const Graph& graph, const Node& node,
                 NodeAttributesBuilder& attrs) -> bool {
        // group!=1 && group == in_channels, conv to depthwise-conv2d,
        // else group = 1 or in_channels % group == 0 && out_channels %
        // group == 0 && weight_channels == in_channels / group, conv to
        // conv2d.
        auto output_shape = node_get_output_shape(node, 0);
        if (output_shape.size() != 4) {
          LOG(WARNING) << "cancel xir conversion ,  conv2d output shape is "
                          "not equals 4 . "
                       << node_as_string(node);
          return false;
        }
        auto group = node_get_attr_int(node, "group");
        if (group == 1) {
          self->set_xir_op_type("conv2d_nchw");
          return true;
        }
        LOG_IF(INFO, ENV_PARAM(DEBUG_TO_XIR_PASS))
            << "convert conv to depthwise-conv2d_nchw : group " << group
            << ", node name " << VAIP_ORT_API(node_get_name)(node);

        // Match conv(*, fix(const),fix(const))
        PatternBuilder b;
        std::shared_ptr<Pattern> pat_input = b.wildcard();
        std::shared_ptr<Pattern> pat_weights =
            b.wildcard(); // change b.xir_const_op() to b.wildcard() for support
                          // xir QDQ op
        std::shared_ptr<Pattern> pat_bias =
            b.wildcard(); // change b.xir_const_op() to b.wildcard() for support
                          // xir QDQ op
        std::shared_ptr<Pattern> pat_conv = b.node3(
            "Conv", {pat_input, pat_weights, pat_bias}, {false, false, true});
        auto binder = pat_conv->match(graph, node);
        if (binder == nullptr) {
          LOG(WARNING) << "cancel xir conversion , " << node_as_string(node);
          return false;
        }

        auto input = (*binder)[pat_input->get_id()];
        auto conv = (*binder)[pat_conv->get_id()];
        auto weights = (*binder)[pat_weights->get_id()];
        auto input_shape = node_get_output_shape(*input.node, 0);
        auto weights_shape = node_get_output_shape(*weights.node, 0);
        if (input_shape.size() != 4) {
          LOG(WARNING) << "cancel xir conversion, conv2d input shape is "
                          "not equals 4 . "
                       << node_as_string(node);
          return false;
        }
        if (weights_shape.size() != 4) {
          LOG(WARNING) << "cancel xir conversion, conv2d weights shape is "
                          "not equals 4 . "
                       << node_as_string(node);
          return false;
        }

        auto in_channels = input_shape[1];       // NCHW
        auto weight_channels = weights_shape[1]; // NCHW
        auto conv_shape = node_arg_get_shape_i64(*conv.node_arg);
        CHECK(conv_shape != nullptr)
            << node_arg_as_string(*conv.node_arg) << " shape absent";

        auto out_channels = (*conv_shape)[1]; // NCHW

        if (in_channels != group ||
            in_channels !=
                out_channels) { // in_channels == out_channels && in_channels ==
                                // group convert to depthwise_conv2d
          // test case: model # RetinaNet
          if (in_channels % group == 0 && out_channels % group == 0 &&
              weight_channels == in_channels / group) {
            self->set_xir_op_type("conv2d_nchw");
            attrs.add("group", group);
            return true;
          } else {
            LOG(WARNING) << "cancel xir conversion, conv2d only support "
                            "group==1 or in_channels == group or in_channels % "
                            "group == 0 && out_channels % group == 0 && "
                            "weight_channels == in_channels / group."
                         << node_as_string(node);
            return false;
          }
        }

        // channel_mutiplier = (int64_t)(in_channels / group);

        // because in_channels == group, so channel_mutiplier == 1

        // from shape info out_channels = channel_mutiplier * in_channles,
        // so out_channels == in_channles is true
        CHECK_EQ(in_channels, out_channels);

        self->set_xir_op_type("depthwise_conv2d_nchw");

        /**
         for dethwise_conv2d
         onnx Conv weights shape : (M, Cin/group , kH, kW)
         M is out channels;

         xir depthwise_conv weigths shape :
         (channel_mutiplier, Cin, kH, kW)

         channel_mutiplier = Cin/group
         Cout = channel_mutiplier * Cin
         becasue Cin == group
         so channel_mutiplier == 1  is true
         so Cin == Cout is true
        */
        // Note : SKIP data layout transform , because channel_mutiplier
        // == 1, data layout transform do not change the arrangement of
        // data in memory. const data shape is old shape.

        return true;
      });
  rules.push_back(std::make_unique<ToXirRule>("Conv", "conv1d_ncd", pass));
  // test case: issue #1143
  (*rules.back())
      .action(convert_pad_mode_attr())
      .rename("kernel_shape", "kernel")
      .rename("strides", "stride")
      .rename("dilations", "dilation")
      .rename("pads", "pad") // pads size = 2, so don't need convert.
      .action([](ToXirRule* self, const Graph& graph, const Node& node,
                 NodeAttributesBuilder& attrs) -> bool {
        if (!ENV_PARAM(XLNX_ENABLE_CONV1D)) {
          LOG(WARNING) << "cancel xir conversion to conv1d : "
                       << node_as_string(node);
          return false;
        }
        // group!=1 && group == in_channels, conv to depthwise-conv1d,
        // else group = 1 or in_channels % group == 0 && out_channels %
        // group == 0 && weight_channels == in_channels / group, conv to
        // conv1d.
        auto output_shape = node_get_output_shape(node, 0);
        if (output_shape.size() != 3) {
          LOG(WARNING) << "cancel xir conversion,  conv1d output shape is "
                          "not equals 3 . "
                       << node_as_string(node);
          return false;
        }
        auto group = node_get_attr_int(node, "group");
        if (group == 1) {
          self->set_xir_op_type("conv1d_ncd");
          return true;
        }

        // LOG(WARNING) << "TODO: support group conv1d and depthwise-conv1d";
        // return false;

        LOG_IF(INFO, ENV_PARAM(DEBUG_TO_XIR_PASS))
            << "convert conv to depthwise-conv1d_ncd : group " << group
            << ", node name " << VAIP_ORT_API(node_get_name)(node);

        // Match conv(*, fix(const),fix(const))
        PatternBuilder b;
        std::shared_ptr<Pattern> pat_input = b.wildcard();
        std::shared_ptr<Pattern> pat_weights = b.wildcard();
        std::shared_ptr<Pattern> pat_bias = b.wildcard();
        std::shared_ptr<Pattern> pat_conv = b.node3(
            "Conv", {pat_input, pat_weights, pat_bias}, {false, false, true});
        auto binder = pat_conv->match(graph, node);
        if (binder == nullptr) {
          LOG(WARNING) << "cancel xir conversion , " << node_as_string(node);
          return false;
        }

        auto input = (*binder)[pat_input->get_id()];
        auto conv = (*binder)[pat_conv->get_id()];
        auto weights = (*binder)[pat_weights->get_id()];
        auto input_shape = node_get_output_shape(*input.node, 0);
        auto weights_shape = node_get_output_shape(*weights.node, 0);
        if (input_shape.size() != 3) {
          LOG(WARNING) << "cancel xir conversion, conv1d input shape is "
                          "not equals 3 . "
                       << node_as_string(node);
          return false;
        }
        if (weights_shape.size() != 3) {
          LOG(WARNING) << "cancel xir conversion, conv1d weights shape is "
                          "not equals 3 . "
                       << node_as_string(node);
          return false;
        }

        auto in_channels = input_shape[1]; // NCD
        // auto weight_channels = weights_shape[1]; // OCD
        auto conv_shape = node_arg_get_shape_i64(*conv.node_arg);
        CHECK(conv_shape != nullptr)
            << node_arg_as_string(*conv.node_arg) << " shape absent";

        auto out_channels = (*conv_shape)[1]; // NCD

        if (group != in_channels || in_channels != out_channels) {
          LOG(WARNING) << "cancel xir conversion ,  conv1d only support "
                          "group==1 or group == in_channels == out_channels "
                       << node_as_string(node);
          return false;
        }

        /*if (in_channels != group ||
          in_channels !=
              out_channels) { // in_channels == out_channels && in_channels ==
                              // group convert to depthwise_conv2d
        // test case: model # RetinaNet
        if (in_channels % group == 0 && out_channels % group == 0 &&
            weight_channels == in_channels / group) {
          self->set_xir_op_type("conv1d_ncd");
          attrs.add("group", group);
          return true;
        } else {
          LOG(WARNING) << "cancel xir conversion ,  conv1d only support "
                          "group==1 or in_channels == group or in_channels % "
                          "group == 0 && out_channels % group == 0 && "
                          "weight_channels == in_channels / group."
                       << node_as_string(node);
          return false;
        }
        }
        // auto channel_mutiplier = (int64_t)(in_channels / group);

        // because in_channels == group, so channel_mutiplier == 1

        // from shape info out_channels = channel_mutiplier * in_channles,
        // so out_channels == in_channles is true
        // CHECK_EQ(in_channels, out_channels);
        */
        self->set_xir_op_type("depthwise_conv1d_ncd");
        return true;
      });
  rules.push_back(std::make_unique<ToXirRule>(
      ToXirRule("com.ms.internal.nhwc:MaxPool", "maxpool2d", pass)));
  (*rules.back())
      .check_scalar()
      .rename_and_convert_HW_to_WH("kernel_shape", "kernel")
      .rename_and_convert_HW_to_WH("strides", "stride")
      .action(convert_pad_attr())
      .add_attr_i("global", 0) // TODO
      .action([](ToXirRule* self, const Graph&, const Node& node,
                 NodeAttributesBuilder& attrs) -> bool {
        auto ceil_mode = node_get_attr_int_with_default(node, "ceil_mode", 0);
        auto pad_mode = std::string();
        if (ceil_mode == 0) {
          pad_mode = "FLOOR";
        } else if (ceil_mode == 1) {
          pad_mode = "CEIL";
        } else {
          LOG(FATAL) << "unknown ceil_mode = " << ceil_mode;
        }
        attrs.add("pad_mode", pad_mode);
        return true;
      });
  rules.push_back(std::make_unique<ToXirRule>(
      ToXirRule("MaxPool", "maxpool2d_nchw", pass)));
  (*rules.back())
      .check_scalar()
      .rename_and_convert_HW_to_WH("kernel_shape", "kernel")
      .rename_and_convert_HW_to_WH("strides", "stride")
      .action(convert_pad_attr())
      .add_attr_i("global", 0) // TODO
      .action([](ToXirRule* self, const Graph&, const Node& node,
                 NodeAttributesBuilder& attrs) -> bool {
        auto ceil_mode = node_get_attr_int_with_default(node, "ceil_mode", 0);
        auto pad_mode = std::string();
        if (ceil_mode == 0) {
          pad_mode = "FLOOR";
        } else if (ceil_mode == 1) {
          pad_mode = "CEIL";
        } else {
          LOG(FATAL) << "unknown ceil_mode = " << ceil_mode;
        }
        attrs.add("pad_mode", pad_mode);
        return true;
      });
  // test case: /home/public/bevdet/LS_int.onnx
  // `add` does not support scalar output for xir.
  rules.push_back(std::make_unique<ToXirRule>("Add", "add", pass));
  (*rules.back()).check_scalar();
  // test case: /home/public/bevdet/LS_int.onnx
  // I guess `sub` is similiar to `add`
  rules.push_back(std::make_unique<ToXirRule>("Sub", "sub", pass));
  (*rules.back()).check_scalar();
  rules.push_back(std::make_unique<ToXirRule>(
      ToXirRule("com.ms.internal.nhwc:AveragePool", "avgpool2d", pass)));
  (*rules.back())
      .check_scalar()
      .rename_and_convert_HW_to_WH("kernel_shape", "kernel")
      .rename_and_convert_HW_to_WH("strides", "stride")
      .rename_and_convert_HW_to_WH("dilations", "dilation")
      .action(convert_pad_attr())
      .action([](ToXirRule* self, const Graph&, const Node& node,
                 NodeAttributesBuilder& attrs) -> bool {
        auto ceil_mode = node_get_attr_int_with_default(node, "ceil_mode", 0);
        auto pad_mode = std::string();
        if (ceil_mode == 0) {
          pad_mode = "FLOOR";
        } else if (ceil_mode == 1) {
          pad_mode = "CEIL";
        } else {
          LOG(FATAL) << "unknown ceil_mode = " << ceil_mode;
        }
        attrs.add("pad_mode", pad_mode);
        return true;
      });
  rules.push_back(
      std::make_unique<ToXirRule>("AveragePool", "avgpool2d_nchw", pass));
  (*rules.back())
      .check_scalar()
      .rename_and_convert_HW_to_WH("kernel_shape", "kernel")
      .rename_and_convert_HW_to_WH("strides", "stride")
      .rename_and_convert_HW_to_WH("dilations", "dilation")
      .action(convert_pad_attr())
      .action([](ToXirRule* self, const Graph&, const Node& node,
                 NodeAttributesBuilder& attrs) -> bool {
        auto ceil_mode = node_get_attr_int_with_default(node, "ceil_mode", 0);
        auto pad_mode = std::string();
        if (ceil_mode == 0) {
          pad_mode = "FLOOR";
        } else if (ceil_mode == 1) {
          pad_mode = "CEIL";
        } else {
          LOG(FATAL) << "unknown ceil_mode = " << ceil_mode;
        }
        attrs.add("pad_mode", pad_mode);
        return true;
      });

  rules.push_back(std::make_unique<ToXirRule>("Mul", "mul", pass));
  (*rules.back()).check_scalar(); // test case : model 18
  rules.push_back(std::make_unique<ToXirRule>("Div", "div", pass));
  (*rules.back()).check_scalar();
  rules.push_back(std::make_unique<ToXirRule>("Max", "max", pass));
  (*rules.back()).check_scalar();
  rules.push_back(std::make_unique<ToXirRule>("Min", "min", pass));
  (*rules.back()).check_scalar();
  rules.push_back(std::make_unique<ToXirRule>(
      "Abs", "abs", pass)); // testcase :  edgenext_small_rw
  (*rules.back()).check_scalar();
  rules.push_back(
      std::make_unique<ToXirRule>("GlobalAveragePool", "avgpool2d_nchw", pass));
  (*rules.back())
      .check_scalar()
      .add_attr_i("global", 1)
      .action([](ToXirRule* self, const Graph&, const Node& node,
                 NodeAttributesBuilder& attrs) -> bool {
        auto& input = *node_get_inputs(node)[0].node;
        auto input_shape = node_get_output_shape(input, 0);
        if (input_shape.size() != 4) {
          LOG(WARNING) << "cancel xir conversion ,  only support 4-dims "
                          "GloablAveragePool to avgpool2d "
                       << node_as_string(node);
          return false;
        }

        auto kernel = std::vector<int64_t>{input_shape[3], input_shape[2]};
        attrs.add("kernel", kernel)
            .add("stride", kernel)
            // TODO:caffe ->CEIL , pytroch -> FLOOR(default)
            // or CEIL , TF ->SAME  or VALID
            // TODO: count_include_invalid & count_include_pad
            // use default value "true"
            .add("pad_mode", "FLOOR");
        return true;
      });
  rules.push_back(std::make_unique<ToXirRule>(
      "com.xilinx:GlobalAveragePool_nhwc", "avgpool2d", pass));
  (*rules.back())
      .check_scalar()
      .add_attr_i("global", 1)
      .action([](ToXirRule* self, const Graph&, const Node& node,
                 NodeAttributesBuilder& attrs) -> bool {
        auto& input = *node_get_inputs(node)[0].node;
        auto input_shape = node_get_attr_ints(input, "shape");
        if (input_shape.size() != 4) {
          LOG(WARNING) << "cancel xir conversion ,  only "
                          "support 4-dims "
                          "GloablAveragePool to avgpool2d "
                       << node_as_string(node);
          return false;
        }

        auto kernel = std::vector<int64_t>{input_shape[2], input_shape[1]};
        attrs.add("kernel", kernel)
            .add("stride", kernel)
            // TODO:caffe ->CEIL , pytroch -> FLOOR(default)
            // or CEIL , TF ->SAME  or VALID
            // TODO: count_include_invalid & count_include_pad
            // use default value "true"
            .add("pad_mode", "FLOOR");
        return true;
      });

  rules.push_back(std::make_unique<ToXirRule>("Flatten", "reshape", pass));
  // test case levit_128.fb_dist_in1k
  // The implementation of xir may be wrong
  // node=@3055 [input.11:(ty=1,shape=[196,256])] com.xilinx::flatten
  // [onnx::Flatten_677:(ty=1,shape=[1,196,256])]
  // xir_op={type=flatten,name=input.11,shape=[1,196,256]}
  (*rules.back())
      .check_scalar()
      .action([](ToXirRule* self, const Graph& /*graph*/, const Node& node,
                 NodeAttributesBuilder& attrs) -> bool {
        auto axis = node_get_attr_int(node, "axis");
        auto& input = *node_get_inputs(node)[0].node;
        auto input_shape = node_get_output_shape(input, 0);
        auto input_shape_size = (long int)input_shape.size();
        if (axis >= input_shape_size) {
          LOG(WARNING) << "cancel xir conversion, axis " << axis
                       << " >= input_shape_size " << input_shape_size
                       << node_as_string(input);
          return false;
        }
        if (axis < 0)
          axis = input_shape_size + axis;
        auto shape = std::vector<int64_t>(2, 1);
        for (long int i = 0; i < axis; i++) {
          shape[0] *= input_shape[i];
        }
        for (long int i = (long int)axis; i < input_shape_size; i++) {
          shape[1] *= input_shape[i];
        }
        attrs.add("shape", shape);
        return true;
      });
  rules.push_back(
      std::make_unique<ToXirRule>("ReduceMean", "reduction_mean", pass));
  (*rules.back())
      .check_scalar() // test case 5021
                      //.rename("axes", "axis")
      .rename("keepdims", "keep_dims")
      .action(convert_reduction("ReduceMean"));
  rules.push_back(
      std::make_unique<ToXirRule>("ReduceMax", "reduction_max", pass));
  //.rename("axes", "axis")
  (*rules.back())
      .rename("keepdims", "keep_dims")
      .action(convert_reduction("ReduceMax"));
  rules.push_back(
      std::make_unique<ToXirRule>("ReduceMin", "reduction_min", pass));
  //.rename("axes", "axis")
  (*rules.back())
      .rename("keepdims", "keep_dims")
      .action(convert_reduction("ReduceMin"));
  rules.push_back(std::make_unique<ToXirRule>(
      "ReduceSum", "reduction_sum", pass)); // testcase: resnest14d fd12f1
  (*rules.back())
      .rename("keepdims", "keep_dims")
      .action(convert_reduction("ReduceSum"));
  rules.push_back(std::make_unique<ToXirRule>("Squeeze", "squeeze", pass));
  auto ptr = [](ToXirRule* self, const Node& node) {
    auto inputs = node_get_inputs(node);
    auto input_shape_ptr = node_arg_get_shape_i64(*inputs[0].node_arg);
    if (inputs.size() == 2) {
      auto input_arg = inputs[1];
      CHECK(input_arg.node != nullptr);
      CHECK_EQ(node_op_type(*input_arg.node), "const");
      auto axes = self->pass_.get_const_data<int64_t>(*input_arg.node);
      auto axis = std::vector<int64_t>(axes.begin(), axes.end());
      for (int i = 0; i < axis.size(); i++) {
        if (axis[i] < 0) {
          axis[i] = axis[i] + input_shape_ptr->size();
        }
      }
      return attr_proto_new_ints("axis", axis);
    }
    CHECK(input_shape_ptr != nullptr)
        << node_arg_as_string(*inputs[0].node_arg) << " shape absent";
    auto input_shape = *input_shape_ptr;
    auto axis = std::vector<int64_t>();
    for (auto i = 0u; i < input_shape.size(); ++i) {
      if (input_shape[i] == 1) {
        axis.push_back(i);
      }
    }
    return attr_proto_new_ints("axis", axis);
  };
  (*rules.back())
      .check_scalar() // model 14, Squeeze op ver=11
      .rename("axes", "axis", ptr)

      // test case
      // /home/public/xieyi/CopyMe/ipu_softmax/scene_detection/quantize_results_onnx_0802_2/quantized.onnx
      .remove_input(1);
  rules.push_back(std::make_unique<ToXirRule>("Unsqueeze", "reshape", pass));
  // test case: efficientnet-b4
  (*rules.back())
      .check_scalar()
      .action([](ToXirRule* self, const Graph& /*graph*/, const Node& node,
                 NodeAttributesBuilder& attrs) -> bool {
        auto inputs = node_get_inputs(node);
        auto output_shape = node_get_output_shape(node, 0);
        attrs.add("shape", output_shape);
        if (inputs.size() == 2) {
          auto input_arg = inputs[1];
          CHECK(input_arg.node != nullptr);
          CHECK_EQ(node_op_type(*input_arg.node), "const");
          self->remove_input(1);
        }
        return true;
      });
  rules.push_back(std::make_unique<ToXirRule>("Reshape", "reshape", pass));
  // test case: /home/public/bevdet/LS_int.onnx
  (*rules.back())
      .check_scalar()
      .check_node_input_zero_shape()
      .action([](ToXirRule* self, const Graph& graph, const Node& node,
                 NodeAttributesBuilder& attrs) -> bool {
        auto b = PatternBuilder();
        std::shared_ptr<Pattern> pat_shape = b.xir_const_op();
        std::shared_ptr<Pattern> pat_input = b.wildcard();
        std::shared_ptr<Pattern> pat_reshape =
            b.node2("Reshape", {pat_input, pat_shape});
        auto binder = pat_reshape->match(graph, node);
        if (binder != nullptr) {
          // remove shape input node , use attr
          // "shape"
          auto ni_input = (*binder)[pat_input->get_id()];
          auto ni_shape = (*binder)[pat_shape->get_id()];
          // fill in attr "shape", and origin "shape"
          // is high level
          if (!node_has_attr(node, "shape")) {
            auto data = self->pass_.get_const_data<int64_t>(*ni_shape.node);
            auto input_shape_ptr = node_arg_get_shape_i64(*ni_input.node_arg);
            CHECK(input_shape_ptr != nullptr)
                << node_arg_as_string(*ni_input.node_arg) << " shape absent";
            auto input_shape = *input_shape_ptr;
            auto shape = std::vector<int64_t>();
            shape.reserve(data.size());
            for (auto dim : data) {
              shape.push_back(dim);
            }
            shape = gen_shape_attr_with_reshape(input_shape, shape);
            attrs.add("shape", shape);
          }
          self->remove_input(1);
        }
        return true;
      });
  rules.push_back(std::make_unique<ToXirRule>("Gemm", "matmul", pass));
  (*rules.back())
      .rename("transA", "transpose_a")
      .rename("transB", "transpose_b")
      .action([](ToXirRule* self, const Graph& /*graph*/, const Node& node,
                 NodeAttributesBuilder& attrs) -> bool {
        auto output_shape = node_get_output_shape(node, 0);
        if (output_shape.empty()) {
          LOG(WARNING) << "cancel xir conversion ,  "
                          "conv2d output shape "
                          "must not empty "
                       << node_as_string(node);
          return false;
        }
        // auto alpha = node_attrs_get_float(attrs,
        // "alpha"); CHECK(alpha.has_value());
        // CHECK_EQ(alpha.value(), 1.0);
        // auto beta = node_attrs_get_float(attrs, "beta");
        // CHECK(beta.has_value());
        // CHECK_EQ(beta.value(), 1.0);
        return true;
      }),
      rules.push_back(
          std::make_unique<ToXirRule>("BatchNormalization", "batchnorm", pass));
  (*rules.back())
      .check_scalar()
      .rename("epsilon", "epsilon")
      .add_attr_i("axis", 1),
      rules.push_back(
          std::make_unique<ToXirRule>("Transpose", "transpose", pass));
  // test case: /home/public/bevdet/LS_int.onnx
  (*rules.back())
      .check_scalar()
      .rename("perm", "order",
              [](ToXirRule* self, const Node& node) {
                auto order = std::vector<int64_t>{};
                auto shape = node_get_output_shape(node, 0);
                for (auto i = 0u; i < shape.size(); ++i) {
                  auto value = (int64_t)shape.size() - i - 1;
                  order.push_back(value);
                }
                return attr_proto_new_ints("order", order);
              }),
      rules.push_back(std::make_unique<ToXirRule>("MatMul", "matmul", pass));
  (*rules.back())
      .check_scalar()
      .add_attr_i("transpose_a", 0)
      .add_attr_i("transpose_b", 0)
      .action([](ToXirRule* self, const Graph& graph, const Node& node,
                 NodeAttributesBuilder& attrs) -> bool {
        // https://jira.xilinx.com/browse/VAI-5228 MatMul dims=3 has mismatch
        // between CPU EP and IPU.
        if (ENV_PARAM(VAIP_SKIP_MATMUL_DIMS_NOT_EQUAL_4)) {
          auto output_shape = node_get_output_shape(node, 0);
          if (output_shape.size() != 4) {
            LOG(WARNING) << "cancel xir conversion, MatMul ouput shape is "
                            "not equals 4 . "
                         << node_as_string(node);
            return false;
          }
        }
        return true;
      });
  rules.push_back(std::make_unique<ToXirRule>("Concat", "concat", pass));
  // test case: /home/public/bevdet/LS_int.onnx
  (*rules.back())
      .check_scalar()
      .check_node_input_zero_shape()
      .action([](ToXirRule* self, const Graph& graph, const Node& node,
                 NodeAttributesBuilder& attrs) -> bool {
        if (ENV_PARAM(VAIP_ALIGNMENT_TO_ORT_SKIP_CONCAT)) {
          LOG(WARNING) << "cancel xir conversion, "
                          "VAIP_ALIGNMENT_TO_ORT_SKIP_CONCAT enabled";
          return false;
        }
        auto shape = node_get_output_shape(node, 0);
        auto axis = node_get_attr_int_with_default(node, "axis", 0);
        if (axis < 0) {
          axis = shape.size() + axis;
        }
        attrs.add("axis", axis);
        return true;
      });
  rules.push_back(std::make_unique<ToXirRule>("Sigmoid", "sigmoid", pass));
  (*rules.back()).check_scalar();
  rules.push_back(std::make_unique<ToXirRule>("ConvTranspose",
                                              "transposed_conv2d_nchw", pass));
  // .copy("group")
  (*rules.back())
      .rename_and_convert_HW_to_WH("kernel_shape", "kernel")
      .rename_and_convert_HW_to_WH("strides", "stride")
      .rename_and_convert_HW_to_WH("dilations", "dilation")
      .check_scalar()
      .action(convert_pad_attr())
      .action([](ToXirRule* self, const Graph& graph, const Node& node,
                 NodeAttributesBuilder& attrs) -> bool {
        if (!node_has_attr(node, "output_padding")) {
          return true;
        }
        auto output_paddings = node_get_attr_ints(node, "output_padding");
        bool supported = std::all_of(output_paddings.begin(),
                                     output_paddings.end(), [](int64_t n) {
                                       if (n == 0) {
                                         return true;
                                       } else {
                                         return false;
                                       }
                                     });
        if (!supported) {
          LOG(WARNING) << "cancel xir conversion, ConvTranspose with non-zero "
                          "output_padding not support now";
          return false;
        } else {
          return true;
        }
      })
      .add_attr_s("pad_mode", std::string("FLOOR"));
  rules.push_back(std::make_unique<ToXirRule>("com.xilinx:ConvTranspose_nhwc",
                                              "transposed_conv2d", pass));
  // .copy("group")
  (*rules.back())
      .rename_and_convert_HW_to_WH("kernel_shape", "kernel")
      .rename_and_convert_HW_to_WH("strides", "stride")
      .rename_and_convert_HW_to_WH("dilations", "dilation")
      .check_scalar()
      .action(convert_pad_attr())
      .add_attr_s("pad_mode", std::string("FLOOR")),
      rules.push_back(
          std::make_unique<ToXirRule>("LeakyRelu", "leaky_relu", pass));
  (*rules.back()).copy("alpha").check_scalar();
  rules.push_back(std::make_unique<ToXirRule>("Pad", "pad", pass));
  (*rules.back())
      .remove_input(1)
      .remove_input(2)
      .action([](ToXirRule* self, const Graph& graph, const Node& node,
                 NodeAttributesBuilder& attrs) -> bool {
        auto mode = node_get_attr_string_with_default(node, "mode",
                                                      std::string("constant"));
        auto xir_mode = std::string("CONSTANT");
        if (mode == "constant") {
          xir_mode = std::string("CONSTANT");
        } else if (mode == "reflect") {
          xir_mode = std::string("REFLECT");
        } else if (mode == "edge") {
          xir_mode = std::string("SYMMETRIC");
        } else {
          LOG(FATAL) << "unknown pad mode: " << mode;
        }
        auto b = PatternBuilder();
        std::shared_ptr<Pattern> pattern_input = b.wildcard();
        std::shared_ptr<Pattern> pattern_pads = b.xir_const_op();
        std::shared_ptr<Pattern> pattern_constant_value = b.xir_const_op();
        std::shared_ptr<Pattern> pattern_this = b.node3(
            "Pad", {pattern_input, pattern_pads, pattern_constant_value},
            {false, true, true});

        auto binder = pattern_this->match(graph, node);
        if (binder == nullptr)
          return false;
        auto pad_node = (*binder)[pattern_this->get_id()];
        auto pads_node = (*binder)[pattern_pads->get_id()];
        gsl::span<const int64_t> pads_data;
        if (pads_node.node != nullptr) {
          pads_data = self->pass_.get_const_data<int64_t>(*pads_node.node);
        } else {
          // since onnx opset version 11 , pads from
          // 'attribute' becomes 'input'
          pads_data = node_get_attr_ints(*pad_node.node, "pads");
        }
        auto size = pads_data.size();
        auto half_size = size / 2;
        auto paddings = std::vector<int64_t>(pads_data.size());
        for (auto i = 0u; i < half_size; ++i) {
          paddings[2 * i] = pads_data[i];
          paddings[2 * i + 1] = pads_data[half_size + i];
        }
        attrs.add("paddings", paddings);
        attrs.add("mode", xir_mode);
        if (mode == "constant") {
          auto constant_value_node =
              (*binder)[pattern_constant_value->get_id()];
          auto constant_value_data = 0.0f;
          if (constant_value_node.node != nullptr) {
            constant_value_data =
                self->pass_.get_const_data<float>(*constant_value_node.node)[0];
          }
          auto constant_values = std::vector<float>(size, constant_value_data);
          attrs.add("constant_values", constant_values);
        }
        return true;
      });
  rules.push_back(std::make_unique<ToXirRule>("Softmax", "softmax", pass));
  (*rules.back()).copy("axis").check_scalar();
  rules.push_back(/*test case:Iris_int.onnx，see #1003*/
                  std::make_unique<ToXirRule>("DepthToSpace",
                                              "pixel_shuffle_nchw", pass));
  (*rules.back())
      .action([](ToXirRule* self, const Graph& /*graph*/, const Node& node,
                 NodeAttributesBuilder& attrs) -> bool {
        auto blocksize = node_get_attr_int(node, "blocksize");
        if (node_has_attr(node, "mode") && // default value is DRC
            "CRD" == node_get_attr_string(node, "mode")) {
          self->set_xir_op_type("pixel_shuffle_nchw");
          attrs.add("upscale", static_cast<int64_t>(1));
          attrs.add("scale", static_cast<int64_t>(blocksize));
        } else { // test case: see #1287
          self->set_xir_op_type("gstiling_nchw");
          attrs.add("reverse", static_cast<int64_t>(1));
          attrs.add("stride", static_cast<int64_t>(blocksize));
        }
        return true;
      })
      .check_scalar();

  rules.push_back(/*test case:#1448*/
                  std::make_unique<ToXirRule>("SpaceToDepth",
                                              "space_to_depth_nchw", pass));
  (*rules.back())
      .action([](ToXirRule* self, const Graph& /*graph*/, const Node& node,
                 NodeAttributesBuilder& attrs) -> bool {
        auto block_size = node_get_attr_int(node, "blocksize");
        attrs.add("block_size", static_cast<int64_t>(block_size));
        return true;
      })
      .check_scalar();

  rules.push_back(/*test case:lcnet_100.ra2_in1k(0ee715e)*/
                  std::make_unique<ToXirRule>("HardSigmoid", "hard_sigmoid",
                                              pass));
  (*rules.back())
      .action([](ToXirRule* self, const Graph& /*graph*/, const Node& node,
                 NodeAttributesBuilder& attrs) -> bool {
        auto alpha = node_get_attr_float_with_default(node, "alpha", 0.2f);
        auto beta = node_get_attr_float_with_default(node, "beta", 0.5f);
        if (std::abs(alpha - float(1) / 6) >
                std::numeric_limits<float>::epsilon() ||
            std::abs(beta - 0.5f) > std::numeric_limits<float>::epsilon()) {
          LOG(WARNING) << "cancel xir conversion, HardSigmoid alpha = " << alpha
                       << ",beta = " << beta;
          return false;
        }
        return true;
      });
  rules.push_back(std::make_unique<ToXirRule>("Resize", "resize", pass));
  (*rules.back())
      .remove_input(1)
      .remove_input(2)
      .action([](ToXirRule* self, const Graph& graph, const Node& node,
                 NodeAttributesBuilder& attrs) -> bool {
        if (ENV_PARAM(VAIP_ALIGNMENT_TO_ORT_SKIP_RESIZE)) {
          LOG(WARNING) << "cancel xir conversion, "
                          "VAIP_ALIGNMENT_TO_ORT_SKIP_RESIZE enabled";
          return false;
        }
        // model 20
        auto& mode = node_get_attr_string(node, "mode");
        auto xir_resize_mode = "";
        if (mode == "linear") {
          xir_resize_mode = "BILINEAR";
        } else if (mode == "nearest") {
          xir_resize_mode = "NEAREST";
        } else {
          LOG(WARNING) << "cancel xir conversion, "
                       << " resize op model is " << mode
                       << node_as_string(node);
          return false;
        }

        auto xir_half_pixel_centers = 0;
        auto xir_align_corners = 0;
        std::string coordinate_transformation_mode =
            "half_pixel"; // default value for Resize
        if (node_has_attr(node, "coordinate_transformation_mode")) {
          coordinate_transformation_mode =
              node_get_attr_string(node, "coordinate_transformation_mode");
        }
        if (coordinate_transformation_mode == "pytorch_half_pixel" ||
            coordinate_transformation_mode == "half_pixel") {
          xir_half_pixel_centers = 1;
          xir_align_corners = 0;
        } else if (coordinate_transformation_mode == "align_corners") {
          xir_align_corners = 1;
          xir_half_pixel_centers = 0;
        } else if (coordinate_transformation_mode == "asymmetric") {
          xir_half_pixel_centers = 0;
          xir_align_corners = 0;
        } else if (coordinate_transformation_mode == "tf_half_pixel_for_nn") {
          // testcase :  issue 1239
          // In  ONNX opset=11
          // https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Resize-11
          // if coordinate_transformation_mode is "half_pixel",
          // x_original = (x_resized + 0.5) / scale - 0.5,
          // if coordinate_transformation_mode is "tf_half_pixel_for_nn",
          // x_original = (x_resized + 0.5) / scale

          // but In XIR ,
          // https://gitenterprise.xilinx.com/VitisAI/vart/blob/dev/cpu-runner/src/op/resize.cpp#L77
          // when mode="NEAREST",  half_pixel_bias=0

          // so when mode == "NEAREST" ,  coordinate_transformation_mode
          // "tf_half_pixel_for_nn" equals "half_pixel"
          if (xir_resize_mode == std::string("NEAREST")) {
            xir_half_pixel_centers = 1;
            xir_align_corners = 0;
          } else {
            LOG(WARNING) << "cancel xir conversion, not support Resize attr "
                            "'coordinate_trandformation_mode' = "
                         << coordinate_transformation_mode
                         << " , mode = " << mode;
            return false;
          }
        } else {
          // equation is at onnx operator web page
          // no corresponding xcompiler implementation of tf_half_pixel_for_nn
          // for now check resize.cpp at vart repo for how the operation is
          // implemented model is located at VAI-3712
          LOG(WARNING) << "cancel xir conversion, not support Resize attr "
                          "'coordinate_trandformation_mode' = "
                       << coordinate_transformation_mode;
          return false;
        }

        /**
         * test case : model 20 (resize (X, roi, scale,
         * sizes)) test case : model 21 (resize (X, roi,
         * scale)) resize(X, roi_, scale, sizes)  ->
         * resize(X) scale & sizes is mutually exclusive
         * scale node -> resize attr 'scale'
         * sizes node -> resize attr 'scale'
         */
        auto b = PatternBuilder();
        std::shared_ptr<Pattern> pattern_input = b.wildcard();
        // test case: model # RetinaNet roi.node_arg not
        // exists
        std::shared_ptr<Pattern> pattern_roi =
            b.wildcard(); // roi maybe optional
        // test case: model # RetinaNet scale.node_arg not
        // exists
        std::shared_ptr<Pattern> pattern_scale =
            b.wildcard(); // scales maybe optional
        std::shared_ptr<Pattern> pattern_sizes = b.xir_const_op();
        std::shared_ptr<Pattern> pattern_resize =
            b.node3("Resize",
                    {pattern_input, pattern_roi, pattern_scale, pattern_sizes},
                    {false, true, true, true});

        auto binder = pattern_resize->match(graph, node);
        if (binder == nullptr) {
          return false;
        }
        auto input = (*binder)[pattern_input->get_id()];
        auto scale = (*binder)[pattern_scale->get_id()];
        auto sizes = (*binder)[pattern_sizes->get_id()];
        auto output = (*binder)[pattern_resize->get_id()];
        auto input_shape = *node_arg_get_shape_i64(*input.node_arg);
        auto output_shape = node_get_output_shape(*output.node, 0);
        if (input_shape.size() != 4 || output_shape.size() != 4) {
          LOG(WARNING) << "cancel xir conversion, "
                       << " resize op shape size() is "
                          "not equals 4 . "
                       << node_as_string(node);
          return false;
        }
        auto height_index = 2;
        auto width_index = 3;
        // if model is origin onnx, it's NCHW
        // if the model is generate by graph_opt, it's NHWC
        // simple assume the first two dims equal is nchw, maybe lead to new
        // issue.
        bool is_nchw = (input_shape[0] == output_shape[0]) &&
                       (input_shape[1] == output_shape[1]);
        bool is_nhwc = (input_shape[0] == output_shape[0]) &&
                       (input_shape[1] != output_shape[1]) &&
                       (input_shape[2] != output_shape[2]) &&
                       (input_shape[3] == output_shape[3]);
        if (!is_nchw && !is_nhwc) {
          LOG(WARNING) << "cancel xir conversion, neither NCHW nor NHWC";
          return false;
        } else if (is_nhwc) {
          height_index = 1;
          width_index = 2;
        }

        auto new_scales = std::vector<float>();
        // test case: model # RetinaNet scale.node_arg not
        // exists
        if (scale.node != nullptr) {
          // reference to onnx's document, Since version 13
          // the 'scales' and 'roi' arguments of the
          // 'Resize' operator have been made optional
          // by onnx, So 'scales' and 'roi' may be null
          // now.
          auto scales = self->pass_.get_const_data<float>(*scale.node);
          // according to the document, the scales should
          // be not specified, which mean it is an empty
          // string, but ORT privides the scales and use
          // empty tensor instead.
          if (!scales.empty()) {
            CHECK_EQ(scales.size(), 4);
            new_scales =
                std::vector<float>{scales[width_index], scales[height_index]};
          } else {
            // protobuf does not support to distributsh
            // between the empty repeated fields and the
            // unset repeated fields, it trigger an Error
            // Attribute 'scale' is expected to have field
            // 'floats'
            new_scales = std::vector<float>{0.0f};
          }
        }

        if (sizes.node != nullptr) {
          auto sizes_data = self->pass_.get_const_data<int64_t>(*sizes.node);
          if (sizes_data.size() != 4) {
            LOG(WARNING) << "cancel xir conversion, "
                         << " resize op sizes_data size() is "
                            "not equals 4 . "
                         << node_as_string(node);
            return false;
          }
          new_scales = std::vector<float>{(float)sizes_data[width_index] /
                                              (float)input_shape[width_index],
                                          (float)sizes_data[height_index] /
                                              (float)input_shape[height_index]};
          self->remove_input(3);
        }
        attrs.add("align_corners", static_cast<int64_t>(xir_align_corners));
        attrs.add("half_pixel_centers",
                  static_cast<int64_t>(xir_half_pixel_centers));
        attrs.add("mode", std::string(xir_resize_mode));
        attrs.add("scale", new_scales);

        return true;
      })
      .check_scalar();
  rules.push_back(std::make_unique<ToXirRule>("Slice", "strided_slice", pass));
  (*rules.back())
      .check_scalar()
      .action([](ToXirRule* self, const Graph& graph, const Node& node,
                 NodeAttributesBuilder& attrs) -> bool {
        // https://jira.xilinx.com/browse/VAI-5228 Slice dims=3 has mismatch
        // between CPU EP and IPU.
        if (ENV_PARAM(VAIP_SKIP_SLICE_DIMS_NOT_EQUAL_4)) {
          auto output_shape = node_get_output_shape(node, 0);
          if (output_shape.size() != 4) {
            LOG(WARNING) << "cancel xir conversion, Slice output shape is "
                            "not equals 4 . "
                         << node_as_string(node);
            return false;
          }
        }

        /*
         * test case : model 5021 (Slice(X, starts, ends,
         * axes)) Slice(X, starts, ends, axes)  ->
         * strided_slice(X) starts node -> resize attr
         * 'begin' ends node -> resize attr 'end' axes node
         * -> resize attr 'strides'
         */
        /*
         * test case : model yolov8m (Slice(X, starts,
         * ends, axes, steps)) Slice(X, starts, ends, axes,
         * steps)  -> strided_slice(X) starts node ->
         * resize attr 'begin' ends node
         * -> resize attr 'end' steps node -> resize attr
         * 'strides'
         */
        auto b = PatternBuilder();
        std::shared_ptr<Pattern> pat_input = b.wildcard();
        std::shared_ptr<Pattern> pat_starts = b.xir_const_op();
        std::shared_ptr<Pattern> pat_ends = b.xir_const_op();
        std::shared_ptr<Pattern> pat_axes = b.xir_const_op();
        std::shared_ptr<Pattern> pat_steps = b.xir_const_op();
        std::shared_ptr<Pattern> pat_reshape = b.node3(
            "Slice", {pat_input, pat_starts, pat_ends, pat_axes, pat_steps},
            {false, false, false, true, true});
        auto binder = pat_reshape->match(graph, node);

        if (binder == nullptr)
          return false;
        auto ni_input = (*binder)[pat_input->get_id()];
        auto ni_starts = (*binder)[pat_starts->get_id()];
        auto ni_ends = (*binder)[pat_ends->get_id()];
        auto ni_axes = (*binder)[pat_axes->get_id()];
        auto ni_steps = (*binder)[pat_steps->get_id()];
        auto starts = self->pass_.const_data_into<int64_t>(*ni_starts.node_arg);
        auto ends = self->pass_.const_data_into<int64_t>(*ni_ends.node_arg);
        auto axes = self->pass_.const_data_into<int64_t>(*ni_axes.node_arg);
        std::vector<int64_t> steps;
        // test case : model yolov8m
        if (ni_steps.node_arg != nullptr) {
          steps = self->pass_.const_data_into<int64_t>(*ni_steps.node_arg);
        } else {
          // test case : #5024 shufflenet_v2_x2_0
          steps = std::vector<int64_t>(axes.size(), 1);
        }
        CHECK(starts.size() == axes.size());
        CHECK(ends.size() == axes.size());
        CHECK(steps.size() == axes.size());

        if (!node_has_attr(node, "shape")) {
          auto input_shape_ptr = node_arg_get_shape_i64(*ni_input.node_arg);
          CHECK(input_shape_ptr != nullptr)
              << node_arg_as_string(*ni_input.node_arg) << " shape absent";
          auto input_shape = *input_shape_ptr;

          auto begin = std::vector<int64_t>(input_shape.size(), 0);

          auto end = std::vector<int64_t>(input_shape.size(),
                                          std::numeric_limits<int32_t>::max());
          auto strides = std::vector<int64_t>(input_shape.size(), 1);
          for (auto i = 0u; i < axes.size(); i++) {
            auto index = (axes[i] + input_shape.size()) % input_shape.size();
            begin[index] = starts[i];

            // onnx slice node :  For slicing to the end of a dimension with
            // unknown size, it is recommended to pass in INT_MAX when slicing
            // forward and 'INT_MIN' when slicing backward.
            // test case :  model PST (with -l 0)
            int64_t end_v =
                std::max(ends[i], (int64_t)std::numeric_limits<int32_t>::min());

            // test case: model xilinxSR
            end[index] =
                std::min(end_v, (int64_t)std::numeric_limits<int32_t>::max());
            strides[index] = steps[i];
          }
          attrs.add("begin", begin);
          attrs.add("end", end);
          attrs.add("strides", strides);
          self->remove_input(1);
          self->remove_input(2);
          self->remove_input(3);
          if (ni_steps.node_arg != nullptr) {
            self->remove_input(4);
          }
          return true;
        }
        return false;
      });
  // testcase:see vaip#1376
  rules.push_back(
      std::make_unique<ToXirRule>("LpNormalization", "l2_normalize", pass));
  (*rules.back())
      .check_scalar()
      .action([](ToXirRule* self, const Graph& graph, const Node& node,
                 NodeAttributesBuilder& attrs) -> bool {
        auto axis = node_get_attr_int_with_default(node, "axis", -1);
        auto p = node_get_attr_int_with_default(node, "p", 2);
        if (p != 2) {
          LOG(WARNING) << "cancel xir conversion, only support p = 2 now"
                       << node_as_string(node);
          return false;
        }
        auto shape = node_get_output_shape(node, 0);
        if (axis < 0) {
          axis = shape.size() + axis;
        }
        std::vector<int64_t> new_axis = {axis};
        attrs.add("axis", new_axis);
        return true;
      });
  rules.push_back(std::make_unique<ToXirRule>("Sqrt", "sqrt", pass));
  (*rules.back()).check_scalar();
  rules.push_back(std::make_unique<ToXirRule>("Pow", "pow", pass));
  (*rules.back()).check_scalar();
  rules.push_back(std::make_unique<ToXirRule>("Expand", "expand", pass));
  (*rules.back()).check_scalar();
  /// end of rules
  rules.push_back(std::make_unique<ToXirRule>("Tanh", "tanh", pass));
  (*rules.back()).check_scalar();
  rules.push_back(std::make_unique<ToXirRule>("Cast", "cast", pass));
  (*rules.back()).check_scalar();
  rules.push_back(std::make_unique<ToXirRule>("Neg", "neg", pass));
  (*rules.back()).check_scalar();
  rules.push_back(
      std::make_unique<ToXirRule>("com.microsoft:Gelu", "gelu", pass));
  (*rules.back()).check_scalar();
  rules.push_back(std::make_unique<ToXirRule>("PRelu", "prelu", pass));
  (*rules.back()).check_scalar();
  rules.push_back(std::make_unique<ToXirRule>("Gather", "gather", pass));
  (*rules.back())
      .copy("axis")
      .action([](ToXirRule* self, const Graph& graph, const Node& node,
                 NodeAttributesBuilder& attrs) -> bool {
        auto b = PatternBuilder();
        std::shared_ptr<Pattern> pat_data = b.wildcard();
        std::shared_ptr<Pattern> pat_indices = b.wildcard();
        std::shared_ptr<Pattern> pat_gather =
            b.node2("Gather", {pat_data, pat_indices});
        auto binder = pat_gather->match(graph, node);
        if (binder == nullptr) {
          return false;
        }
        auto ni_indices = (*binder)[pat_indices->get_id()];
        CHECK(ni_indices.node_arg != nullptr);
        if (node_arg_is_scalar(*ni_indices.node_arg)) {
          attrs.add("scalar_index", static_cast<int64_t>(1));
        }
        return true;
      });

  rules.push_back(std::make_unique<ToXirRule>("Tile", "broadcast_tile", pass));
  (*rules.back()).check_scalar();
  rules.push_back(std::make_unique<ToXirRule>("Identity", "identity", pass));
  rules.push_back(std::make_unique<ToXirRule>(
      "LayerNormalization", "layernorm",
      pass)); // testcases mixer_b16_224.miil_in21k_ft_in1k.onnx
  (*rules.back()).check_scalar();
  rules.push_back(std::make_unique<ToXirRule>("InstanceNormalization",
                                              "instancenorm_nchw", pass));
  (*rules.back())
      .check_scalar()
      .rename("epsilon", "eps")
      .action(convert_instnorm("InstanceNormalization"));
  rules.push_back(
      std::make_unique<ToXirRule>("com.vai.quantize:VitisInstanceNormalization",
                                  "instancenorm_nchw", pass));
  (*rules.back())
      .check_scalar()
      .rename("epsilon", "eps")
      .action(convert_instnorm("com.vai.quantize:VitisInstanceNormalization"));
  rules.push_back(std::make_unique<ToXirRule>("com.vai.quantize:BFPFixNeuron",
                                              "fix", pass));
  (*rules.back())
      .action([](ToXirRule* self, const Graph& graph, const Node& node,
                 NodeAttributesBuilder& attrs) -> bool {
        auto bfp_method =
            node_get_attr_string_with_default(node, "bfp_method", "to_bfp");
        auto bit_width = node_get_attr_int_with_default(node, "bit_width", 16);
        auto block_size = node_get_attr_int_with_default(node, "block_size", 8);
        auto axis = node_get_attr_int_with_default(node, "axis", 1);
        auto rounding_mode =
            node_get_attr_int_with_default(node, "rounding_mode", 0);
        auto sub_block_size =
            node_get_attr_int_with_default(node, "sub_block_size", 2);
        auto sub_block_shift_bits =
            node_get_attr_int_with_default(node, "sub_block_shift_bits", 2);
        auto convert_to_bfloat_before_bfp = node_get_attr_int_with_default(
            node, "convert_to_bfloat_before_bfp", 0);

        auto round_mode = "PY3_ROUND";
        if (rounding_mode == 0) {
          round_mode = "STD_ROUND";
        } else if (rounding_mode == 1) {
          round_mode = "DPU_ROUND";
        } else if (rounding_mode == 2) {
          round_mode = "PY3_ROUND";
        } else {
          LOG(WARNING) << "cancel xir conversion , BFPFixNeuron, not support "
                          "rounding mode "
                       << rounding_mode;
          return false;
        }

        auto fix_point = 0; // this fix_point is meaningless, xir need required
        attrs.add("fix_point", static_cast<int64_t>(fix_point))
            .add("if_signed", static_cast<int64_t>(1))
            .add("bfp_method", bfp_method)
            .add("bit_width", static_cast<int64_t>(bit_width))
            .add("block_size", static_cast<int64_t>(block_size))
            .add("axis", static_cast<int64_t>(axis))
            .add("round_mode", round_mode)
            .add("sub_block_size", static_cast<int64_t>(sub_block_size))
            .add("sub_block_shift_bits",
                 static_cast<int64_t>(sub_block_shift_bits))
            .add("convert_to_bfloat_before_bfp",
                 static_cast<int64_t>(convert_to_bfloat_before_bfp));

        return true;
      });
  auto supported_ops = std::vector<std::string>();
  for (auto& pass_conf : pass.get_context()->get_config_proto().passes()) {
    for (auto op : pass_conf.pass_dpu_param().supported_op()) {
      supported_ops.push_back(op);
    }
  }

  for (auto node_idx : graph_get_node_in_topoligical_order(graph)) {
    auto node = VAIP_ORT_API(graph_get_node)(graph, node_idx);
    CHECK(node != nullptr) << "node_idx " << node_idx << " ";
    if (!supported_ops.empty()) {
      auto op_type = node_op_type(*node);
      auto it = std::find(supported_ops.begin(), supported_ops.end(), op_type);
      if (it == supported_ops.end()) {
        continue;
      }
    }
    for (auto& rule : rules) {
      auto ok = rule->apply(graph, *node);
      MY_LOG(2) << "try rule " << rule->debug_string() << " " << ok;
      if (ok) {
        break;
      }
    }
  }
}
} // namespace vaip_pass_to_xir_ops
