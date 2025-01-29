/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */
#include <glog/logging.h>
#include <iostream>

#include "vaip/vaip.hpp"
#include "vitis/ai/env_config.hpp"

DEF_ENV_PARAM(DEBUG_CONVERT_IN_TO_GN, "0")

namespace {
using namespace vaip_core;
static std::ostream& operator<<(std::ostream& s,
                                const std::vector<int64_t>& v) {
  s << "[";
  for (auto c = 0u; c < v.size(); ++c) {
    if (c != 0) {
      s << ",";
    }
    s << v[c];
  }
  s << "]";
  return s;
}

struct MergeReshapeInstanceNorm {
  MergeReshapeInstanceNorm(IPass& self) : self_{self} {}
  template <typename T>
  inline std::vector<T> create_new_scale_data(const Node& node,
                                              int64_t gn_channel) {
    auto scale_const_data = self_.get_const_data<T>(node);
    auto new_scale_data = std::vector<T>();
    new_scale_data.reserve(gn_channel);
    // expand data to gn_channel size
    while (new_scale_data.size() < (size_t)gn_channel) {
      new_scale_data.insert(new_scale_data.end(), scale_const_data.begin(),
                            scale_const_data.end());
    }
    return new_scale_data;
  }
  std::unique_ptr<Rule> create_rule() {
    auto builder = PatternBuilder();
    std::shared_ptr<Pattern> pat_input = builder.wildcard();
    std::shared_ptr<Pattern> pat_reshape_top =
        builder.node2("com.xilinx:reshape", {pat_input});
    std::shared_ptr<Pattern> pat_q_top =
        builder.node3("com.xilinx:quantize_linear",
                      {pat_reshape_top, builder.wildcard(), builder.wildcard()},
                      {false, false, true});
    std::shared_ptr<Pattern> pat_dq_top =
        builder.node3("com.xilinx:dequantize_linear",
                      {pat_q_top, builder.wildcard(), builder.wildcard()},
                      {false, false, true});

    std::shared_ptr<Pattern> pat_scale_input_1 = builder.xir_const_op();
    std::shared_ptr<Pattern> pat_scale_input_2 = builder.xir_const_op();
    std::shared_ptr<Pattern> pat_scale_input_3 = builder.xir_const_op();
    std::shared_ptr<Pattern> pat_instance_scale = builder.node2(
        "com.xilinx:dequantize_linear",
        {pat_scale_input_1, pat_scale_input_2, pat_scale_input_3});

    std::shared_ptr<Pattern> pat_bias_input_1 = builder.xir_const_op();
    std::shared_ptr<Pattern> pat_bias_input_2 = builder.xir_const_op();
    std::shared_ptr<Pattern> pat_bias_input_3 = builder.xir_const_op();
    std::shared_ptr<Pattern> pat_instance_bias =
        builder.node2("com.xilinx:dequantize_linear",
                      {pat_bias_input_1, pat_bias_input_2, pat_bias_input_3});

    std::shared_ptr<Pattern> pat_instance =
        builder.node2("com.xilinx:instancenorm_ncd",
                      {pat_dq_top, pat_instance_scale, pat_instance_bias});

    std::shared_ptr<Pattern> pat_q_bottom =
        builder.node3("com.xilinx:quantize_linear",
                      {pat_instance, builder.wildcard(), builder.wildcard()},
                      {false, false, true});
    std::shared_ptr<Pattern> pat_dq_bottom =
        builder.node3("com.xilinx:dequantize_linear",
                      {pat_q_bottom, builder.wildcard(), builder.wildcard()},
                      {false, false, true});

    std::shared_ptr<Pattern> pat_reshape_bottom =
        builder.node2("com.xilinx:reshape", {pat_dq_bottom});

    return Rule::create_rule(
        pat_reshape_bottom,
        [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
          auto ni_input = binder[pat_input->get_id()];
          auto ni_instance = binder[pat_instance->get_id()];
          auto ni_reshape = binder[pat_reshape_bottom->get_id()];
          auto ni_scale_input_1 = binder[pat_scale_input_1->get_id()];
          auto ni_scale_input_2 = binder[pat_scale_input_2->get_id()];
          auto ni_scale_input_3 = binder[pat_scale_input_3->get_id()];
          auto ni_instance_scale = binder[pat_instance_scale->get_id()];

          auto ni_bias_input_1 = binder[pat_bias_input_1->get_id()];
          auto ni_bias_input_2 = binder[pat_bias_input_2->get_id()];
          auto ni_bias_input_3 = binder[pat_bias_input_3->get_id()];
          auto ni_instance_bias = binder[pat_instance_bias->get_id()];

          auto input_shape = node_arg_get_shape_i64(*ni_input.node_arg);
          auto instance_shape = node_arg_get_shape_i64(*ni_instance.node_arg);
          auto output_shape = node_arg_get_shape_i64(*ni_reshape.node_arg);

          if (input_shape == nullptr || output_shape == nullptr ||
              instance_shape == nullptr) {
            LOG(WARNING) << "cancel convert IN to GN,  input shape is null, "
                         << node_as_string(*ni_instance.node);
            return false;
          }
          LOG_IF(INFO, ENV_PARAM(DEBUG_CONVERT_IN_TO_GN))
              << "convert IN to GN : input_shape : " << *input_shape //
              << " output_shape " << *output_shape                   //
              << " instance_shape " << *instance_shape;

          if (*input_shape != *output_shape) {
            LOG(WARNING) << "cancel convert IN to GN , pattern input shape not "
                            "eq output shape.";
            return false;
          }
          auto gn_op_type = "groupnorm_nchw";
          auto dim_size = (*input_shape).size();

          if (dim_size == 4) {
            gn_op_type = "groupnorm_nchw";
          } else if (dim_size == 3) {
            gn_op_type = "groupnorm_ncd";
          } else {
            LOG(WARNING) << "cancel convert IN to GN, only support 3/4 dims "
                            "InstanceNorm "
                         << node_as_string(*ni_instance.node);
            return false;
          }

          auto gn_channel = (*input_shape)[1];
          auto in_channel = (*instance_shape)[1];

          if (in_channel == 0 || gn_channel < in_channel ||
              gn_channel % in_channel != 0) {
            LOG(WARNING) << "cancel convert IN to GN, pattern input 'C' ("
                         << gn_channel
                         << ")must be an interger multiple of instance 'C'"
                         << in_channel;
            return false;
          }

          auto group = gn_channel / in_channel;

          auto scale_data_type =
              node_arg_get_element_type(*ni_scale_input_1.node_arg);

          std::vector<uint8_t> new_scale_data;
          std::vector<int8_t> new_scale_data_int8;
          gsl::span<char> span_scale_data;
          if (scale_data_type == 2) {
            new_scale_data = create_new_scale_data<uint8_t>(
                *ni_scale_input_1.node, gn_channel);
            span_scale_data =
                gsl::span<char>((char*)new_scale_data.data(),
                                new_scale_data.size() * sizeof(uint8_t));
          } else if (scale_data_type == 3) {
            new_scale_data_int8 = create_new_scale_data<int8_t>(
                *ni_scale_input_1.node, gn_channel);
            span_scale_data =
                gsl::span<char>((char*)new_scale_data_int8.data(),
                                new_scale_data_int8.size() * sizeof(int8_t));
          } else {
            LOG(WARNING)
                << "cancel convert IN to GN, not support scale_data_type "
                << scale_data_type << " "
                << data_type_to_string(scale_data_type);
            return false;
          }

          auto bais_data_type =
              node_arg_get_element_type(*ni_bias_input_1.node_arg);

          if (bais_data_type != 6) {
            // int32
            LOG(WARNING)
                << "cancel convert IN to GN, not support bais_data_type "
                << bais_data_type;
            return false;
          }
          // get data from context
          auto bias_const_data =
              self_.get_const_data<int32_t>(*ni_bias_input_1.node);
          auto new_bias_data = std::vector<int32_t>();
          new_bias_data.reserve(gn_channel);
          // expand data to gn_channel size
          while (new_bias_data.size() < (size_t)gn_channel) {
            new_bias_data.insert(new_bias_data.end(), bias_const_data.begin(),
                                 bias_const_data.end());
          }

          gsl::span<char> span_bias_data =
              gsl::span<char>((char*)new_bias_data.data(),
                              new_bias_data.size() * sizeof(int32_t));

          auto gn_const_input_shape = std::vector<int64_t>{(int64_t)gn_channel};
          auto& new_const_scale =
              NodeBuilder(*graph, self_)
                  .clone_op_type(*ni_scale_input_1.node)
                  .clone_data_type(*ni_scale_input_1.node)
                  .set_anchor_point3(*ni_scale_input_1.node_arg, {"reshape"},
                                     gn_const_input_shape)
                  .build();

          self_.create_const(new_const_scale, span_scale_data);

          auto& new_const_bias =
              NodeBuilder(*graph, self_)
                  .clone_op_type(*ni_bias_input_1.node)
                  .clone_data_type(*ni_bias_input_1.node)
                  .set_anchor_point3(*ni_bias_input_1.node_arg, {"reshape"},
                                     gn_const_input_shape)
                  .build();

          self_.create_const(new_const_bias, span_bias_data);
          auto& new_dq_scale =
              NodeBuilder(*graph, self_)
                  .clone_op_type(*ni_instance_scale.node)
                  .clone_data_type(*ni_instance_scale.node)
                  .set_input_nodes({&new_const_scale, ni_scale_input_2.node,
                                    ni_scale_input_3.node})
                  .set_anchor_point3(*ni_instance_scale.node_arg, {"reshape"},
                                     gn_const_input_shape)
                  .build();

          auto& new_dq_bias =
              NodeBuilder(*graph, self_)
                  .clone_op_type(*ni_instance_bias.node)
                  .clone_data_type(*ni_instance_bias.node)
                  .set_input_nodes({&new_const_bias, ni_bias_input_2.node,
                                    ni_bias_input_3.node})
                  .set_anchor_point3(*ni_instance_bias.node_arg, {"reshape"},
                                     gn_const_input_shape)
                  .build();

          NodeBuilder(*graph, self_)
              .set_op_type(gn_op_type)
              .set_input_nodes({ni_input.node, &new_dq_scale, &new_dq_bias})
              .clone_attrs(*ni_instance.node)
              .add("num_groups", group)
              .add("num_channels", gn_channel)
              .set_anchor_point1(*ni_reshape.node)
              .build();

          return true;
        });
  }
  void process(IPass& self, Graph& graph) { create_rule()->apply(&graph); }

public:
  IPass& self_;
};
} // namespace

DEFINE_VAIP_PASS(MergeReshapeInstanceNorm,
                 vaip_pass_convert_instancenorm_to_groupnorm)
