/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */
#ifdef __GNUC__
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wpedantic"
#  pragma GCC diagnostic ignored "-Wconversion"
#endif

#include "dtype_util.h"
#include "vaip/dd/coeffs.hpp"
#include "vaip/dd/dd_utils.hpp"

#include "vaip/pattern_zoo.hpp"

#ifdef __GNUC__
#  pragma GCC diagnostic pop
#endif
#include "vaip/vaip.hpp"
#include "vitis/ai/env_config.hpp"
#include <functional>
#include <glog/logging.h>
#include <numeric>
DEF_ENV_PARAM(DEBUG_DD_MERGE_QDQ_ADJUSTMENT, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_DD_MERGE_QDQ_ADJUSTMENT) >= n)

// add the following line in your vaip_config.json
/*
    { "name": "vaip_pass_dd_merge_qdq_adjustment",
       "plugin": "vaip-pass_dd_merge_qdq_adjustment",
       "disabled": false
    }
*/
/*This pass is used in PSR when QGelu and QElementMul_qdq have a parent node
 QSlice. In this scenario, the nodes have to take the QDQ values from the QSlice
 node. This pass will run at the end of all fusions, and it will adjust the QDQ
 values of QGelu and QElementMul to their correct values from the QSlice parent
 node.*/
namespace {
using namespace vaip_core;
struct Qdq_adjustment {
  Qdq_adjustment(IPass& self) : self_{self} {}

  void qdq_post_processing(Graph& graph) {
    for (const auto node_idx : graph_get_node_in_topoligical_order(graph)) {
      auto node = VAIP_ORT_API(graph_get_node)(graph, node_idx);
      std::string precision = "a168"; // not used
      auto node_ctx = vaip::dtype_util::build_context(graph, node, precision);
      auto node_op = VAIP_ORT_API(node_op_type)(*node);
      auto nab = NodeAttributesBuilder();

      if ((node_op == "QGelu" || node_op == "QELWEMUL_qdq") &&
          node_has_attr(*node, "generic_fusion")) {
        for (auto x : node_ctx.parent_ops) {
          auto parent_op_type = VAIP_ORT_API(node_op_type)(*x);
          if (parent_op_type == "QSlice") {
            auto& slice_attrs = node_get_attributes_ref(*x);

            auto attr_proto = node_attributes_get(slice_attrs, "q_scale");
            float slice_q_scale =
                VAIP_ORT_API(attr_proto_get_float)(*attr_proto);

            attr_proto = node_attributes_get(slice_attrs, "q_zp");
            float slice_q_zp = VAIP_ORT_API(attr_proto_get_int)(*attr_proto);

            auto& qgelu_attrs = node_get_attributes_ref(*node);
            attr_proto = node_attributes_get(qgelu_attrs, "input_q_params");
            // std::cout<<"slice_q"<<slice_q_scale<<std::endl;
            // std::cout<<"slice_zero point"<<slice_q_zp<<std::endl;
            std::vector<float> input_q_params = {slice_q_scale, slice_q_zp};

            // VAIP_ORT_API(attr_proto_get_int)(*attr_proto)[0]= slice_q_scale;
            // VAIP_ORT_API(attr_proto_get_int)(*attr_proto)[1]= slice_q_zp;

            // nab.add("slice_q_scale", slice_q_scale);
            nab.add("input_q_params", input_q_params);
            auto xx = const_cast<Node*>(node);
            nab.merge_into(*xx);
          }
        }
      }
    }
  }
  void process(IPass& self, Graph& graph) { qdq_post_processing(graph); }

  IPass& self_;
};
} // namespace

DEFINE_VAIP_PASS(Qdq_adjustment, vaip_pass_dd_merge_qdq_adjustment)
