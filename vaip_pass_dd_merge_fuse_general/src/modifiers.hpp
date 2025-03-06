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

// DEF_ENV_PARAM(DEBUG_DD_PATTERN, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_DD_PATTERN) >= n)
namespace {
using namespace vaip_core;

struct FuseState {

  binder_t& binder;
  onnxruntime::Graph* graph;
  NodeBuilder& node_builder;
  PassProto& pass_proto;
  IPass* self;

  std::string op_name;                       // op type name
  std::string node_name;                     // name of output node
  int64_t generic_fusion = 1LL;              // if ==0 then dont fuse

  std::vector<std::string> modifiers;        // list of modifers

  std::vector<std::string> input_node_names; // list of input names to pattern

  NodeInput out_node;                        // output node in

  std::vector<const NodeArg*>
      input_node_args; // list of input node args for fused node

  std::vector<std::string>
      node_names; // list of nodes which have matched the pattern

  std::vector<std::string> in_dtypes;
  std::vector<std::string> out_dtypes;

  std::vector<float> input_q_params;
  std::vector<float> output_q_params;

  std::vector<int64_t> input_shape;
  std::vector<std::vector<int64_t>> input_shapes;
  std::vector<int64_t> output_shape;

  FuseState(onnxruntime::Graph* _graph, binder_t& _binder,
            NodeBuilder& _node_builder, PassProto& _pass_proto, IPass* _self)
      : binder(_binder), graph(_graph), node_builder(_node_builder),
        pass_proto(_pass_proto), self(_self) {
    op_name = pass_proto.pass_fusion_param().op_name();

    {
      auto modifiers_extractors =
          pass_proto.pass_fusion_param().extractor().modifiers();
      modifiers = std::vector<std::string>(modifiers_extractors.begin(),
                                           modifiers_extractors.end());
    }

    {
      auto input_nodes = pass_proto.pass_fusion_param().pattern().input_names();
      input_node_names =
          std::vector<std::string>(input_nodes.begin(), input_nodes.end());
    }

    out_node = binder[pass_proto.pass_fusion_param().pattern().output_names()];
    node_name = node_arg_get_name(*out_node.node_arg);

    node_names = vaip::dd::get_node_names(graph, binder);
  }
};

static void extract_subgraph(FuseState* fuse_state) {
  auto& context =
      dynamic_cast<PassContextImp&>(*(fuse_state->self->get_context()));
  auto subgraph_metadefs_ = context.context_proto.mutable_subgraph_metadefs();

  std::vector<std::string> inputs;
  for (auto in_nodearg : fuse_state->input_node_args)
    inputs.push_back(node_arg_get_name(*in_nodearg));

  std::vector<std::string> outputs = {fuse_state->node_name};

  auto constant_initializers = std::vector<std::string>{};

  auto level_0_graph =
      (Graph*)context.get_context_resource("__level_0_graph").get();

  auto [meta_def, err] = fuse_state->self->try_fuse(
      *level_0_graph, fuse_state->op_name + fuse_state->node_name, inputs,
      outputs, constant_initializers, fuse_state->op_name);

  if (meta_def == nullptr) {
    LOG(FATAL) << "Cannot fuse:  " << err.comments;
  } else {
    (*subgraph_metadefs_)[fuse_state->node_name].set_op_name(
        fuse_state->op_name);
    (*subgraph_metadefs_)[fuse_state->node_name].mutable_metadef()->CopyFrom(
        *meta_def);
  }
}

static void quantop_predicate(FuseState* fuse_state) {
  if (fuse_state->op_name == "QuantOP") {
    auto node_name = node_arg_get_name(*fuse_state->binder["dq"].node_arg);
    if (node_name == "input_1_QuantizeLinear_Output" ||
        node_name == "input_2_QuantizeLinear_Output" ||
        node_name == "input_1_q_to_dq") {
      fuse_state->generic_fusion = 0; // Don't fuse
    }
  }
}

static void dequantop_predicate(FuseState* fuse_state) {
  if (fuse_state->op_name == "DeQuantOP") {
    auto node_name = node_arg_get_name(*fuse_state->binder["dq"].node_arg);
    if (node_name == "output_1" ||
        node_name == "output_1_channel_first_0_DequantizeLinear_Output") {
      fuse_state->generic_fusion = 0; // Don't fuse
    }
  }
}

static void QBatchMatMul_predicate(FuseState* fuse_state) {
  // Assuming second input is weight
  auto w_shape = *node_arg_get_shape_i64(*fuse_state->input_node_args[1]).get();
  auto act_shape =
      *node_arg_get_shape_i64(*fuse_state->input_node_args[0]).get();
  if (w_shape.size() != 3 ||
      w_shape[1] != act_shape.back()) { // check the inner dimension
    fuse_state->generic_fusion = 0;
  }
}

static void mhamzdk5_predicate(FuseState* fuse_state) {
  if (fuse_state->op_name != "mzdk5MHA") {
    throw std::runtime_error("Predicate specific to mzdk5MHA op used for " +
                             fuse_state->op_name + " .");
    return;
  }

  // Assuming first input is q node and second is k node
  auto input_q_node = fuse_state->binder[fuse_state->input_node_names[0]];
  auto input_k_node = fuse_state->binder[fuse_state->input_node_names[1]];

  auto q_shape = *node_arg_get_shape_i64(*input_q_node.node_arg).get();
  auto kt_shape = *node_arg_get_shape_i64(*input_k_node.node_arg).get();

  int64_t M = q_shape[0] * q_shape[1] * q_shape[2];
  int64_t K = q_shape.back();
  int64_t N = kt_shape.back();
  std::vector<int64_t> qkt_mm_shape = {M, K, N};

  bool offload_to_mha = true;
  std::vector<std::vector<int64_t>> unsupported_MKN_shapes = {
      {4096, 64, 4096}, {1024, 64, 1024}, {256, 64, 256}}; // QKT
  for (const auto& v : unsupported_MKN_shapes) {
    if (v[0] == qkt_mm_shape[0] && v[1] == qkt_mm_shape[1] &&
        v[2] == qkt_mm_shape[2]) {
      offload_to_mha = false;
    }
  }

  if (!offload_to_mha)
    fuse_state->generic_fusion = 0LL;
}

static std::vector<const Node*> get_all_parent_nodes(const Node* cnode) {
  auto node_inputs = node_get_inputs(*cnode);
  std::vector<const Node*> ret;
  for (const auto& ni : node_inputs) {
    if (ni.node != nullptr) {
      ret.emplace_back(ni.node);
    }
  }
  return ret;
}

static bool check_no_op_parent(Graph& g, const Node* a,
                               NodeArg*& updated_node_arg,
                               std::string no_op_name) {
  auto inputs = get_all_parent_nodes(a);
  if (inputs.size() == 0)
    return false;
  auto x = inputs[0];
  auto parent_op_type = VAIP_ORT_API(node_op_type)(*x);
  if (parent_op_type != no_op_name)
    return false;
  else {
    inputs = get_all_parent_nodes(x);
    if (inputs.size() == 0)
      return false;
    x = inputs[0];
    parent_op_type = VAIP_ORT_API(node_op_type)(*x);
    if (parent_op_type != "DequantizeLinear")
      return false;
  }
  auto input_node_args = node_get_input_node_args(*x);
  for (auto ni : input_node_args) {
    if (!node_arg_is_constant(g, *ni)) {
      updated_node_arg = const_cast<NodeArg*>(ni);
      continue;
    }
  }
  return true;
}

static void psw_prepro_hardcoding(FuseState* fuse_state) {
  NodeArg* no_op_node_arg = nullptr;
  auto in_node = fuse_state->binder["input_0"];
  bool no_op_in_parent = check_no_op_parent(*(fuse_state->graph), in_node.node,
                                            no_op_node_arg, "Unsqueeze");
  if (no_op_in_parent) {
    fuse_state->input_node_args[0] = no_op_node_arg;
    auto temp = fuse_state->input_node_args[0];
    fuse_state->input_shape = *node_arg_get_shape_i64(*temp).get();
  }
}

const NodeArg& predicate_qslice_hardcode(std::string arg,
                                         NodeBuilder& node_builder,
                                         binder_t& binder,
                                         onnxruntime::Graph* graph,
                                         std::string& node_name) {
  std::vector<std::string> attrs = splitString(arg);

  // Throw error if 3 attr not present
  if (3 != attrs.size()) {
    std::string error_msg = "qslice_hardcode argument should have 3 space "
                            "separated arguments passed.\n"
                            "Received string = " +
                            arg +
                            "\n"
                            "Size = " +
                            std::to_string(attrs.size());
    throw std::runtime_error(error_msg);
  }

  float q_s =
      node_arg_get_const_data_as_float(*graph, *binder[attrs[0]].node_arg);

  int64_t q_z = vaip::dd::get_zp_from_node(*graph, *binder[attrs[1]].node_arg);

  auto start_data =
      node_arg_get_const_data_as_i64s(*graph, *binder[attrs[2]].node_arg);
  int64_t slice_idx = start_data[0] == 0 ? 0 : 1;

  auto value = std::vector<uint8_t>(16, 0);
  auto shape = std::vector<int64_t>{16};
  auto name = node_name + "_qdq";
  auto new_tensor = tensor_proto_new_u8(name, shape, value);
  VAIP_ORT_API(graph_add_initialized_tensor)(*graph, *new_tensor);

  node_builder.add("q_scale", q_s);
  node_builder.add("q_zp", q_z);
  node_builder.add("slice_idx", slice_idx);

  return VAIP_ORT_API(node_arg_new)(*graph, name, &shape,
                                    ONNX_NAMESPACE::TensorProto_DataType_UINT8);
}

static void qslice_modifier(FuseState* fuse_state) {
  auto ed = *(find(fuse_state->modifiers.begin(), fuse_state->modifiers.end(),
                   "qslice_modifier") +
              1);
  const NodeArg& qdq_node_arg = predicate_qslice_hardcode(
      ed, fuse_state->node_builder, fuse_state->binder, fuse_state->graph,
      fuse_state->node_name);
  fuse_state->input_node_args.push_back(&qdq_node_arg);
}

static void set_explicit_io_dtype(std::string ed,
                                  std::vector<std::string>& in_dtypes,
                                  std::string default_dtype = ".") {
  auto dtypes = splitString(ed);
  for (int i = 0; i < dtypes.size(); i++) {
    // In case in_dtypes.size() < dtypes.size()
    if (i >= in_dtypes.size())
      in_dtypes.push_back(default_dtype);

    if ("." == dtypes[i])
      continue;
    else
      in_dtypes[i] = dtypes[i];
  }
}

static void mhamzdk5_modifier(FuseState* fuse_state) {
  // if next set of nodes contain unfused concat pattern, i.e if concat op is a
  // child node of final_quant_node vsm_sc and vsm_zp (output scale and
  // zeropoint of calling op) are updated with  scale and zeropoint of
  // QuantizeLinear after concat (concat op's output scale and zeropoint)

  // TODO : remove this after inferring all input shapes
  // Assuming first input is q node and second is k node
  auto input_q_node = fuse_state->binder[fuse_state->input_node_names[0]];
  auto input_k_node = fuse_state->binder[fuse_state->input_node_names[1]];

  fuse_state->node_builder.add(
      "q_shape_back",
      (*node_arg_get_shape_i64(*input_q_node.node_arg).get()).back());
  fuse_state->node_builder.add(
      "k_shape_back",
      (*node_arg_get_shape_i64(*input_k_node.node_arg).get()).back());

  // Check if these's only one consumer and get it's name and nodearg
  if (graph_get_consumer_nodes(
          *fuse_state->graph, node_arg_get_name(*fuse_state->out_node.node_arg))
          .size() == 1) {
    std::vector<const Node*> final_quant_node_nextnodes =
        graph_get_consumer_nodes(
            *fuse_state->graph,
            node_arg_get_name(*fuse_state->out_node.node_arg));
    std::string dq_before_concat_node_name =
        node_get_first_output_name(*final_quant_node_nextnodes[0]);
    auto dq_before_concat_node_arg = VAIP_ORT_API(graph_get_node_arg)(
        *fuse_state->graph, dq_before_concat_node_name);

    // Check if these's only one consumer and get it's name and nodearg
    if (graph_get_consumer_nodes(*fuse_state->graph,
                                 node_arg_get_name(*dq_before_concat_node_arg))
            .size() == 1) {
      std::vector<const Node*> dq_before_concat_next_nodes =
          graph_get_consumer_nodes(
              *fuse_state->graph,
              node_arg_get_name(*dq_before_concat_node_arg));
      std::string concat_node_name =
          node_get_first_output_name(*dq_before_concat_next_nodes[0]);
      auto concat_node_arg = VAIP_ORT_API(graph_get_node_arg)(
          *fuse_state->graph, concat_node_name);
      // Get the op_type of 2nd level consumer and check if it is concat
      auto concat_node_op_type =
          VAIP_ORT_API(node_op_type)(*dq_before_concat_next_nodes[0]);

      if (concat_node_op_type == "Concat") {
        // if 2nd level consumer is Concat op, get it's child QuantizeLinear
        // node's scale and zero point and update vsm_sc and vsm_zp
        if (graph_get_consumer_nodes(*fuse_state->graph,
                                     node_arg_get_name(*concat_node_arg))
                .size() == 1) {
          std::vector<const Node*> concat_node_nextnodes =
              graph_get_consumer_nodes(*fuse_state->graph,
                                       node_arg_get_name(*concat_node_arg));
          std::string Q_node_after_concat_name =
              node_get_first_output_name(*concat_node_nextnodes[0]);
          auto Q_node_input_node_args =
              node_get_input_node_args(*concat_node_nextnodes[0]);

          size_t vsm_sc_idx = 13;
          size_t vsm_zp_idx = 14;
          fuse_state->input_node_args[vsm_sc_idx] = Q_node_input_node_args[1];
          fuse_state->input_node_args[vsm_zp_idx] = Q_node_input_node_args[2];
          fuse_state->in_dtypes[vsm_sc_idx] =
              vaip::dd::nodearg_dtype_to_string(*Q_node_input_node_args[1]);
          fuse_state->in_dtypes[vsm_zp_idx] =
              vaip::dd::nodearg_dtype_to_string(*Q_node_input_node_args[2]);
        }
      }
    }
  }
}

static void in_dtype_modifier(FuseState* fuse_state) {
  auto ed = *(find(fuse_state->modifiers.begin(), fuse_state->modifiers.end(),
                   "in_dtype_modifier") +
              1);
  set_explicit_io_dtype(ed, fuse_state->in_dtypes);
}

static void out_dtype_modifier(FuseState* fuse_state) {
  auto ed = *(find(fuse_state->modifiers.begin(), fuse_state->modifiers.end(),
                   "out_dtype_modifier") +
              1);
  set_explicit_io_dtype(ed, fuse_state->out_dtypes);
}

static std::pair<std::vector<int64_t>, std::vector<int64_t>>
get_NCHW_NHWC(const std::vector<int64_t>& shapes) {
  if (shapes.size() == 4) {
    if (shapes[1] == shapes[2]) {
      return {{shapes[0], shapes[3], shapes[1], shapes[2]}, shapes};
    } else if (shapes[2] == shapes[3]) {
      return {shapes, {shapes[0], shapes[2], shapes[3], shapes[1]}};
    }
  }
  return {shapes, shapes};
}

static void apply_qgroupnorm_hardcodings(FuseState* fuse_state) {

  std::string concat_in_sibling = "false";
  auto graph = fuse_state->graph;
  auto in_node = fuse_state->binder["input_0"];
  auto node_found = in_node.node;
  float in_scale = node_arg_get_const_data_as_float(
      *graph, *fuse_state->binder["constant_12"].node_arg);
  uint16_t in_zero_point = vaip::dd::get_zp_from_node(
      *graph, *fuse_state->binder["constant_13"].node_arg);
  auto in_shape = node_arg_get_shape_i64(*in_node.node_arg);

  if (node_found != nullptr) {
    auto op_type = VAIP_ORT_API(node_op_type)(*node_found);
    if (VAIP_ORT_API(node_op_type)(*node_found) == "IConv") {
      auto concat_attr = node_has_attr(*node_found, "concat_in_child");

      if (concat_attr &&
          node_get_attr_string(*node_found, "concat_in_child") == "true") {

        in_scale = node_get_attr_float(*node_found, "output_scale");
        in_zero_point =
            (uint16_t)(node_get_attr_float(*node_found, "output_zp"));
        concat_in_sibling = "true";
      }
    } else if (VAIP_ORT_API(node_op_type)(*node_found) == "QuantizeLinear") {
      auto quant_node_name = node_get_first_output_name(*node_found);
      auto quant_consumers = graph_get_consumer_nodes(*graph, quant_node_name);
      for (auto consumer : quant_consumers) {
        if (VAIP_ORT_API(node_op_type)(*consumer) == "DequantizeLinear") {
          auto dequant_node_name = node_get_first_output_name(*consumer);
          auto dequant_consumers =
              graph_get_consumer_nodes(*graph, dequant_node_name);
          if ((VAIP_ORT_API(node_op_type)(*dequant_consumers[0]) == "Concat")) {
            auto concat_node_name =
                node_get_first_output_name(*dequant_consumers[0]);
            auto concat_consumers =
                graph_get_consumer_nodes(*graph, concat_node_name);
            auto Q_node_input_node_args =
                node_get_input_node_args(*concat_consumers[0]);
            in_scale = node_arg_get_const_data_as_float(
                *graph, *Q_node_input_node_args[1]);
            in_zero_point =
                vaip::dd::get_zp_from_node(*graph, *Q_node_input_node_args[2]);
            concat_in_sibling = "true";
          }
        }
      }
    }
  }

  fuse_state->input_q_params.push_back(in_scale);
  fuse_state->input_q_params.push_back(float(in_zero_point));
  fuse_state->node_builder.add("concat_in_sibling", concat_in_sibling);
  fuse_state->node_builder.add("input_scale", in_scale);
  fuse_state->node_builder.add("input_zp", float(in_zero_point));
}

static void apply_qgroupnorm_hardcodings_0(FuseState* fuse_state) {
  apply_qgroupnorm_hardcodings(fuse_state);
  auto in_node = fuse_state->binder["input_0"];
  auto in_shape = node_arg_get_shape_i64(*in_node.node_arg);
  auto [nchw_shape, nhwc_shape] = get_NCHW_NHWC(*in_shape);
  std::vector<int64_t> new_out_shape{1, nhwc_shape[1] * nhwc_shape[2],
                                     nhwc_shape[3]};
  fuse_state->input_shape = nhwc_shape;
  fuse_state->output_shape = new_out_shape;
}

static void apply_qgroupnorm_hardcodings_1(FuseState* fuse_state) {
  apply_qgroupnorm_hardcodings(fuse_state);
  auto in_node = fuse_state->binder["input_0"];
  auto in_shape = node_arg_get_shape_i64(*in_node.node_arg);
  auto [nchw_shape, nhwc_shape] = get_NCHW_NHWC(*in_shape);
  fuse_state->input_shape = nhwc_shape;
  fuse_state->output_shape = nchw_shape;
}

static void set_accessor_attributes(FuseState* fuse_state) {

  auto accessor_attributes = fuse_state->pass_proto.pass_fusion_param()
                                 .extractor()
                                 .accessor_attributes();

  for (const auto& attribute_accessor : accessor_attributes) {
    auto attr_proto = node_get_attr(
        *fuse_state->binder[attribute_accessor.node_binder_name()].node,
        attribute_accessor.attribute_name());
    add_attributes(fuse_state->node_builder, attr_proto,
                   attribute_accessor.node_binder_name());
  }
}

static void set_explicit_attribute(FuseState* fuse_state) {
  auto explicit_attributes = fuse_state->pass_proto.pass_fusion_param()
                                 .extractor()
                                 .explicit_attributes();
  set_explicit_attributes(explicit_attributes, fuse_state->node_builder);
}

static void set_initializers(FuseState* fuse_state) {
  auto initializer_extractors =
      fuse_state->pass_proto.pass_fusion_param().extractor().initializers();

  for (const auto& init_name : initializer_extractors) {
    add_initializers(fuse_state->input_node_args, init_name, fuse_state->binder,
                     fuse_state->graph, fuse_state->in_dtypes);
  }
}

// Convert 2D vector of int64_t to vector of strings
// Each inner vector becomes a comma-separated string enclosed in brackets
static std::vector<std::string>
vector2dToStrings(const std::vector<std::vector<int64_t>>& vec2d) {
  std::vector<std::string> result;
  result.reserve(vec2d.size());

  for (const auto& innerVec : vec2d) {
    std::ostringstream oss;
    oss << "[";

    for (size_t i = 0; i < innerVec.size(); ++i) {
      oss << innerVec[i];
      if (i < innerVec.size() - 1) {
        oss << ",";
      }
    }

    oss << "]";
    result.push_back(oss.str());
  }

  return result;
}

static void infer_input_shapes(FuseState* fuse_state) {
  for (auto node_arg : fuse_state->input_node_args) {
    auto shape = *node_arg_get_shape_i64(*node_arg).get();
    fuse_state->input_shapes.push_back(shape);
  }
}

static void populate_node(FuseState* fuse_state) {
  fuse_state->node_builder.add("in_dtypes", fuse_state->in_dtypes);
  fuse_state->node_builder.add("out_dtypes", fuse_state->out_dtypes);

  fuse_state->node_builder.set_input_node_args(fuse_state->input_node_args);

  fuse_state->node_builder.add("generic_fusion", fuse_state->generic_fusion);

  fuse_state->node_builder.add("nodes", fuse_state->node_names);

  if (fuse_state->input_q_params.size())
    fuse_state->node_builder.add("input_q_params", fuse_state->input_q_params);
  if (fuse_state->output_q_params.size())
    fuse_state->node_builder.add("output_q_params",
                                 fuse_state->output_q_params);

  fuse_state->node_builder.add("input_shape", fuse_state->input_shape);
  fuse_state->node_builder.add("input_shapes",
                               vector2dToStrings(fuse_state->input_shapes));
  fuse_state->node_builder.add("output_shape", fuse_state->output_shape);

  fuse_state->node_builder.set_op_type(fuse_state->op_name, "com.xilinx");

  fuse_state->node_builder.set_anchor_point1(*fuse_state->out_node.node);
}

// util function for skipping 3 parents of in_node used in PSR QConv2MatMul
static std::pair<const NodeArg*, std::vector<std::string>>
find_new_input(const Node* in_node) {
  auto current_node = in_node;
  int node_cnt = 3;
  std::vector<std::string> n_names;
  const NodeArg* new_input_arg = nullptr;
  while (current_node && node_cnt > 0) {
    auto node_inputs = node_get_inputs(*current_node);
    if (node_inputs.size() > 0) {
      current_node = node_inputs[0].node;
      n_names.push_back(node_arg_get_name(*node_inputs[0].node_arg));
      if (node_cnt == 1) {
        new_input_arg = node_inputs[0].node_arg;
      }
    } else {
      break;
    }
    node_cnt--;
  }
  return {new_input_arg, n_names};
}

static void inputToThreeParentUp(FuseState* fuse_state) {
  auto [new_input_arg, extra_names] =
      find_new_input(fuse_state->binder[fuse_state->input_node_names[0]].node);
  if (new_input_arg) {
    fuse_state->input_node_args[0] = new_input_arg;
    fuse_state->in_dtypes[0] =
        vaip::dd::nodearg_dtype_to_string(*new_input_arg);
    // fuse_state->input_shape = *node_arg_get_shape_i64(*new_input_arg).get();
  }
}

static void convertIconvToQconv2Matmul(FuseState* fuse_state) {
  auto node_arg_str = fuse_state->node_names[2];
  const NodeArg* node_arg_ = fuse_state->binder[node_arg_str].node_arg;
  auto tranpose_node =
      VAIP_ORT_API(graph_producer_node)(*fuse_state->graph, node_arg_str);
  std::vector<const NodeArg*> trans_inputs =
      node_get_input_node_args(*tranpose_node);
  auto new_input_arg = trans_inputs[0];
  std::vector<int64_t> a = {1, 1, 1, 1};
  std::vector<int64_t> b = {0, 0, 0, 0};
  std::vector<int64_t> c = {1, 1};
  std::vector<int64_t> d = {3, 3};

  auto wt_shape = *node_arg_get_shape_i64(*new_input_arg).get();
  if (wt_shape.size() >= 2 && wt_shape[wt_shape.size() - 1] == 1 &&
      wt_shape[wt_shape.size() - 2] == 1) {

    fuse_state->op_name = "QConv2MatMul";
    fuse_state->node_builder.add("kernel_shape", c);
    fuse_state->node_builder.add("pads", b);
    // std::cout<<"changing the op_type\n";
  } else {
    fuse_state->node_builder.add("kernel_shape", d);
    fuse_state->node_builder.add("pads", a);
  }

  fuse_state->node_builder.add("from_iconv", "true");
  fuse_state->node_builder.add("weight_shape", wt_shape);
  fuse_state->node_builder.add("dialation", c);
  fuse_state->node_builder.add("group", (int64_t)1);
  fuse_state->node_builder.add("strides", c);
  fuse_state->node_builder.add("from_iconv", "true");
  fuse_state->node_builder.add("auto_pad", "NOTSET");
}

static std::tuple<float, uint16_t, std::string> get_concat_qparams_conv_iconv(
    onnxruntime::Graph* graph, std::string myname, float conv_outq_scale,
    uint16_t conv_outq_zero_point, std::string concat_in_child) {
  // if (graph_get_consumer_nodes(*graph,node_arg_get_name(*out_node.node_arg
  // )).size() == 1) {
  std::vector<const Node*> final_quant_node_nextnodes =
      graph_get_consumer_nodes(*graph, myname);
  for (auto consumer : final_quant_node_nextnodes) {
    std::string dq_before_concat_node_name =
        node_get_first_output_name(*consumer);
    auto dq_before_concat_node_arg =
        VAIP_ORT_API(graph_get_node_arg)(*graph, dq_before_concat_node_name);

    if (graph_get_consumer_nodes(*graph,
                                 node_arg_get_name(*dq_before_concat_node_arg))
            .size() == 1) {
      std::vector<const Node*> dq_before_concat_next_nodes =
          graph_get_consumer_nodes(
              *graph, node_arg_get_name(*dq_before_concat_node_arg));
      std::string concat_node_name =
          node_get_first_output_name(*dq_before_concat_next_nodes[0]);
      auto concat_node_arg =
          VAIP_ORT_API(graph_get_node_arg)(*graph, concat_node_name);

      auto concat_node_op_type =
          VAIP_ORT_API(node_op_type)(*dq_before_concat_next_nodes[0]);

      if (concat_node_op_type == "Concat") {
        if (graph_get_consumer_nodes(*graph,
                                     node_arg_get_name(*concat_node_arg))
                .size() == 1) {
          std::vector<const Node*> concat_node_nextnodes =
              graph_get_consumer_nodes(*graph,
                                       node_arg_get_name(*concat_node_arg));
          std::string Q_node_after_concat_name =
              node_get_first_output_name(*concat_node_nextnodes[0]);
          auto Q_node_input_node_args =
              node_get_input_node_args(*concat_node_nextnodes[0]);

          auto Q_node_output_arg =
              node_get_output_node_args(*concat_node_nextnodes[0]);
          std::string q_node_arg_name =
              node_arg_get_name(*Q_node_output_arg[0]);

          if (q_node_arg_name !=
              "/up_blocks.2/cats.2/Concat_output_0_QuantizeLinear_Output") {

            concat_in_child = "true";
            conv_outq_scale = node_arg_get_const_data_as_float(
                *graph, *Q_node_input_node_args[1]);
            conv_outq_zero_point =
                vaip::dd::get_zp_from_node(*graph, *Q_node_input_node_args[2]);
          } else {
            MY_LOG(1) << "Found the staturation point of concat update, so not "
                         "updating the parent ICONV";
          }
        }
      }
    }
  }
  return std::make_tuple(conv_outq_scale, conv_outq_zero_point,
                         concat_in_child);
}

std::tuple<float, uint16_t, std::string>
get_sibling_concat_qparams_iconv(onnxruntime::Graph* graph,
                                 const NodeInput& in_node, float in_scale,
                                 uint16_t in_zero_point) {
  auto node_found = in_node.node;
  std::string concat_in_sibling = "false";
  if (node_found != nullptr) {
    auto is_Iconv = VAIP_ORT_API(node_op_type)(*node_found);
    if (VAIP_ORT_API(node_op_type)(*node_found) == "IConv") {
      auto concat_attr = node_has_attr(*node_found, "concat_in_child");

      if (concat_attr &&
          node_get_attr_string(*node_found, "concat_in_child") == "true") {
        in_scale = node_get_attr_float(*node_found, "output_scale");
        in_zero_point =
            (uint16_t)(node_get_attr_float(*node_found, "output_zp"));
        concat_in_sibling = "true";
        MY_LOG(1) << "Conv has concat in silbling";
      }
    } else if (VAIP_ORT_API(node_op_type)(*node_found) ==
               "QuantizeLinear") { // if producer node is QuantizeLinear, as
                                   // IConv is the first PASS
      auto quant_node_name = node_get_first_output_name(*node_found);
      auto quant_consumers = graph_get_consumer_nodes(*graph, quant_node_name);
      for (auto consumer :
           quant_consumers) {     // for each consumer of QuantizeLinear
        if (VAIP_ORT_API(node_op_type)(*consumer) ==
            "DequantizeLinear") { // we dont want to go into another ICONV, so
                                  // check for Dequant here
          auto dequant_node_name = node_get_first_output_name(*consumer);
          auto dequant_consumers =
              graph_get_consumer_nodes(*graph, dequant_node_name);
          if ((VAIP_ORT_API(node_op_type)(*dequant_consumers[0]) ==
               "Concat")) { // Check if Concat is in the consumer of
                            // DequantLinear
            auto concat_node_name =
                node_get_first_output_name(*dequant_consumers[0]);
            auto concat_consumers =
                graph_get_consumer_nodes(*graph, concat_node_name);
            auto Q_node_input_node_args =
                node_get_input_node_args(*concat_consumers[0]);
            auto Q_node_output_arg =
                node_get_output_node_args(*concat_consumers[0]);
            std::string q_node_arg_name =
                node_arg_get_name(*Q_node_output_arg[0]);

            if (q_node_arg_name !=
                "/up_blocks.3/cats.0/Concat_output_0_QuantizeLinear_Output") {
              in_scale = node_arg_get_const_data_as_float(
                  *graph, *Q_node_input_node_args[1]);
              in_zero_point = vaip::dd::get_zp_from_node(
                  *graph, *Q_node_input_node_args[2]);
              concat_in_sibling = "true";

              MY_LOG(1) << "Conv node with concat sibling and Add as parent "
                           "updated here ";
            } else {
              MY_LOG(1) << "Found the saturation point node, so not updating "
                           "the sibling ICONV";
            }
          }
        }
      }
    }
  }
  return std::make_tuple(in_scale, in_zero_point, concat_in_sibling);
}

static void getChildConcatParams(FuseState* fuse_state) {
  auto output_node_argg_str = fuse_state->node_names[5];
  auto output_node_ = VAIP_ORT_API(graph_producer_node)(*fuse_state->graph,
                                                        output_node_argg_str);

  std::vector<const NodeArg*> out_inputs =
      node_get_input_node_args(*output_node_);

  auto conv_outq_scale =
      node_arg_get_const_data_as_float(*fuse_state->graph, *out_inputs[1]);
  auto conv_outq_zero_point =
      node_arg_get_const_data_as_u16(*fuse_state->graph, *out_inputs[2]);

  // qparams if concat is a consumer
  std::string concat_in_child = "false";
  auto concat_params = get_concat_qparams_conv_iconv(
      fuse_state->graph, output_node_argg_str, conv_outq_scale,
      conv_outq_zero_point, concat_in_child);

  if (std::get<2>(concat_params) == "true") {
    conv_outq_scale = std::get<0>(concat_params);
    conv_outq_zero_point = std::get<1>(concat_params);
    concat_in_child = std::get<2>(concat_params);
  }
  std::ostringstream oss1;
  oss1 << std::fixed << std::setprecision(25) << conv_outq_scale;
  fuse_state->node_builder.add("output_scale", conv_outq_scale);
  fuse_state->node_builder.add("output_scale_str", oss1.str());
  fuse_state->node_builder.add("output_zp", (float)conv_outq_zero_point);
  fuse_state->node_builder.add("concat_in_child", concat_in_child);
}

static void getSiblingConcatParams(FuseState* fuse_state) {

  auto input_node_argg_str = fuse_state->node_names[1];
  auto act_node = VAIP_ORT_API(graph_producer_node)(*fuse_state->graph,
                                                    input_node_argg_str);

  std::vector<const NodeArg*> act_inputs = node_get_input_node_args(*act_node);
  auto activation_node_inputs = node_get_inputs(*act_node);
  auto in_scale =
      node_arg_get_const_data_as_float(*fuse_state->graph, *act_inputs[1]);
  auto in_zero_point =
      node_arg_get_const_data_as_u16(*fuse_state->graph, *act_inputs[2]);

  std::string concat_in_sibling = "false";
  auto node_name_aks = node_arg_get_name(*activation_node_inputs[0].node_arg);
  auto sibling_concat_params = get_sibling_concat_qparams_iconv(
      fuse_state->graph, activation_node_inputs[0], in_scale, in_zero_point);

  if (std::get<2>(sibling_concat_params) == "true") {
    in_scale = std::get<0>(sibling_concat_params);
    in_zero_point = std::get<1>(sibling_concat_params);
    concat_in_sibling = std::get<2>(sibling_concat_params);
    // std::cout<<"input "<<in_scale<<"  "<<in_zero_point<<"
    // "<<concat_in_sibling<<std::endl;
  }
  fuse_state->node_builder.add("input_scale", (float)in_scale);
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(25) << in_scale;
  fuse_state->node_builder.add("input_scale_str", oss.str());
  fuse_state->node_builder.add("concat_in_sibling", concat_in_sibling);
  fuse_state->node_builder.add("input_zp", (float)in_zero_point);
}

static void infer_input_node_args(FuseState* fuse_state) {
  for (const std::string& node_name : fuse_state->input_node_names) {
    const NodeArg* node_arg_ = fuse_state->binder[node_name].node_arg;
    if (node_arg_) {
      fuse_state->input_node_args.push_back(node_arg_);
    }
  }
}

static void infer_input_dtypes(FuseState* fuse_state) {
  for (const NodeArg* node_arg_ : fuse_state->input_node_args) {
    fuse_state->in_dtypes.push_back(
        vaip::dd::nodearg_dtype_to_string(*node_arg_));
  }
}

static void infer_output_dtypes(FuseState* fuse_state) {
  fuse_state->out_dtypes.push_back(
      vaip::dd::nodearg_dtype_to_string(*fuse_state->out_node.node_arg));
}

static void infer_input_shape(FuseState* fuse_state) {
  auto temp = fuse_state->input_node_args[0];
  fuse_state->input_shape = *node_arg_get_shape_i64(*temp).get();
}

static void infer_output_shape(FuseState* fuse_state) {
  fuse_state->output_shape =
      *node_arg_get_shape_i64(*fuse_state->out_node.node_arg).get();
}

static void infer_io_qparams(FuseState* fuse_state) {
  auto input_q_params_extractors =
      fuse_state->pass_proto.pass_fusion_param().extractor().input_q_params();
  auto output_q_params_extractors =
      fuse_state->pass_proto.pass_fusion_param().extractor().output_q_params();

  fuse_state->input_q_params = get_q_params(
      input_q_params_extractors, fuse_state->graph, fuse_state->binder);
  fuse_state->output_q_params = get_q_params(
      output_q_params_extractors, fuse_state->graph, fuse_state->binder);
}

static void print_op_name(FuseState* fuse_state) {
  std::cout << "op name : " << fuse_state->op_name << std::endl;
}

static void task_runner(FuseState* fuse_state,
                        std::vector<std::string>& modifiers) {
  std::unordered_map<std::string, std::function<void(FuseState*)>>
      function_registry;

  // list of predicates
  function_registry["quantop_predicate"] = quantop_predicate;
  function_registry["dequantop_predicate"] = dequantop_predicate;
  function_registry["mhamzdk5_predicate"] = mhamzdk5_predicate;
  function_registry["QBatchMatMul_predicate"] = QBatchMatMul_predicate;

  // list of op specific hardcoding modifiers
  function_registry["qslice_modifier"] = qslice_modifier;
  function_registry["mhamzdk5_modifier"] = mhamzdk5_modifier;

  // list of generic modifiers
  function_registry["in_dtype_modifier"] = in_dtype_modifier;
  function_registry["out_dtype_modifier"] = out_dtype_modifier;
  function_registry["apply_qgroupnorm_hardcodings_0"] =
      apply_qgroupnorm_hardcodings_0;
  function_registry["apply_qgroupnorm_hardcodings_1"] =
      apply_qgroupnorm_hardcodings_1;
  function_registry["psw_prepro_hardcoding"] = psw_prepro_hardcoding;

  function_registry["infer_input_node_args"] = infer_input_node_args;
  function_registry["infer_input_dtypes"] = infer_input_dtypes;
  function_registry["inputToThreeParentUp"] = inputToThreeParentUp;
  function_registry["convertIconvToQconv2Matmul"] = convertIconvToQconv2Matmul;
  function_registry["getChildConcatParams"] = getChildConcatParams;
  function_registry["getSiblingConcatParams"] = getSiblingConcatParams;
  function_registry["infer_output_dtypes"] = infer_output_dtypes;
  function_registry["infer_input_shape"] = infer_input_shape;
  function_registry["infer_output_shape"] = infer_output_shape;
  function_registry["infer_io_qparams"] =
      infer_io_qparams; // TODO: Separate out input and output qparams
  function_registry["infer_input_shapes"] = infer_input_shapes;

  function_registry["set_initializers"] =
      set_initializers; // TODO: this can be moved to fuse state constructor.
                        // PS: this is infer not set type of modifier

  function_registry["set_accessor_attributes"] = set_accessor_attributes;
  function_registry["set_explicit_attribute"] = set_explicit_attribute;
  function_registry["extract_subgraph"] = extract_subgraph;
  function_registry["populate_node"] = populate_node;

  function_registry["print_op_name"] = print_op_name;

  for (std::string modifier : modifiers) {
    auto it = function_registry.find(modifier);
    if (it != function_registry.end()) {
      it->second(fuse_state);
    } else {
      // std::cout << "Unknown modifier : " + modifier << std::endl;
      // throw std::runtime_error("Unknown modifier : "+modifier);
    }
  }
}

} // namespace
