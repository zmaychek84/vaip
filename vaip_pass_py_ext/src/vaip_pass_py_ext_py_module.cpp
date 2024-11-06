/*
 *     The Xilinx Vitis AI Vaip in this distribution are provided under the
 * following free and permissive binary-only license, but are not provided in
 * source code form.  While the following free and permissive license is similar
 * to the BSD open source license, it is NOT the BSD open source license nor
 * other OSI-approved open source license.
 *
 *      Copyright (C) 2022 Xilinx, Inc. All rights reserved.
 *      Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights
 * reserved.
 *
 *      Redistribution and use in binary form only, without modification, is
 * permitted provided that the following conditions are met:
 *
 *      1. Redistributions must reproduce the above copyright notice, this list
 * of conditions and the following disclaimer in the documentation and/or other
 * materials provided with the distribution.
 *
 *      2. The name of Xilinx, Inc. may not be used to endorse or promote
 * products redistributed with this software without specific prior written
 * permission.
 *
 *      THIS SOFTWARE IS PROVIDED BY XILINX, INC. "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL XILINX, INC. BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 *      PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE
 */
// clang-format off
#include "./common.hpp"
#include "./util.hpp"
#include "vaip/anchor_point.hpp"
#include "vaip/vaip_ort.hpp" // set_the_global_api
#include <vaip/my_ort.h>
// clang-format on
namespace {
using namespace vaip_core;

static std::shared_ptr<Pattern> parse_pattern(py::object pattern_def) {
  auto builder = PatternBuilder();
  return parse_pattern0(builder, pattern_def);
}

static bool is_anchor_point(const py::object& anchor_point_json) {
  auto m = py::module::import("voe.anchor_point");
  auto is_anchor_point_f = m.attr("is_anchor_point");
  return py::cast<bool>(is_anchor_point_f(anchor_point_json));
}
static bool as_boolean(py::object value) {
  bool ret = false;
  if (py::isinstance<py::bool_>(value)) {
    ret = py::cast<bool>(value);
  } else if (value.is_none()) {
    ret = false;
  } else {
    ret = true;
  }
  return ret;
}
PYBIND11_MODULE(voe_cpp2py_export, m) {
  m.doc() = "vaip pass python extension";
  py::class_<ModelWrapper>(m, "ModelWrapper")
      .def("get_main_graph", [](ModelWrapper& model) {
        return GraphWrapper{VAIP_ORT_API(model_main_graph)(*model.model)};
      });
  m.def("model_load", [](const std::string& filename) {
    auto model_ptr = model_load(filename);
    return ModelWrapper{std::move(model_ptr)};
  });
  py::class_<Rule>(m, "Rule");
  py::class_<Binder>(m, "Binder");
  py::class_<PatternBuilder>(m, "PatternBuilder")
      .def(py::init<>())
      .def("create_by_py", &PatternBuilder::create_by_py)
      .def("create_by_json", &PatternBuilder::create_by_json);
  py::class_<NodeBuilder, std::shared_ptr<NodeBuilder>>(m, "NodeBuilder")
      .def("set_input_args",
           [](NodeBuilder* self, const std::vector<NodeArgWrapper>& args) {
             auto args2 = std::vector<const NodeArg*>(args.size());
             for (auto i = 0u; i < args.size(); ++i) {
               args2[i] = &args[i].node_arg;
             }
             return &self->set_input_node_args(args2);
           })
      .def("set_input_nodes", // TO BE REMOVED.
           [](NodeBuilder* self, const std::vector<NodeWrapper>& args) {
             auto args2 = std::vector<const Node*>(args.size());
             for (auto i = 0u; i < args.size(); ++i) {
               args2[i] = &args[i].node;
             }
             return &self->set_input_nodes(args2);
           })
      .def("clone_inputs",
           [](NodeBuilder* self, NodeWrapper node) {
             return &self->clone_inputs(node.node);
           })
      .def("set_op_type",
           [](NodeBuilder* self, const std::string& op_type,
              const std::string& domain) {
             return &self->set_op_type(op_type, domain);
           })
      .def("clone_op_type",
           [](NodeBuilder* self, NodeWrapper node) {
             return &self->clone_op_type(node.node);
           })
      .def("clone_attrs",
           [](NodeBuilder* self, NodeWrapper node) {
             return &self->clone_attrs(node.node);
           })
      .def("set_attr",
           [](NodeBuilder* self, const std::string& name,
              py::object attr_value) {
             if (py::isinstance<py::list>(attr_value)) {
               std::vector<int64_t> data;
               for (auto item : attr_value) {
                 if (!py::isinstance<py::int_>(item))
                   LOG(FATAL) << "TODO: Unknown: " << py::str(item)
                              << " :: " << py::str(item.get_type())
                              << " attr_name " << name;
                 else
                   data.push_back(static_cast<int64_t>(py::cast<int>(item)));
               }
               self->add(name, data);
             } else if (py::isinstance<py::int_>(attr_value)) {
               self->add(name, static_cast<int64_t>(py::cast<int>(attr_value)));
             } else if (py::isinstance<py::float_>(attr_value)) {
               self->add(name, static_cast<float>(py::cast<float>(attr_value)));
             } else if (py::isinstance<py::str>(attr_value)) {
               if (name == "data_type") {
                 self->set_data_type(py::str(attr_value));
               } else {
                 self->add(name, py::str(attr_value));
               }
             } else {
               LOG(FATAL) << "TODO: Unknown: " << py::str(attr_value)
                          << " :: " << py::str(attr_value.get_type());
             }
             return self;
           })
      .def("add_int", // TODO: remove it
           [](NodeBuilder* self, const std::string& name,
              const int64_t& value) { return &self->add(name, value); })
      .def("add_ints", // TODO: remove it
           [](NodeBuilder* self, const std::string& name,
              const std::vector<int64_t>& value) {
             return &self->add(name, value);
           })
      .def("add_string", // TODO: remove it
           [](NodeBuilder* self, const std::string& name,
              const std::string& value) {
             if (name == "data_type") {
               return &self->set_data_type(value);
             }
             return &self->add(name, value);
           })
      .def("set_shape",
           [](NodeBuilder* self, const std::vector<int64_t>& shape) {
             return &self->set_shape(gsl::span<const int64_t>(shape));
           })
      .def("clone_shape",
           [](NodeBuilder* self, NodeInput ni) {
             if (ni.node) {
               return &self->clone_shape(*ni.node);
             } else {
               return &self->clone_shape(*ni.node_arg);
             }
           })
      .def("clone_data_type",
           [](NodeBuilder* self, NodeInput ni) {
             if (ni.node) {
               return &self->clone_data_type(*ni.node);
             } else {
               return &self->clone_data_type(*ni.node_arg);
             }
           })
      .def("set_data_type", &NodeBuilder::set_data_type)
      .def("set_anchor_point1",
           [](NodeBuilder* self, NodeWrapper node) {
             return &self->set_anchor_point1(node.node);
           })
      .def("set_anchor_point_node_arg1",
           [](NodeBuilder* self, NodeArgWrapper node_arg) {
             return &self->set_anchor_point1(node_arg.node_arg);
           })
      .def("set_anchor_point2",
           [](NodeBuilder* self, NodeInput ni,
              const py::object& anchor_point_json) {
             // TODO check is_anchor_point
             CHECK(is_anchor_point(anchor_point_json))
                 << "invaild anchor point json : " << anchor_point_json;
             if (ni.node)
               return &self->set_anchor_point2(
                   node_get_output_node_arg(*ni.node),
                   AnchorPoint::Description::create_by_json(
                       py::cast<std::string>(anchor_point_json)));
             else
               return &self->set_anchor_point2(
                   *ni.node_arg, AnchorPoint::Description::create_by_json(
                                     py::cast<std::string>(anchor_point_json)));
           })
      .def("set_anchor_point3",
           [](NodeBuilder* self, NodeWrapper node,
              const py::object& anchor_point_json,
              const std::vector<int64_t>& shape) {
             CHECK(is_anchor_point(anchor_point_json))
                 << "invaild anchor point json : " << anchor_point_json;
             return &self->set_anchor_point3(
                 node_get_output_node_arg(node.node),
                 AnchorPoint::Description::create_by_json(
                     py::cast<std::string>(anchor_point_json)),
                 shape);
           })
      .def("build", [](NodeBuilder* self) {
        return NodeWrapper{const_cast<Node&>(self->build())};
      });
  py::class_<GraphWrapper>(m, "GraphWrapper")
      .def("builder",
           [](GraphWrapper graph, IPass* self) {
             return std::make_shared<NodeBuilder>(graph.graph, *self);
           })
      .def("resolve", [](GraphWrapper& graph,
                         bool force) { graph_resolve(graph.graph, force); })
      .def("get_node_in_topoligical_order",
           [](GraphWrapper& graph) {
             return graph_get_node_in_topoligical_order(graph.graph);
           })
      .def("get_node", [](GraphWrapper& graph, int index) {
        Node* node =
            const_cast<Node*>(VAIP_ORT_API(graph_get_node)(graph.graph, index));
        return NodeWrapper{*node};
      });
  py::class_<NodeInput>(m, "NodeInput")
      // i.def(py::init<const Node*, const NodeArg*>())
      .def("node",
           [](NodeInput ni) {
             py::object ret = py::none();
             if (ni.node != nullptr) {
               ret = py::cast(NodeWrapper{const_cast<Node&>(*ni.node)});
             }
             return ret;
           })
      .def("node_arg",
           [](NodeInput ni) {
             return NodeArgWrapper{const_cast<NodeArg&>(*ni.node_arg)};
           })
      .def("empty", [](const NodeInput& ni) { return !ni.is_matched(); })
      .def("const_data",
           [](const NodeInput& ni, IPass* pass,
              GraphWrapper graph) -> py::object {
             auto ret = py::object();
             if (ni.node != nullptr) {
               auto data_type = node_get_output_element_type(*ni.node);
               if (data_type == onnx::TensorProto_DataType_FLOAT) {
                 auto const_data = pass->get_const_data<float>(*ni.node);
                 auto value =
                     std::vector<float>(const_data.begin(), const_data.end());
                 ret = py::cast(value);
               } else if (data_type == onnx::TensorProto_DataType_UINT16) {
                 auto const_data = pass->get_const_data<uint16_t>(*ni.node);
                 auto value = std::vector<uint16_t>(const_data.begin(),
                                                    const_data.end());
                 ret = py::cast(value);
               } else if (data_type == onnx::TensorProto_DataType_INT16) {
                 auto const_data = pass->get_const_data<int16_t>(*ni.node);
                 auto value =
                     std::vector<int16_t>(const_data.begin(), const_data.end());
                 ret = py::cast(value);
               } else if (data_type == onnx::TensorProto_DataType_INT64) {
                 auto const_data = pass->get_const_data<int64_t>(*ni.node);
                 auto value =
                     std::vector<int64_t>(const_data.begin(), const_data.end());
                 ret = py::cast(value);
               } else {
                 LOG(FATAL) << "not supported data_type : " << data_type;
               }
             } else {
               auto data_type = node_arg_get_element_type(*ni.node_arg);
               auto& tensor =
                   node_arg_get_const_data_as_tensor(graph.graph, *ni.node_arg);
               if (data_type == onnx::TensorProto_DataType_FLOAT) {
                 auto const_data = tensor_proto_as_floats(graph.graph, tensor);
                 auto value =
                     std::vector<float>(const_data.begin(), const_data.end());
                 ret = py::cast(value);
               } else if (data_type == onnx::TensorProto_DataType_INT8) {
                 auto const_data = tensor_proto_as_i8s(graph.graph, tensor);
                 auto value =
                     std::vector<int8_t>(const_data.begin(), const_data.end());
                 ret = py::cast(value);
               } else if (data_type == onnx::TensorProto_DataType_UINT8) {
                 auto const_data = tensor_proto_as_u8s(graph.graph, tensor);
                 auto value =
                     std::vector<uint8_t>(const_data.begin(), const_data.end());
                 ret = py::cast(value);
               } else if (data_type == onnx::TensorProto_DataType_UINT16) {
                 auto const_data = tensor_proto_as_u16s(graph.graph, tensor);
                 auto value = std::vector<uint16_t>(const_data.begin(),
                                                    const_data.end());
                 ret = py::cast(value);
               } else if (data_type == onnx::TensorProto_DataType_INT16) {
                 auto const_data = tensor_proto_as_i16s(graph.graph, tensor);
                 auto value =
                     std::vector<int16_t>(const_data.begin(), const_data.end());
                 ret = py::cast(value);
               } else {
                 LOG(FATAL) << "not supported data_type : " << data_type;
               }
             }
             return ret;
           })

      .def("data_type",
           [](NodeInput ni) {
             return data_type_to_string(
                 node_arg_get_element_type(*ni.node_arg));
           })
      .def("__str__", [](const NodeInput& ni) {
        std::ostringstream str;
        if (ni.node == nullptr) {
          str << node_arg_as_string(*ni.node_arg);
        } else {
          str << node_as_string(*ni.node);
        }
        return str.str();
      });
  py::class_<NodeWrapper>(m, "Node")
      .def("op_type",
           [](const NodeWrapper& self) {
             return VAIP_ORT_API(node_op_type)(self.node);
           })
      .def("as_node_input",
           [](const NodeWrapper& self) {
             auto node_arg = node_get_output_node_args(self.node);
             CHECK(node_arg.size() == 1);
             return NodeInput{&self.node, node_arg[0]};
           })
      .def("inputs",
           [](const NodeWrapper& n) { return node_get_inputs(n.node); })
      .def("outputs",
           [](const NodeWrapper& n) {
             auto ret = std::vector<NodeInput>{};
             for (auto arg : node_get_output_node_args(n.node)) {
               ret.emplace_back(
                   NodeInput{const_cast<const Node*>(&n.node), arg});
             }
             return ret;
           })
      .def("has_attr",
           [](const NodeWrapper& self, const std::string& attr_name) -> bool {
             return node_has_attr(self.node, attr_name);
           })
      .def("attr",
           [](const NodeWrapper& self,
              const std::string& attr_name) -> py::object {
             auto attr_value = node_get_attr(self.node, attr_name);
             auto attr_type = VAIP_ORT_API(attr_proto_get_type)(*attr_value);
             auto ret = py::object();
             if (attr_type == onnx::AttributeProto_AttributeType_INT) {
               auto value = VAIP_ORT_API(attr_proto_get_int)(*attr_value);
               ret = py::cast(value);
             } else if (attr_type == onnx::AttributeProto_AttributeType_FLOAT) {
               auto value = VAIP_ORT_API(attr_proto_get_float)(*attr_value);
               ret = py::cast(value);
             } else if (attr_type == onnx::AttributeProto_AttributeType_INTS) {
               auto value = VAIP_ORT_API(attr_proto_get_ints)(*attr_value);
               py::list ret_list;
               for (auto v : value) {
                 ret_list.append(v);
               }
               ret = ret_list;
             } else if (attr_type ==
                        onnx::AttributeProto_AttributeType_FLOATS) {
               auto value = VAIP_ORT_API(attr_proto_get_floats)(*attr_value);
               py::list ret_list;
               for (auto v : value) {
                 ret_list.append(v);
               }
               ret = ret_list;
             } else if (attr_type ==
                        onnx::AttributeProto_AttributeType_STRINGS) {
               auto value = VAIP_ORT_API(attr_proto_get_strings)(*attr_value);
               py::list ret_list;
               for (auto v : value) {
                 ret_list.append(v);
               }
               ret = ret_list;
             } else {
               LOG(FATAL) << "TODO: attr_type: " << attr_type
                          << "  ::attr_name " << attr_name;
             }
             return ret;
           })
      .def("create_const",
           [](const NodeWrapper& self, IPass* pass,
              const py::object& obj) -> void {
             auto data_type = node_get_output_element_type(self.node);
             gsl::span<char> span_data;
             if (py::isinstance<py::list>(obj)) {
               std::vector<float> data;
               std::vector<int32_t> data_int32;
               std::vector<int8_t> data_int8;
               switch (data_type) {
               case 1: // ONNX_NAMESPACE::TensorProto::FLOAT
                 data = py::cast<std::vector<float>>(obj);
                 span_data = gsl::span<char>((char*)data.data(),
                                             data.size() * sizeof(float));
                 break;
               case 6: // ONNX_NAMESPACE::TensorProto::INT32
                 data_int32 = py::cast<std::vector<int32_t>>(obj);
                 span_data =
                     gsl::span<char>((char*)data_int32.data(),
                                     data_int32.size() * sizeof(int32_t));
                 break;
               case 3: // ONNX_NAMESPACE::TensorProto::INT8
                 data_int8 = py::cast<std::vector<int8_t>>(obj);
                 span_data = gsl::span<char>((char*)data_int8.data(),
                                             data_int8.size() * sizeof(int8_t));
                 break;
               default:
                 LOG(FATAL)
                     << "create_const not supported! data_type " << data_type;
               }
               pass->create_const(self.node, span_data);
             } else if (py::isinstance<py::float_>(obj)) {
               auto data = py::cast<float>(obj);
               span_data = gsl::span<char>((char*)&data, sizeof(data));
               pass->create_const(self.node, span_data);
             } else {
               LOG(FATAL) << "create_const not supported! obj type "
                          << py::str(obj.get_type().attr("__name__"));
             }
           })
      .def("attr_int", // TODO: remove this function
           [](const NodeWrapper& n, const std::string& attr_name) {
             return node_get_attr_int(n.node, attr_name);
           })
      .def("attr_ints", // TODO: remove this function
           [](const NodeWrapper& n, const std::string& attr_name) {
             auto attr = node_get_attr_ints(n.node, attr_name);
             return std::vector<int64_t>(attr.begin(), attr.end());
           })
      .def("attr_float", // TODO: remove this function
           [](const NodeWrapper& n, const std::string& attr_name) {
             return node_get_attr_float(n.node, attr_name);
           })
      .def("get_const_data_floats", // TODO: remove this function
           [](const NodeWrapper& n, IPass* pass) {
             auto const_data = pass->get_const_data<float>(n.node);
             return std::vector<float>(const_data.begin(), const_data.end());
           })
      .def("__str__",
           [](const NodeWrapper& n) { return node_as_string(n.node); });
  py::class_<NodeArgWrapper>(m, "NodeArg")
      .def("shape",
           [](const NodeArgWrapper& node_arg) {
             auto shape = node_arg_get_shape_i64(node_arg.node_arg);
             CHECK(shape != nullptr)
                 << node_arg_as_string(node_arg.node_arg) << " shape absent";
             return std::vector<int64_t>(shape->begin(), shape->end());
           })
      .def("is_unknown_shape",
           [](const NodeArgWrapper& node_arg) {
             return node_arg_is_unknown_shape(node_arg.node_arg);
           })
      .def("is_dynamic_shape",
           [](const NodeArgWrapper& node_arg) {
             return node_arg_is_dynamic_shape(node_arg.node_arg);
           })
      .def("consumers",
           [](const NodeArgWrapper& self,
              const GraphWrapper& graph) -> py::object {
             auto& arg_name = node_arg_get_name(self.node_arg);
             auto node_ptrs = graph_get_consumer_nodes(graph.graph, arg_name);

             std::vector<NodeInput> res;
             for (auto& n : node_ptrs) {
               auto node_arg = node_get_output_node_args(*n);
               res.push_back(NodeInput{n, node_arg[0]});
             }
             return py::cast(res);
           })
      .def("is_constant",
           [](const NodeArgWrapper& self, const GraphWrapper& graph) {
             return node_arg_is_constant(graph.graph, self.node_arg);
           })
      .def("name",
           [](const NodeArgWrapper& self) {
             return node_arg_get_name(self.node_arg);
           })
      .def("data_type",
           [](const NodeArgWrapper& self) {
             return data_type_to_string(
                 node_arg_get_element_type(self.node_arg));
           })
      .def("__str__", [](const NodeArgWrapper& node_arg) {
        return node_arg_as_string(node_arg.node_arg);
      });
  py::class_<MetaDefProto, std::unique_ptr<MetaDefProto>>(m, "MetaDefProto")
      .def(
          "set_generic_param",
          [](MetaDefProto* meta_def, const std::string& key,
             const std::string& value) {
            meta_def->mutable_generic_param()->insert({key, value});
          },
          "Add an entry to generic_param with passed key and value.\n"
          "\n"
          "Parameters:\n"
          "meta_def(meta_def*): The pointer of meta_def proto created after "
          "calling try_fuse.\n"
          "key(str): The key of the entry.\n"
          "value(str): The value of the entry, it is normally a file path.")
      .def("get_outputs", [](MetaDefProto* meta_def, GraphWrapper& graph) {
        py::list ret_list;
        for (const std::string& output : meta_def->outputs()) {
          auto* node_arg =
              VAIP_ORT_API(graph_get_node_arg)(graph.graph, output);
          ret_list.append(NodeArgWrapper{const_cast<NodeArg&>(*node_arg)});
        }
        return ret_list;
      });
  py::class_<IPass>(m, "Pass")
      .def("fuse",
           [](IPass& self, GraphWrapper& graph, MetaDefProto* meta_def) {
             return NodeWrapper{const_cast<Node&>(
                 self.fuse(graph.graph, MetaDefProto(*meta_def)))};
           })
      .def("try_fuse",
           [](IPass& self, GraphWrapper& graph, const std::string& name,
              const std::vector<NodeInput>& inputs,
              const std::vector<NodeInput>& outputs,
              const std::vector<std::string>& constant_initializers1,
              const std::string& device) -> std::unique_ptr<MetaDefProto> {
             auto inputs_name = std::vector<std::string>();
             inputs_name.reserve(inputs.size());
             for (const auto& input : inputs) {
               CHECK(input.node_arg != nullptr);
               inputs_name.push_back(node_arg_get_name(*input.node_arg));
             }
             auto outputs_name = std::vector<std::string>();
             outputs_name.reserve(outputs.size());
             for (const auto& output : outputs) {
               CHECK(output.node_arg != nullptr);
               outputs_name.push_back(node_arg_get_name(*output.node_arg));
             }
             return self
                 .try_fuse(graph.graph, name, inputs_name, outputs_name,
                           constant_initializers1, device)
                 .first;
           })
      .def("has_session_option",
           [](IPass& self, const std::string& option) {
             const auto& session_option =
                 self.get_config_proto().provider_options();
             return session_option.find(option) != session_option.end();
           })
      .def("get_session_option",
           [](IPass& self, const std::string& option) {
             const auto& session_option =
                 self.get_config_proto().provider_options();
             return session_option.find(option)->second;
           })
      .def("get_cache_dir",
           [](IPass& self) { return self.get_cache_file_name("").string(); })
      .def("get_supported_ops", [](IPass& self) {
        std::vector<std::string> supported_ops;
        for (auto& pass_conf :
             self.get_context()->get_config_proto().passes()) {
          if (pass_conf.name() == "fuse_DPU") {
            for (auto op : pass_conf.pass_dpu_param().supported_op()) {
              supported_ops.push_back(op);
            }
          }
        }
        return py::cast(supported_ops);
      });

  py::class_<Pattern, std::shared_ptr<Pattern>>(m, "Pattern")
      .def_static("parse", parse_pattern)
      .def("__str__", &Pattern::debug_string)
      .def("match", [](Pattern& self, const GraphWrapper& graph,
                       const NodeWrapper& node) {
        return self.match(graph.graph, node.node);
      });
  m.def("rule", [](py::object pattern, py::function action) {

  });
  m.def("log", [](const std::string& msg) { LOG(INFO) << msg; });
  m.def("_init_vaip_ort_api", [](py::capsule api) {
    vaip_core::set_the_global_api(api.get_pointer<OrtApiForVaip>());
  });
  m.def("_process_graph", [](py::capsule graph_v, py::capsule pass_v) {
    IPass* pass = pass_v.get_pointer<IPass>();
    auto& py_ext_proto = pass->get_pass_proto().py_ext();
    auto m = py::module::import(py_ext_proto.module_name().c_str());
    auto create_rules = m.attr(py_ext_proto.method_name().c_str());
    py::object rules1 = create_rules();
    CHECK(py::isinstance<py::list>(rules1))
        << py_ext_proto.module_name() << "." << py_ext_proto.method_name()
        << " must return a list";
    py::list rules = rules1;
    std::vector<std::unique_ptr<BaseRule>> cpp_rules;
    auto is_rule = py::module::import("voe.rule_ext").attr("is_rule");
    for (auto i = 0u; i < rules.size(); ++i) {
      auto rule = rules[i];
      CHECK(is_rule(rule)) << " rule[" << i << "] must be a vaip.rule.Rule";
      py::object pattern_def = rule.attr("pattern")();
      py::object action = rule.attr("_action");
      py::object where = rule.attr("_where");
      auto builder = PatternBuilder();
      auto pattern = parse_pattern0(builder, pattern_def);
      auto bindings = builder.bindings();
      cpp_rules.push_back(Rule::create_rule(
          pattern,
          [rule, where, action, bindings, pass](onnxruntime::Graph* graph,
                                                binder_t& binder) -> bool {
            py::dict py_binder;
            for (auto& binding : bindings) {
              py_binder[binding.first.c_str()] =
                  py::cast(binder[binding.second]);
            }
            rule.attr("initialize")(GraphWrapper{*graph}, pass, py_binder);
            py::object py_ret = where(**py_binder);
            if (!as_boolean(py_ret)) {
              return false;
            }
            py_ret = action(**py_binder);
            return as_boolean(py_ret);
          }));
    }
    auto chain = BaseRule::create_rule_chain(std::move(cpp_rules));
    auto graph = graph_v.get_pointer<onnxruntime::Graph>();
    chain->apply(graph);
  });
}

} // namespace
