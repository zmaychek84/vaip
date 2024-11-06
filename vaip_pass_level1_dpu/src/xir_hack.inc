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
#include <vaip/vaip.hpp>

#include "vaip/xir_headers.hpp"

namespace xir {
class XIR_DLLESPEC OpDefFactoryImp : public OpDefFactory {
public:
  void register_h(const OpDef& def) override;
  const OpDef* create(const std::string& type) const;
  const std::vector<std::string> get_registered_ops() const;
  const OpDef* get_op_def(const std::string& type,
                          bool register_custome_op_if_not_exists = true);
  const OpDef& get_const_op_def(const std::string& type) const;

private:
  void register_customized_operator_definition(const std::string& type);

private:
#ifdef _WIN32
#  pragma warning(push)
#  pragma warning(disable : 4251)
#endif
  std::unordered_map<std::string, OpDef> store_;
#ifdef _WIN32
#  pragma warning(pop)
#endif
};

XIR_DLLESPEC OpDefFactoryImp* op_def_factory();

} // namespace xir

namespace vaip_core {
const xir::OpDef& get_const_op_def(const std::string& type) {
  return xir::op_def_factory()->get_const_op_def(type);
}
} // namespace vaip_core

/// begin function declaration
static std::map<std::string, std::vector<int>>
get_map_index_from_op_def(const xir::OpDef& opdef, size_t num_of_args);

/// begin function implementation.
static std::map<std::string, std::vector<xir::Op*>>
get_ops_map(const xir::Graph* g, const xir::OpDef& opdef,
            const std::vector<NodeArg*>& args) {
  // Onnx does not support varadic parameters, it mimic such
  // feature with a single
  auto ret = std::map<std::string, std::vector<xir::Op*>>{};
  auto map_index = get_map_index_from_op_def(opdef, args.size());
  for (auto& index : map_index) {
    auto arg_name = index.first;
    auto arg_index = index.second;
    ret[arg_name] = std::vector<xir::Op*>{};
    auto& arg_ops = ret[arg_name];
    for (auto i : arg_index) {
      CHECK(args[i] != nullptr);
      if (!vaip_core::node_arg_exists(*args[i])) {
        continue;
      }
      auto& name = vaip_core::node_arg_get_name(*args[i]);
      auto xir_op = const_cast<xir::Op*>(g->get_op(name));
      // CHECK(xir_op != nullptr) << "TODO: no such op, name=" << name;
      if (xir_op != nullptr) {
        arg_ops.push_back(xir_op);
      } else {
        LOG(INFO) << "no such op, name = " << name;
      }
    }
  }
  return ret;
}

static std::map<std::string, std::vector<int>>
get_map_index_from_op_def(const xir::OpDef& opdef, size_t num_of_args) {
  if (opdef.name() == "matmul") {
    auto matmul = std::map<std::string, std::vector<int>>();
    matmul["input"] = std::vector<int>{0, 1};
    if (num_of_args == 3) {
      matmul["bias"] = std::vector<int>{2};
    }
    return matmul;
  }
  auto ret = std::map<std::string, std::vector<int>>{};
  auto input_args = opdef.input_args();
  auto index = 0u;
  // REQUIRED
  for (index = 0u; index < num_of_args &&       //
                   index < input_args.size() && //
                   (input_args[index].occur_type == xir::OpArgDef::REQUIRED);
       ++index) {
    auto& arg_def = input_args[index];
    auto arg_name = arg_def.name;
    ret[arg_name] = std::vector<int>{(int)index};
  }
  // Optional
  for (; index < num_of_args && index < input_args.size() &&
         (input_args[index].occur_type == xir::OpArgDef::OPTIONAL);
       ++index) {
    auto& arg_def = input_args[index];
    auto arg_name = arg_def.name;
    ret[arg_name] = std::vector<int>{(int)index};
  }
  // repeated.
  if (index < num_of_args && index == input_args.size() - 1 &&
      (input_args[index].occur_type == xir::OpArgDef::REPEATED ||
       input_args[index].occur_type == xir::OpArgDef::REQUIRED_AND_REPEATED)) {
    auto& arg_def = input_args[index];
    auto arg_name = arg_def.name;
    ret[arg_name] = std::vector<int>{};
    auto& index_out = ret[arg_name];
    for (; index < num_of_args; ++index) {
      index_out.push_back((int)(index));
    }
  }
  CHECK_EQ(index, num_of_args) << " all args must be processed, please check " << opdef.name()<<
                                  " opdef. I know it is not easy.";
  for (; index < input_args.size(); ++index) {
    auto& arg_def = input_args[index];
    CHECK(arg_def.occur_type != xir::OpArgDef::REQUIRED)
        << "LOGICAL ERROR:"
        << " I know it is not easy";
  }
  return ret;
}

namespace vaip {
std::string occur_type_to_string(int type) {
  switch (type) {
  case xir::OpArgDef::REQUIRED:
    return "REQUIRED";
    /// Never or once
  case xir::OpArgDef::OPTIONAL:
    return "OPTIONAL";
    /// No limitation
  case xir::OpArgDef::REPEATED:
    return "REPEATED";
    /// At least once
  case xir::OpArgDef::REQUIRED_AND_REPEATED:
    return "REQUIRED_AND_REPEATED";
  default:
    break;
  }
  return "unkknow";
}
void write_stream(std::ostream& str, const xir::AttrDef& attr_def) {
  str << "\t" << attr_def.name << " "
      << occur_type_to_string(attr_def.occur_type) //
      << "\n";
  str << "\t" << attr_def.annotation << "\n\n";
}

void write_stream(std::ostream& str, const xir::OpArgDef& arg_def) {
  str << "\t" << arg_def.name
      << " :: " << xir::DataType(arg_def.data_type, 32).to_string() << " "
      << occur_type_to_string(arg_def.occur_type) //
      << "\n";
  str << "\t" << arg_def.annotation << "\n\n";
}

void write_stream(std::ostream& str, const xir::OpDef& def) {
  str << "====================== " << def.name() << " =================="
      << "\n";
  str << "op: " << def.name() << '\n';
  str << "doc: " << def.annotation() << "\n";
  str << "args: \n";
  for (auto& arg : def.input_args()) {
    write_stream(str, arg);
  }
  str << "attrs: \n";
  for (auto& attr : def.attrs()) {
    write_stream(str, attr);
  }
}

static std::string convert_to_onnx_name_convetion(const std::string& name) {
  std::string ret = name;
  std::replace(ret.begin(), ret.end(), '-', '_');
  return ret;
}

static std::string stringify(const std::string& s) {
  std::ostringstream str;
  str << '"';
  for (auto x : s) {
    if (x == '"') {
      str << "\\\"";
    } else if (x == '\n') {
      str << "\\n\"\n\"";
    } else {
      str << x;
    }
  }
  str << '"';
  return str.str();
}

static std::string occur_type_to_formal_parameter_option(int type) {
  switch (type) {
  case xir::OpArgDef::REQUIRED:
    return "OpSchema::Single";
    /// Never or once
  case xir::OpArgDef::OPTIONAL:
    return "OpSchema::Optional";
    /// No limitation
  case xir::OpArgDef::REPEATED:
    return "OpSchema::Variadic";
    /// At least once
  case xir::OpArgDef::REQUIRED_AND_REPEATED:
    return "OpSchema::Variadic";
  default:
    break;
  }
  return "unkknow";
}

static int occur_type_to_minimal_arity(int type) {
  switch (type) {
  case xir::OpArgDef::REQUIRED:
    return 1;
    /// Never or once
  case xir::OpArgDef::OPTIONAL:
    return 0;
  case xir::OpArgDef::REPEATED:
    return 0;
    /// At least once
  case xir::OpArgDef::REQUIRED_AND_REPEATED:
    return 1;
  default:
    break;
  }
  return -1;
}

static std::string
type_index_to_attribute_proto_data_type(const std::type_index& type_index) {
  if (type_index == std::type_index(typeid(int32_t))) {
    return "onnx::AttributeProto::INT";
  } else if (type_index == std::type_index(typeid(float))) {
    return "onnx::AttributeProto::FLOAT";
  } else if (type_index == std::type_index(typeid(double))) {
    return "onnx::AttributeProto::FLOAT";
  } else if (type_index == std::type_index(typeid(std::string))) {
    return "onnx::AttributeProto::STRING";
  } else if (type_index == std::type_index(typeid(std::vector<int32_t>))) {
    return "onnx::AttributeProto::INTS";
  } else if (type_index == std::type_index(typeid(std::vector<float>))) {
    return "onnx::AttributeProto::FLOATS";
  } else if (type_index == std::type_index(typeid(std::vector<std::string>))) {
    return "onnx::AttributeProto::STRINGS";
  } else if (type_index == std::type_index(typeid(std::vector<char>))) {
    return "onnx::AttributeProto::TENSOR";
  } else if (type_index == std::type_index(typeid(bool))) {
    return "onnx::AttributeProto::INT";
  } else {
    LOG_IF(FATAL, true) << "unsupported type" << type_index.name();
  }
  return "undefined";
}

static void write_stream_op_def(std::ostream& str, const xir::OpDef& def) {
  str << "ONNX_VAIP_OPERATOR_SET_SCHEMA(\n";
  str << "    " << convert_to_onnx_name_convetion(def.name()) << ", 1,\n";
  str << "    OpSchema()\n";
  str << "        .SetDoc(" << stringify(def.annotation()) << ")\n";
  auto arg_index = 0;
  for (auto& arg : def.input_args()) {
    str << ".Input(" << arg_index << " , "
        << stringify(convert_to_onnx_name_convetion(arg.name)) << ","
        << stringify(arg.annotation) << ", \"T" << arg_index << "\","
        << occur_type_to_formal_parameter_option(arg.occur_type) << ","
        << occur_type_to_minimal_arity(arg.occur_type) << ","
        << "true, " // is_homogeneous
        << "OpSchema::Differentiable)\n"
        << "        .TypeConstraint(\"T" << arg_index
        << "\", "
           "OpSchema::all_tensor_types_with_bfloat(),\n"
           "                        \"Constrain input and output types to all"
           "tensor\"\n"
           "                        \"types.\")\n";
    arg_index = arg_index + 1;
  }
  auto attr_index = 0;
  for (auto& attr : def.attrs()) {
    auto is_required = (attr.occur_type == xir::AttrDef::REQUIRED);
    if (attr.name == "data_type") {
      continue;
    }
    if (attr.name == "shape") {
      continue;
    }
    str << ".Attr(" << stringify(attr.name)             //
        << "     ," << stringify(attr.annotation)       //
        << "     ," << type_index_to_attribute_proto_data_type(attr.data_type)
        << "     ," << (is_required ? "true" : "false") //
        << "     )"                                     //
        ;
    attr_index = attr_index + 1;
  }
  str << "        .Attr(\"data_type\",\n"
         "              \"The data type of the data of output feature maps, "
         "we "
         "use \"\n"
         "              \"FLOAT32 as the default.\",\n"
         "              AttributeProto::STRING, true)\n"
         "        .Attr(\"shape\", \"The shape of the output tensor.\", "
         "AttributeProto::INTS,\n"
         "              true)\n"
         "\n";

  str << "      .Attr(\"origin_nodes\", \"Origin onnxruntime model's node "
         "indexs.\",\n"
         "                AttributeProto::STRINGS, OPTIONAL_VALUE)\n";
  str << "  .Output(0, \"output\",\n"
         "  \"Output tensor containing the same value of the provided\"\n"
         "                \"tensor.\",\n"
         "                \"T\")\n"
         "        .TypeConstraint(\"T\", "
         "OpSchema::all_tensor_types_with_bfloat(),\n"
         "                        \"Constrain input and output types to all "
         "tensor\"\n"
         "                        \"types.\")\n"
         "        .TypeAndShapeInferenceFunction(xir_shape_infer));\n";
}

void show_all_op_defs() {
  std::ofstream str("opdef.doc");
  auto f = xir::op_def_factory();
  for (auto& it : f->get_registered_ops()) {
    auto& opdef = f->get_const_op_def(it);
    write_stream(str, opdef);
    // break;
  }
  CHECK(str.good());
  str.close();
}

void generate_all_op_defs() {
  std::ofstream str("./src/xir_ops/xir_ops_genereted_defs.inc");
  auto f = xir::op_def_factory();
  for (auto& it : f->get_registered_ops()) {
    if (it == "matmul") {
      continue;
    }
    write_stream_op_def(str, f->get_const_op_def(it));
  }
  CHECK(str.good());
  str.close();
  str.open("./src/xir_ops/xir_ops_genereted_defs_register.inc");
  CHECK(str.good());
  for (auto& it : f->get_registered_ops()) {
    if (it == "matmul") {
      continue;
    }
    str << "fn(GetOpSchema<ONNX_VAIP_OPERATOR_SET_SCHEMA_CLASS_NAME(1, ";
    str << convert_to_onnx_name_convetion(it) << ")>());\n";
  }
  CHECK(str.good());
  str.close();
}
} // namespace vaip
// Local Variables:
// mode:c++
// coding: utf-8-unix
// End:
