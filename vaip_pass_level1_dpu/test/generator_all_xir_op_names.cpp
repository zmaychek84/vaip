/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

#include "../../vaip/include/vaip/xir_headers.hpp" // include using relative path, otherwise it use the file from install path that does not exist on the first build
#include <fstream>
#include <iostream>
#include <stdexcept>
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

static std::string convert_to_onnx_name_convetion(const std::string& name) {
  std::string ret = name;
  std::replace(ret.begin(), ret.end(), '-', '_');
  return ret;
}

static void generate_all_xir_op_names() {
  std::ofstream str("./vaip/src/xir_ops/xir_ops_genereted_names.inc");
  auto f = xir::op_def_factory();
  str << "namespace vaip_core{\n";
  str << "static  std::vector<std::string> XIR_OP_NAMES = "
         "std::vector<std::string>{";
  auto ops = f->get_registered_ops();
  for (auto i = 0u; i < ops.size() - 1; ++i) {
    str << "\"" << convert_to_onnx_name_convetion(ops[i]) << "\",";
  }
  str << "\"" << convert_to_onnx_name_convetion(ops[ops.size() - 1])
      << "\"};\n";
  str << "}";
  CHECK(str.good());
  str.close();
}

int main(int argc, char* argv[]) {
  try {
    generate_all_xir_op_names();
  } catch (const std::exception& e) {
    std::cerr << "exception occurs : " << e.what() << "\n";
  } catch (const ErrorCode& e) {
    std::cerr << "exception caught " << e.getErrMsg() << "\n";
  }

  return 0;
}
