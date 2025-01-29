/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */
#include <exception>
#include <vaip/vaip.hpp>
using namespace vaip_core;
using namespace std;
int main(int argc, char* argv[]) {
  try {
    {
      auto x = init_interpreter();
      {
        auto y = init_interpreter();
        cerr << "y = " << y.get() << endl;
        cerr << "x = " << x.get() << endl;
        eval_python_code("print('hello python')");
        eval_python_code("print('hello python')");
      }
    }

    auto b = PatternBuilder();
    std::string py_pattern = slurp("/tmp/test_conv_pattern.py");

    auto p = b.create_by_py(py_pattern);
    cerr << (void*)p.get() << endl;
    cerr << p->debug_string() << endl;
    cerr << "BYE" << endl;
  } catch (const std::exception& e) {
    std::cerr << "exception occurs : " << e.what() << "\n";
  }
  return 0;
}
