/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

#include <glog/logging.h>
#include <iostream>
#include <mutex>
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;
using namespace std;

std::shared_ptr<void> init_interpreter() {
  static std::mutex mtx;
  static std::weak_ptr<void> py_interpreter_holder;
  std::shared_ptr<void> ret;
  std::lock_guard<std::mutex> lock(mtx);
  if (!Py_IsInitialized()) {
    static py::scoped_interpreter inter{};
    auto p = static_cast<void*>(&inter);
    ret = std::shared_ptr<void>(
        p, [](void* p) { cout << "delete init_interpreter" << endl; });
    py_interpreter_holder = ret;
  }
  if (!ret) {
    ret = py_interpreter_holder.lock();
  }
  return ret;
}

void run(string in, string out, vector<int64_t> input_shape, string mode) {
  auto inter = init_interpreter();
  py::list shape;
  for (const auto& elem : input_shape) {
    shape.append(py::cast(elem));
  }
  auto m = py::module::import("voe.tools.quantize_pass");
  m.attr("quantize_static")(in, out, shape, mode);
}

void usage() {
  std::cout << "usage: test_python_quantize_model <input onnx model> <output "
               "onnx model> <input shape eg. 1,3,224,224 > <mode>"
            << std::endl;
}

std::vector<int64_t> splitAndConvertToInt(const std::string& str,
                                          char delimiter) {
  std::vector<int64_t> result;
  std::stringstream ss(str);
  std::string token;

  while (std::getline(ss, token, delimiter))
    result.push_back(std::stoi(token));

  return result;
}

int main(int argc, char* argv[]) {
  try {
    {
      py::scoped_interpreter inter{};
      py::module::import("vai_q_onnx.quantize");
      cout << "import done" << endl;
    }
    if (0) {
      if (argc < 4)
        usage();
      auto shape = splitAndConvertToInt(argv[3], ',');
      run(argv[1], argv[2], shape, argv[4]);
      cout << "run done" << endl;
    }
  } catch (const std::exception& e) {
    std::cerr << "exception occurs : " << e.what() << "\n";
  }

  return 0;
}
