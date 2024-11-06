/*
 *     The Xilinx Vitis AI Vaip in this distribution are provided under the
 * following free and permissive binary-only license, but are not provided in
 * source code form.  While the following free and permissive license is similar
 * to the BSD open source license, it is NOT the BSD open source license nor
 * other OSI-approved open source license.
 *
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
