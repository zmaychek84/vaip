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

#include <glog/logging.h>
//
#include "vaip/vaip.hpp"
#include "vitis/ai/env_config.hpp"
#include <filesystem>
#include <mutex>
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include <unordered_map>
DEF_ENV_PARAM(DEBUG_VAIP_PY_EXT, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_VAIP_PY_EXT) >= n)

namespace {
using namespace vaip_core;
namespace py = pybind11;

struct PyExt {
  PyExt(IPass& self) : self_{self} {
    self.add_context_resource("py_ext.interpreter",
                              vaip_core::init_interpreter());
  }
  void process(IPass& self, Graph& graph) {
    py::gil_scoped_acquire acquire;
    try {
      auto module_vaip_pass_py_ext =
          py::module::import("voe.voe_cpp2py_export"); // load it if not
      // initialize the VAIP ORT API, otherwise, on WINDOWS, because of
      // DLL, there are two versions of the global api object, the one
      // in the pyd file is not initialized properly.
      module_vaip_pass_py_ext.attr("_init_vaip_ort_api")(
          py::capsule(vaip_core::api()));
      module_vaip_pass_py_ext.attr("_process_graph")(py::capsule((void*)&graph),
                                                     py::capsule((void*)&self));
    } catch (py::error_already_set& e) {
      LOG(FATAL) << e.what();
    }
  }
  IPass& self_;
};
} // namespace

DEFINE_VAIP_PASS(PyExt, vaip_pass_py_ext)
