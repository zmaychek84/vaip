/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
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
