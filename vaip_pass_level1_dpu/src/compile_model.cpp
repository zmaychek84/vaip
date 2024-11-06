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

#include "compile_model.hpp"
#include "xclbin_file.hpp"

#include "vitis/ai/env_config.hpp"
#if FIND_FINGERPRINT
#  include "xir/dpu_controller.hpp"
#endif

#include "../../vaip/src/config.hpp"

DEF_ENV_PARAM_2(XLNX_TARGET_NAME, "AMD_AIE2_Nx4_Overlay", std::string)
DEF_ENV_PARAM(XLNX_READ_TARGET_NAME_FROM_DPU, "1")
DEF_ENV_PARAM(VAIP_COMPILE_RESERVE_CONST_DATA, "-1")
DEF_ENV_PARAM(OPT_LEVEL, "0")
DEF_ENV_PARAM(DPU_SUBGRAPH_NUM, "-1")
DEF_ENV_PARAM(DEBUG_COMPILE_MODEL, "0")
DEF_ENV_PARAM(USE_CPU_RUNNER, "0")
DEF_ENV_PARAM_2(XLNX_VART_FIRMWARE, "", std::string)
DEF_ENV_PARAM(XLNX_ONNX_EP_VERBOSE, "0")
DEF_ENV_PARAM(XLNX_ENABLE_OP_NAME_PROTECTION, "1")
#define LOG_VERBOSE(n)                                                         \
  LOG_IF(INFO, ENV_PARAM(XLNX_ONNX_EP_VERBOSE) >= n)                           \
      << "[XLNX_ONNX_EP_VERBOSE] "

namespace vaip_core {
static std::string finger_print_to_hex_str(uint64_t fingerprint) {
  std::ostringstream ss;
  ss << "0x" << std::hex << fingerprint;
  return ss.str();
}
#if FIND_FINGERPRINT
static std::string get_finger_print() {
  auto dpu_controller = xir::DpuController::get_instance();
  CHECK_GT(dpu_controller->get_num_of_dpus(), 0);
  return finger_print_to_hex_str(dpu_controller->get_fingerprint(0));
}
#endif

template <typename T> std::string vec_to_string(const std::vector<T>& v) {
  std::ostringstream oss;
  oss << '[';
  if (!v.empty()) {
    oss << v.front();
    for (auto it = std::next(v.begin()); it != v.end(); ++it) {
      oss << ',' << *it;
    }
  }
  oss << ']';
  return oss.str();
}
void set_fingerprint_with_xcompiler_attr(xir::Attrs* attr,
                                         std::string fingerprint) {

  if (fingerprint.rfind("0x", 0) == 0) {
    attr->set_attr<std::vector<std::string>>("fingerprint", {fingerprint});
    LOG_VERBOSE(1) << "fingerprint: " << fingerprint;
  } else {
    attr->set_attr<std::vector<std::string>>("target", {fingerprint});
    LOG_VERBOSE(1) << "XLNX_TARGET_NAME = " << fingerprint;
  }
}
static std::unique_ptr<xir::Attrs>
create_compile_attrs(const PassContext& context,
                     const PassDpuParamProto& dpu_param) {
  auto compile_options = xir::Attrs::create();
  auto dl_analyzer_enabled = ENV_PARAM(XLNX_ONNX_EP_DL_ANALYZER_PROFILING) ||
                             ENV_PARAM(XLNX_ONNX_EP_DL_ANALYZER_VISUALIZATION);

  for (const auto& param : dpu_param.xcompiler_attrs()) {
    if (param.second.has_bool_value()) {
      compile_options->set_attr<bool>(param.first, param.second.bool_value());
    } else if (param.second.has_int_value()) {
      compile_options->set_attr<std::int32_t>(param.first,
                                              param.second.int_value());
    } else if (param.second.has_uint_value()) {
      if (param.first == "profile") {
        compile_options->set_attr<std::uint32_t>(
            param.first, dl_analyzer_enabled ? 3 : param.second.uint_value());
      } else
        compile_options->set_attr<std::uint32_t>(param.first,
                                                 param.second.uint_value());
    } else if (param.second.has_string_value()) {
      compile_options->set_attr<std::string>(param.first,
                                             param.second.string_value());
    } else if (param.second.int_values_size() > 0) {
      auto values = std::vector<std::int32_t>(param.second.int_values().begin(),
                                              param.second.int_values().end());
      compile_options->set_attr<std::vector<std::int32_t>>(param.first, values);
    } else if (param.second.string_values_size() > 0) {
      auto values =
          std::vector<std::string>(param.second.string_values().begin(),
                                   param.second.string_values().end());
      compile_options->set_attr<std::vector<std::string>>(param.first, values);
    } else {
      LOG(FATAL) << "xcompiler attr type not supported , key " << param.first;
    }
  }
  if (context.get_provider_option("opt_level").has_value()) {
    std::string str_opt_level =
        context.get_provider_option("opt_level").value();
    std::uint32_t opt_level = atoi(str_opt_level.c_str());
    compile_options->set_attr<std::uint32_t>("opt_level", opt_level);
  }
  if (ENV_PARAM(XLNX_ENABLE_OP_NAME_PROTECTION)) {
    compile_options->set_attr<bool>("enable_op_tensor_name_protection", true);
  }
  if (ENV_PARAM(DPU_SUBGRAPH_NUM) != -1) {
    compile_options->set_attr<std::uint32_t>("dpu_subgraph_num",
                                             ENV_PARAM(DPU_SUBGRAPH_NUM));
  }

  auto fingerprint = get_xcompiler_fingerprint(context, dpu_param);
  set_fingerprint_with_xcompiler_attr(compile_options.get(), fingerprint);

  if (ENV_PARAM(XLNX_VART_FIRMWARE) != "")
    LOG_VERBOSE(1) << "XLNX_VART_FIRMWARE = " << ENV_PARAM(XLNX_VART_FIRMWARE);

  if (ENV_PARAM(VAIP_COMPILE_RESERVE_CONST_DATA) != -1) {
    compile_options->set_attr<bool>(
        "reserve_const_data", ENV_PARAM(VAIP_COMPILE_RESERVE_CONST_DATA) != 0);
  }
  LOG_VERBOSE(1) << "USE_CPU_RUNNER = " << ENV_PARAM(USE_CPU_RUNNER);

  auto keys = compile_options->get_keys();
  for (auto& key : keys) {
    auto attr = compile_options->get_attr(key);
    if (attr.type() == typeid(bool)) {
      LOG_VERBOSE(1) << key << " = "
                     << (compile_options->get_attr<bool>(key) ? "True"
                                                              : "False");
    } else if (attr.type() == typeid(std::uint32_t)) {
      LOG_VERBOSE(1) << key << " = "
                     << compile_options->get_attr<std::uint32_t>(key);
    } else if (attr.type() == typeid(std::string)) {
      LOG_VERBOSE(1) << key << " = "
                     << compile_options->get_attr<std::string>(key);
    } else if (attr.type() == typeid(std::int32_t)) {
      LOG_VERBOSE(1) << key << " = "
                     << compile_options->get_attr<std::int32_t>(key);
    } else if (attr.type() == typeid(std::vector<std::int32_t>)) {
      LOG_VERBOSE(1)
          << key << " = "
          << vec_to_string<std::int32_t>(
                 compile_options->get_attr<std::vector<std::int32_t>>(key));
    } else if (attr.type() == typeid(std::vector<std::string>)) {
      LOG_VERBOSE(1) << key << " = "
                     << vec_to_string<std::string>(
                            compile_options->get_attr<std::vector<std::string>>(
                                key));
    } else {
      LOG(WARNING) << "not support data type .  key " << key;
    }
  }
  return compile_options;
}

// only Backward Compatibility (mepTable and targetproto)
bool is_absolute_path(const std::string& path) {
#ifdef _WIN32
  if (path.size() > 1 && path[1] == ':') {
    return true;
  }
#else
  if (path.size() > 0 && path[0] == '/') {
    return true;
  }
#endif
  return false;
}

std::string get_xclbin_fullpath(const std::string& xclbin) {
  if (is_absolute_path(xclbin)) {
    return xclbin;
  }
#ifdef _WIN32
  // maybe can add search path
  const std::filesystem::path dir("C:\\Windows\\System32\\AMD");
  auto full_path = dir / xclbin;
  return full_path.string();
#endif
  return xclbin;
}
// return xcompiler fingerprint
std::string get_xcompiler_fingerprint(const PassContext& pass_context,
                                      const PassDpuParamProto& dpu_param) {
  auto ret = ENV_PARAM(XLNX_TARGET_NAME);
  auto xclbin = dpu_param.xclbin();
  if (!xclbin.empty()) {
    auto fullpath_xclbin =
        pass_context.xclbin_path_to_cache_files(get_xclbin_fullpath(xclbin));

    auto fingerprint = get_xclbin_fingerprint(pass_context, fullpath_xclbin);
    if (fingerprint) {
      ret = finger_print_to_hex_str(*fingerprint);
    } else {
      LOG(WARNING)
          << "get fingerprint from xclbin failed, use XLNX_TARGET_NAME";
      ret = ENV_PARAM(XLNX_TARGET_NAME);
    }
  } else {
    LOG(WARNING) << "vaip_config does not have xclbin filed configured, use "
                    "XLNX_TARGET_NAME";
    ret = ENV_PARAM(XLNX_TARGET_NAME);
  }

#if FIND_FINGERPRINT
  if (ENV_PARAM(XLNX_READ_TARGET_NAME_FROM_DPU)) {
    if (xir::DpuController::exist_dpu_devices()) {
      auto finger_print = get_finger_print();
      LOG_IF(INFO, ENV_PARAM(DEBUG_COMPILE_MODEL))
          << "read fingerprint from DPU, fingerprint : " << finger_print;
      return finger_print;
    }
  }
#endif
  return ret;
}
std::unique_ptr<xir::Graph>
compiler_xir_model(std::unique_ptr<xir::Graph> graph,
                   const PassContext& pass_context,
                   const PassDpuParamProto& dpu_param) {
  auto compile_options = create_compile_attrs(pass_context, dpu_param);
  LOG_IF(INFO, ENV_PARAM(DEBUG_COMPILE_MODEL))
      << " compiler xir model."
      << "\ncompile attrs: " << compile_options->debug_info();
  if (graph->get_root_subgraph()->is_leaf()) {
    graph->get_root_subgraph()->create_children(); // xcompiler need this graph
                                                   // must be create_children();
  }

  std::unique_ptr<xir::Graph> ret;
#if WITH_XCOMPILER
  auto model = std::string();
  graph->serialize_to_string(&model);
  auto allocator = [](void* state, size_t size) {
    auto s = (std::string*)state;
    s->resize(size);
    return s->data();
  };
  auto config_model = xir::Graph::create("config_model");
  config_model->set_attrs(std::move(compile_options));
  auto config_string = std::string();
  config_model->serialize_to_string(&config_string);

  VAIP_ORT_API(vaip_xcompiler_compile)
  (model.data(), model.size(), config_string.data(), config_string.size(),
   reinterpret_cast<void*>(&ret), [](void* env, void* data, size_t size) {
     auto state = reinterpret_cast<std::unique_ptr<xir::Graph>*>(env);
     *state = xir::Graph::deserialize_from_string(
         std::string((const char*)data, size));
   });
#else
  LOG(FATAL)
      << "xcompiler is disabled, please build voe with WITH_XCOMPILER=on";
#endif
  return ret;
}

} // namespace vaip_core
