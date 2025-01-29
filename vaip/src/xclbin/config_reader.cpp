/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */
#include "config_reader.hpp"
#include "nlohmann/json.hpp"
#include "onnxruntime_api.hpp"
#include "vitis/ai/env_config.hpp"
#include "xclbin_file.hpp"
#include <filesystem>
#include <fstream>
#include <glog/logging.h>
#include <stdlib.h>
#include <string>
#include <unordered_map>

DEF_ENV_PARAM_2(XLNX_VART_FIRMWARE, "", std::string)
DEF_ENV_PARAM_2(DEBUG_LOG_LEVEL, "error", std::string)
DEF_ENV_PARAM(XLNX_ENABLE_BATCH, "0")
DEF_ENV_PARAM(NUM_OF_DPU_RUNNERS, "1")

namespace vaip_core {

// FIXME: The name of this function is misleading.
static nlohmann::json
get_config_json_str_from_config_file(const std::string& filename) {
  std::ifstream f(filename);
  if (!f.is_open()) {
    std::string err_msg =
        std::string{"failed to open config file: "} + filename;
    throw std::runtime_error(err_msg);
  }
  return nlohmann::json::parse(f);
}

static void
update_enable_batch(const onnxruntime::ProviderOptions& session_option) {
  if (session_option.find("xlnx_enable_batch") != session_option.end()) {
    std::string enable_batch = session_option.at("xlnx_enable_batch");
    if (enable_batch == "1") {
      ENV_PARAM(XLNX_ENABLE_BATCH) = 1;
    } else if (enable_batch == "0") {
      ENV_PARAM(XLNX_ENABLE_BATCH) = 0;
    } else {
      ENV_PARAM(XLNX_ENABLE_BATCH) = 1;
    }
  }
}

static void
update_num_dpu_runners(const onnxruntime::ProviderOptions& session_option) {
  int num_of_dpu_runners = 1;
  if (session_option.find("num_of_dpu_runners") != session_option.end()) {
    std::string str_of_dpu_runners = session_option.at("num_of_dpu_runners");
    num_of_dpu_runners = atoi(str_of_dpu_runners.c_str());
    if (num_of_dpu_runners <= 0 || num_of_dpu_runners > 8) {
      return;
    }
    ENV_PARAM(NUM_OF_DPU_RUNNERS) = num_of_dpu_runners;
#ifdef _WIN32
    _putenv_s("NUM_OF_DPU_RUNNERS", std::to_string(num_of_dpu_runners).c_str());
#else
    setenv("NUM_OF_DPU_RUNNERS", std::to_string(num_of_dpu_runners).c_str(), 1);
#endif
  }
}

static void
update_log_level(const onnxruntime::ProviderOptions& session_option) {
  std::string log_level = ENV_PARAM(DEBUG_LOG_LEVEL);
  if (session_option.find("log_level") != session_option.end()) {
    log_level = session_option.at("log_level");
  }

  if (log_level == "info") {
    FLAGS_minloglevel = google::GLOG_INFO;
  } else if (log_level == "warning") {
    FLAGS_minloglevel = google::GLOG_WARNING;
  } else if (log_level == "error") {
    FLAGS_minloglevel = google::GLOG_ERROR;
  } else if (log_level == "fatal") {
    FLAGS_minloglevel = google::GLOG_FATAL;
  } else {
    FLAGS_minloglevel = google::GLOG_ERROR;
  }
}
static void restore_session_options(nlohmann::json& ret,
                                    std::string entry_second) {
  std::map<std::string, std::string> session_config_options_entry_list = {};
#if VAIP_ORT_API_MAJOR >= 10
  auto options = reinterpret_cast<Ort::SessionOptions*>(
      (uintptr_t)std::stoull(entry_second));
  VAIP_ORT_API(session_option_configuration)
  (&session_config_options_entry_list, options,
   [](void* mmap, const char* name, const char* value) {
     auto* map_ptr =
         reinterpret_cast<std::map<std::string, std::string>*>(mmap);
     map_ptr->insert({name, value});
   });
#else
  LOG(WARNING) << "ORT API version is less than 10, now used is "
               << VAIP_ORT_API_MAJOR
               << ", session options will not be restored.";
#endif
  for (const auto& option : session_config_options_entry_list) {
    std::string key = option.first;
    // Here are the actual session_configs.
    ret["ort_session_config"][key] = option.second;
  }
}

static nlohmann::json
get_config_json(const onnxruntime::ProviderOptions& options) {
  nlohmann::json ret;
  update_log_level(options);
  update_enable_batch(options);
  update_num_dpu_runners(options);
  bool config_set = options.find("config_file") != options.end();
  if (config_set) {
    ret = get_config_json_str_from_config_file(options.at("config_file"));
  } else if (VAIP_ORT_API(vaip_get_default_config)) {
    ret = nlohmann::json::parse(VAIP_ORT_API(vaip_get_default_config)());
  } else {
    auto default_config =
        "{\"passes\":[{\"name\":\"init\",\"plugin\":\"vaip_pass_init\"}]}";
    LOG(INFO) << "can not find onnxruntime_vitisai_ep.dll , get_default_config "
                 "return only init pass.";
    ret = nlohmann::json::parse(default_config);
  }
  bool xclbin_in_config_file = ret.count("xclbin") > 0;
  const std::string session_prefix = "ort_session_config.";
  for (const auto& entry : options) {
    // In the case where xclbin are (mistakenly) specified both in
    // `ProviderOptions["config_file"]` and in `ProvidersOptions["xclbin"]`.
    if (entry.first == "xclbin" && xclbin_in_config_file) {
      continue;
    }
    if (entry.first == "session_options") {
      // The key here is "session_options," and the value is a string that
      // points to the session_options object.
      restore_session_options(ret, entry.second);
      continue;
    }
    // this is actually provider options, for backward compatibility, we keep
    // this key in the root of the json.
    ret["sessionOptions"][entry.first] = entry.second;
  }
  return ret;
}

std::string get_config_json_str(const onnxruntime::ProviderOptions& options) {
  try {
    auto data = vaip_core::get_config_json(options);
    return data.dump();
  } catch (const std::exception& e) {
    LOG(FATAL) << "Error: " << e.what() << std::endl;
    return "";
  }
}
} // namespace vaip_core
