/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

#pragma once
#include <iostream>
#include <memory>
#include <string>

#include "vart/runner_ext.hpp"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"
#include <glog/logging.h>
#include <xir/attrs/attrs.hpp>
#include <xrt/xrt_bo.h>

namespace vaip_core {
// XRTUpdateQosImpl class
// update efficient mode directly through xrt::hw_context
class XRTUpdateQosImpl : public QoSUpdateInterface {
public:
  explicit XRTUpdateQosImpl(xrt::hw_context* context)
      : hw_context_(context), support_eff_mode_(true) {}

  void update_qos(const std::string& perf_pref_value) override {
    if (hw_context_) {
      xrt::hw_context::qos_type qos_map_perf;
      qos_map_perf["perf_pref"] = (perf_pref_value == "Efficient") ? 1 : 0;
      std::lock_guard<std::mutex> lock(support_eff_mode_lock);
      try {

        if (support_eff_mode_) {
          hw_context_->update_qos(qos_map_perf);
        }
      } catch (std::exception& e) {
        if (std::string(e.what()).find("perf_pref") != std::string::npos) {
          LOG(WARNING) << "XRT device doesn't support efficient mode, will "
                          "ignore the QoS request.";
          support_eff_mode_ = false;
        } else {
          throw;
        }
      }
    } else {
      LOG(WARNING)
          << "Error: hw_context_ is null in XRTUpdateQosImpl::update_qos";
    }
  }

private:
  xrt::hw_context* hw_context_;
  bool support_eff_mode_;
  mutable std::mutex support_eff_mode_lock;
};

// GEUpdateQosImpl class
// update efficient mode through set_run_attrs api of vart runner
class GEUpdateQosImpl : public QoSUpdateInterface {
public:
  explicit GEUpdateQosImpl(vart::RunnerExt* runner)
      : runner_(runner), support_eff_mode_(true) {}

  void update_qos(const std::string& perf_pref_value) override {
    if (runner_) {
      std::lock_guard<std::mutex> lock(support_eff_mode_lock);
      try {
        if (support_eff_mode_) {
          std::shared_ptr<xir::Attrs> attrs = xir::Attrs::create();
          if (perf_pref_value == "Default") {
            attrs->set_attr<std::string>("performance_preference", "Default");
          } else {
            attrs->set_attr<std::string>("performance_preference",
                                         "HighEfficiencyMode");
          }
          auto unique_attrs = xir::Attrs::clone(attrs.get());
          runner_->set_run_attrs(unique_attrs);
        }
      } catch (std::exception& e) {
        if (std::string(e.what()).find("perf_pref") != std::string::npos) {
          LOG(WARNING) << "XRT device doesn't support efficient mode, will "
                          "ignore the update efficient mode through "
                          "set_run_attrs method in dpu custom op.";
          support_eff_mode_ = false;
        } else {
          LOG(FATAL) << "-- Error: Failed to update efficient mode through "
                        "set_run_attrs method in dpu custom op: "
                     << e.what();
        }
      }
    } else {
      LOG(WARNING) << "Error: runner_ is null in GEUpdateQosImpl::update_qos";
    }
  }

private:
  vart::RunnerExt* runner_;
  bool support_eff_mode_;
  mutable std::mutex support_eff_mode_lock;
};
} // namespace vaip_core
