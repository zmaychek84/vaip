/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

/*

**/
#include <glog/logging.h>

#include "../include/onnxruntime_vitis_ai_custom_ops.hpp"
#include "custom_op_gqa.hpp"
#include "custom_op_gqo.hpp"
#include "custom_op_mha.hpp"
#include "custom_op_prefill_gqa.hpp"
#include "custom_op_rope.hpp"
#include "custom_op_slrn.hpp"
#include "custom_op_sslrn.hpp"
#include "custom_op_ssmlp.hpp"

static void AddOrtCustomOpDomainToContainer(Ort::CustomOpDomain&& domain) {
  static std::vector<Ort::CustomOpDomain> ort_custom_op_domain_container;
  static std::mutex ort_custom_op_domain_mutex;
  std::lock_guard<std::mutex> lock(ort_custom_op_domain_mutex);
  ort_custom_op_domain_container.push_back(std::move(domain));
}

OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options,
                                          const OrtApiBase* api) {
  Ort::Global<void>::api_ = api->GetApi(ORT_API_VERSION);
  // placeholder code, to be removed
  static const ort_mha_custom_op::MyCustomOp mycustom_op_mha;
  static const ort_gqa_custom_op::MyCustomOp mycustom_op_gqa;
  static const ort_prefill_gqa_custom_op::PrefillGQACustomOp
      mycustom_op_prefill_gqa;
  static const ort_gqo_custom_op::MyCustomOp mycustom_op_gqo;
  static const ort_ssmlp_custom_op::MyCustomOp mycustom_op_ssmlp;
  static const ort_sslrn_custom_op::MyCustomOp1 mycustom_op_sslrn;
  static const ort_slrn_custom_op::MyCustomOp1 mycustom_op_slrn;

  static const ort_rope_custom_op::MyCustomOp mycustom_op_rope;
  static const char* c_OpDomain = "com.amd";
  OrtStatus* result = nullptr;

  Ort::ConstSessionOptions options_obj{options};
  try {
    Ort::CustomOpDomain domain{c_OpDomain};
    domain.Add(&mycustom_op_mha);
    domain.Add(&mycustom_op_gqa);
    domain.Add(&mycustom_op_prefill_gqa);
    domain.Add(&mycustom_op_gqo);
    domain.Add(&mycustom_op_ssmlp);
    domain.Add(&mycustom_op_sslrn);
    domain.Add(&mycustom_op_slrn);
    domain.Add(&mycustom_op_rope);

    Ort::UnownedSessionOptions session_options(options);
    session_options.Add(domain);
    AddOrtCustomOpDomainToContainer(std::move(domain));
  } catch (const std::exception& e) {
    ([&]() {
      Ort::Status status{e};
      result = status.release();
    }());
  }
  return result;
}

OrtStatus* ORT_API_CALL RegisterCustomOpsAltName(OrtSessionOptions* options,
                                                 const OrtApiBase* api) {
  return RegisterCustomOps(options, api);
}
