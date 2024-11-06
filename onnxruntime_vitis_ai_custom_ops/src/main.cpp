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

/*

**/
#include <glog/logging.h>

#include "../include/onnxruntime_vitis_ai_custom_ops.hpp"
#include "custom_op_gqa.hpp"
#include "custom_op_gqo.hpp"
#include "custom_op_mha.hpp"
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
