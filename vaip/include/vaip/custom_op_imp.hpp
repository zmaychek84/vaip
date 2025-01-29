/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

#pragma once
#include "./pass_context.hpp"
#include "./vaip_plugin.hpp"
#include "onnxruntime_api.hpp"
#include "vaip/capability.pb.h"
#include <filesystem>
#include <vaip/custom_op.h>
struct OrtSession;
typedef struct OrtSession OrtSession;

namespace vaip_core {

class ExecutionProviderConcrete
    : public ExecutionProvider,
      public WithPlugin<ExecutionProviderConcrete,
                        std::shared_ptr<const PassContext>,
                        const MetaDefProto&> {
public:
  static constexpr char entry_point[] = "create_execution_provider";

public:
  VAIP_DLL_SPEC
  ExecutionProviderConcrete(std::shared_ptr<const PassContext> context,
                            const MetaDefProto& meta_def);
  virtual ~ExecutionProviderConcrete();

  //
  std::shared_ptr<PassContext> get_context() {
    return std::const_pointer_cast<PassContext>(context_);
  }

protected:
  std::shared_ptr<const PassContext> context_;
  std::shared_ptr<MetaDefProto> meta_def_;
};

template <typename T, typename CustomOpImp, typename = void>
struct CustomOp_compile_t {
  static std::unique_ptr<CustomOp>
  CustomOp_compile(const T* self, std::shared_ptr<const PassContext> context,
                   std::shared_ptr<MetaDefProto> meta_def) {
    auto ret = std::make_unique<CustomOpImp>(context, meta_def, nullptr);
    const_cast<PassContext*>(context.get())->on_custom_op_create_end();
    return ret;
  }
};

template <typename T, typename CustomOpImp>
struct CustomOp_compile_t<
    T, CustomOpImp, std::void_t<decltype(std::declval<T&>().get_model())>> {
  static std::unique_ptr<CustomOp>
  CustomOp_compile(const T* self, std::shared_ptr<const PassContext> context,
                   std::shared_ptr<MetaDefProto> meta_def) {
    auto ret =
        std::make_unique<CustomOpImp>(context, meta_def, self->get_model());
    const_cast<T*>(self)->set_model(nullptr);
    const_cast<PassContext*>(context.get())->on_custom_op_create_end();
    return ret;
  }
};

template <class CustomOpImp>
class ExecutionProviderImp : public ExecutionProviderConcrete {
public:
  ExecutionProviderImp(std::shared_ptr<const PassContext> context,
                       const MetaDefProto& meta_def)
      : ExecutionProviderConcrete{context, meta_def} {}
  virtual ~ExecutionProviderImp() {}
  virtual DllSafe<std::vector<std::string>>
  get_meta_def_inputs() const override final {
    return DllSafe<std::vector<std::string>>(std::vector<std::string>{
        meta_def_->inputs().begin(), meta_def_->inputs().end()});
  }
  virtual DllSafe<std::vector<std::string>>
  get_meta_def_outputs() const override final {
    return DllSafe<std::vector<std::string>>(std::vector<std::string>{
        meta_def_->outputs().begin(), meta_def_->outputs().end()});
  }
  virtual DllSafe<std::vector<std::string>>
  get_meta_def_nodes() const override final {
    return DllSafe<std::vector<std::string>>(std::vector<std::string>{
        meta_def_->nodes().begin(), meta_def_->nodes().end()});
  }
  virtual DllSafe<std::vector<std::string>>
  get_meta_def_constant_initializer() const override final {
    return DllSafe<std::vector<std::string>>(
        std::vector<std::string>{meta_def_->constant_initializers().begin(),
                                 meta_def_->constant_initializers().end()});
  }
  // VAIP_ORT_API_MAJOR >= 11
  virtual bool get_meta_def_fallback_CPU() const final {
    return meta_def_->fallback_cpu();
  }

  virtual std::unique_ptr<CustomOp> compile() const override final {
    return CustomOp_compile_t<ExecutionProviderImp,
                              CustomOpImp>::CustomOp_compile(this, context_,
                                                             meta_def_);
  };
};

class CustomOpImp : public CustomOp {
public:
  VAIP_DLL_SPEC CustomOpImp(std::shared_ptr<const PassContext> context,
                            const std::shared_ptr<MetaDefProto>& meta_def,
                            onnxruntime::Model* model);
  VAIP_DLL_SPEC virtual ~CustomOpImp();

public:
  virtual void Compute(const OrtApi* api, OrtKernelContext* context) const = 0;
  /*
  How to use?
  1. After try_fuse, in meta_def, set_fallback_cpu(true)
  auto [meta_def, fuse_error] = self_.try_fuse(onnx_graph_,
  subgraph->get_name(), inputs, outputs, {}, "DPU");
  meta_def->set_fallback_cpu(true);
  2. In custom_op.cpp, if you need to fall back to CPU, call ComputeCpu(api,
  context);
  */
  VAIP_DLL_SPEC void ComputeCpu(const OrtApi* api,
                                OrtKernelContext* context) const;
  std::shared_ptr<PassContext> get_context() const {
    return std::const_pointer_cast<PassContext>(context_);
  }

protected:
  std::shared_ptr<const PassContext> context_;
  std::shared_ptr<MetaDefProto> meta_def_;
  Ort::Session session_;
};

} // namespace vaip_core
