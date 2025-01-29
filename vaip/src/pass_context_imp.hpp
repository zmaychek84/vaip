/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */
#pragma once
#include <deque>

#include <vaip/custom_op.h>
#include <vaip/dll_safe.h>

#include "vaip/model.hpp"
#include "vaip/pass.hpp"
#include "vaip/pass_context.hpp"
#include "vaip/vaip_io.hpp"

namespace vaip_core {
class CacheFileReaderImp : public CacheFileReader {
public:
  CacheFileReaderImp(bool in_mem, const std::string& filename, FILE* fp);
  virtual ~CacheFileReaderImp();

private:
  size_t size() const override final;
  void rewind() const override final;
  virtual std::size_t fread(void* buffer,
                            std::size_t size) const override final;

private:
  const bool in_mem_;
  const std::string name_;
  size_t size_;
  FILE* fp_;
};
class CacheFileWriterImp : public CacheFileWriter {
public:
  CacheFileWriterImp(bool in_mem, const std::string& fileanme, FILE* fp);
  virtual ~CacheFileWriterImp();

private:
  virtual std::size_t fwrite(const void* buffer,
                             std::size_t size) const override final;

private:
  const bool in_mem_;
  const std::string name_;
  FILE* fp_;
};

class CacheFileStreamWriter : public IStreamWriter {
public:
  CacheFileStreamWriter(std::unique_ptr<CacheFileWriter>&& writer)
      : writer_(std::move(writer)) {}

private:
  virtual size_t write(const char* data, size_t size) override final;

private:
  std::unique_ptr<CacheFileWriter> writer_;
};
class CacheFileStreamWriterBuilder : public IStreamWriterBuilder {
public:
  CacheFileStreamWriterBuilder(PassContext* ctx) : context(ctx) {}

private:
  virtual std::unique_ptr<IStreamWriter>
  build(const std::string& filename) override final;

private:
  PassContext* context;
};

class CacheFileStreamReader : public IStreamReader {
public:
  CacheFileStreamReader(const std::string& name,
                        std::unique_ptr<CacheFileReader> reader)
      : reader_(std::move(reader)) {}

private:
  virtual size_t read(char* data, size_t size) override final;

private:
  std::unique_ptr<CacheFileReader> reader_;
};

class PassContextImp : public PassContext {
public:
  std::vector<char> const_data_;
  std::map<std::string, std::shared_ptr<std::function<void(gsl::span<char>)>>>
      const_lazy_;
  std::filesystem::path log_dir;
  std::map<std::string, std::vector<AttributeProtoPtr>> node_extra_attrs;
  std::deque<IPass*> current_pass_stack;
  ContextProto context_proto;
  bool is_ep_context_model = false;
  bool cache_dir_set = false;
  std::filesystem::path model_path;
  std::unique_ptr<vaip_cxx::Model> ep_context_model_;
  std::chrono::time_point<std::chrono::steady_clock> start_ =
      std::chrono::steady_clock::now();
  mutable int suffix_counter = 0;
  std::unordered_map<std::string, std::shared_ptr<void>> pass_resources;

public:
  ~PassContextImp();
  int allocate_suffix() const;
  virtual std::filesystem::path get_log_dir() const override final;
  virtual std::optional<std::string>
  get_provider_option(const std::string& option_name) const override final;
  virtual std::optional<std::string>
  get_session_config(const std::string& option_name) const override final;
  virtual std::string
  get_provider_option(const std::string& option_name,
                      const std::string& default_value) const override final;
  virtual int64_t get_provider_option_i64(const std::string& option_name,
                                          int64_t default_value) const;
  virtual bool cache_in_mem() const override final;
  virtual void set_is_ep_context_model(bool is_ep_context_model) override final;
  virtual bool get_is_ep_context_model() override final;
  virtual std::string
  get_session_config(const std::string& option_name,
                     const std::string& default_value) const override final;
  virtual std::string
  get_run_option(const std::string& option_name,
                 const std::string& default_value) const override final;
  virtual std::string
  get_ep_dynamic_option(const std::string& option_name,
                        const std::string& default_value) const override final;
  virtual void add_QosUpdater(
      const std::shared_ptr<QoSUpdateInterface>& updater) const override final;
  virtual void
  update_all_qos(const std::string& workload_type) const override final;

  virtual const ConfigProto& get_config_proto() const override final;

private:
  template <typename T>
  std::optional<std::vector<T>>
  read_file_generic(const std::string& filename) const;

public:
  virtual std::optional<std::vector<char>>
  read_file_c8(const std::string& filename) const override final;
  std::optional<std::vector<uint8_t>>
  read_file_u8(const std::string& filename) const override final;
  virtual std::unique_ptr<CacheFileReader>
  open_file_for_read(const std::string& filename) const override final;
  virtual std::unique_ptr<CacheFileWriter>
  open_file_for_write(const std::string& filename) override final;
  virtual FILE* open_file(const std::string& filename) const override final;
  virtual bool write_file(const std::string& filename,
                          gsl::span<const char> data) override final;
  virtual void restore_cache_files() override final;
  virtual bool has_cache_file(const std::string& filename) const override final;
  virtual std::vector<char> cache_files_to_tar_mem() override final;

  virtual bool cache_files_to_tar_file(
      const std::filesystem::path& tar_file) const override final;
  virtual bool cache_files_to_tar_file(FILE* file) const override final;
  virtual bool
  tar_file_to_cache_files(const std::filesystem::path& tar_file) override final;
  virtual bool tar_mem_to_cache_files(const char* data,
                                      size_t size) override final;
  virtual bool tar_file_to_cache_files(FILE* file) override final;

  virtual std::shared_ptr<void>
  get_context_resource(const std::string& name) const override final;
  virtual std::filesystem::path xclbin_path_to_cache_files(
      const std::filesystem::path& path) const override final;
  virtual std::optional<std::vector<char>>
  read_xclbin(const std::filesystem::path& path) const override final;
  virtual std::unique_ptr<PassContextTimer>
  measure(const std::string& label) override final;
  virtual void on_custom_op_create_end() override final;

  // helper class
  struct WithPass {
    WithPass(PassContextImp& context, IPass& pass);
    WithPass(const WithPass&) = delete;
    ~WithPass();
    PassContextImp* _context;
  };
  WithPass with_current_pass(IPass& pass);
  void add_context_resource(const std::string& name,
                            std::shared_ptr<void> resource);
  virtual void save_context_json() const override final;

private:
  // use std::map to keep filename ordered.
  std::map<std::string, FILE*> cache_files_;
  friend int vitisai_ep_set_ep_dynamic_options(
      const std::vector<std::unique_ptr<vaip_core::ExecutionProvider>>& eps,
      const char* const* keys, const char* const* values, size_t kv_len);
  std::map<std::string, std::string> ep_dynamic_options;
  mutable std::mutex ep_dynamic_options_lock;
  // for share context, many context may be same. may need to change container
  // to set.
  mutable std::vector<std::shared_ptr<QoSUpdateInterface>> qos_updaters_;
  int created_customop_count = 0;
};

struct PassContextTimerImp : public PassContextTimer {
  PassContextTimerImp(const std::string& label, PassContextImp& context);
  virtual ~PassContextTimerImp();
  std::string label_;
  PassContextImp& context_;
  std::chrono::time_point<std::chrono::steady_clock> start_;
  MemUsageProto mem_usage_;
};
} // namespace vaip_core
