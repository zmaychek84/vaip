/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

#pragma once

#include "../../vaip/src/config.hpp"
#include "dlanalyzer.hpp"
#include "vaip/vaip.hpp"
#include "vart/runner_ext.hpp"

#include <algorithm>
#include <future>
#include <mutex>

#include <xir/graph/graph.hpp>

namespace vaip {
class SharedContextRunnerContainer;
class Context;
} // namespace vaip

namespace vaip_dpu_custom_op {
using namespace vaip_core;
class RunnerRequestsQueue;

class MyCustomOp : public CustomOpImp {

public:
  MyCustomOp(std::shared_ptr<const PassContext> context,
             const std::shared_ptr<MetaDefProto>& meta_def,
             onnxruntime::Model* model);

  virtual ~MyCustomOp();

public:
  const google::protobuf::RepeatedPtrField<MetaSchedule>&
  get_input_schdules() const {
    return input_schedules_;
  }
  const google::protobuf::RepeatedPtrField<MetaSchedule>&
  get_output_schdules() const {
    return output_schedules_;
  }

private:
  virtual void Compute(const OrtApi* api,
                       OrtKernelContext* context) const override final;

private:
  std::shared_ptr<GraphHolder> graph_holder_;
  const xir::Subgraph* subgraph_;
  const google::protobuf::RepeatedPtrField<MetaSchedule>& input_schedules_;
  const google::protobuf::RepeatedPtrField<MetaSchedule>& output_schedules_;
  std::unique_ptr<RunnerRequestsQueue> runnerRequestsQueue_;
  bool share_context_;
};

struct RunnerHolder {
  std::unique_ptr<vart::RunnerExt> runner_;
#if ENABLE_XRT_SHARED_CONTEXT
  std::shared_ptr<vaip::Context> context_;
#else
  std::shared_ptr<xir::Attrs> attrs_;
#endif
};

class RunnerRequestsQueue {
public:
  RunnerRequestsQueue(std::vector<std::unique_ptr<RunnerHolder>>& runners) {
    std::shared_ptr<RunnerHolder> runner_request;
    for (auto& runner : runners) {
      runner_request = std::move(runner);
      runner_requests_.push_back(runner_request);
    }
  }

  ~RunnerRequestsQueue() {
    // clearing out the runner_requests_ vector pool in the class's
    // destructor
    for (auto& pointer : runner_requests_) {
      pointer = nullptr;
    }
    runner_requests_.erase(
        std::remove(runner_requests_.begin(), runner_requests_.end(), nullptr),
        runner_requests_.end());
  }

  void putIdleRequest(std::shared_ptr<RunnerHolder> runner_request) {
    std::unique_lock<std::mutex> lock(_mutex);
    runner_requests_.push_back(runner_request);
    _cv.notify_one();
  }

  std::shared_ptr<RunnerHolder> getIdleRequest() {
    std::unique_lock<std::mutex> lock(_mutex);
    _cv.wait(lock, [this] { return runner_requests_.size() > 0; });
    auto request = runner_requests_.at(0);
    runner_requests_.erase(runner_requests_.begin());
    return request;
  }

private:
  std::mutex _mutex;
  std::condition_variable _cv;
  std::vector<std::shared_ptr<RunnerHolder>> runner_requests_;
};

} // namespace vaip_dpu_custom_op