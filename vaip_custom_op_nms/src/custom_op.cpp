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

#include "onnxruntime_api.hpp"

#include <glog/logging.h>
#include <sstream>
//
#include "./custom_op.hpp"

namespace vaip_nms_custom_op {

MyCustomOp::MyCustomOp(std::shared_ptr<const PassContext> context,
                       const std::shared_ptr<MetaDefProto>& meta_def,
                       onnxruntime::Model* model)
    : CustomOpImp(context, meta_def, model) {}

MyCustomOp::~MyCustomOp() {}

static std::string shape_to_string(const std::vector<int64_t>& shape) {
  std::ostringstream str;
  str << "[";
  int c = 0;
  for (auto s : shape) {
    if (c != 0) {
      str << ",";
    }
    str << s;
    c = c + 1;
  }
  str << "]";
  return str.str();
}

struct Box {
  Box(float x1, float y1, float x2, float y2) {
    x1_ = std::min(x1, x2);
    y1_ = std::min(y1, y2);
    x2_ = std::max(x1, x2);
    y2_ = std::max(y1, y2);
  }
  Box(const float* box) {
    auto x1 = box[0];
    auto y1 = box[1];
    auto x2 = box[2];
    auto y2 = box[3];
    x1_ = std::min(x1, x2);
    y1_ = std::min(y1, y2);
    x2_ = std::max(x1, x2);
    y2_ = std::max(y1, y2);
  }
  float x1_;
  float y1_;
  float x2_;
  float y2_;

  float area() {
    auto ret = (x2_ - x1_) * (y2_ - y1_);
    CHECK_GE(ret, 0.0f);
    return ret;
  }

  float intersect(const Box& other) {
    auto x1 = std::max(other.x1_, x1_);
    auto y1 = std::max(other.y1_, y1_);
    auto x2 = std::min(other.x2_, x2_);
    auto y2 = std::min(other.y2_, y2_);
    auto intersect = std::max(0.f, x2 - x1) * std::max(0.f, y2 - y1);
    CHECK_GE(intersect, 0.0f);
    return intersect;
  }
};

std::ostream& operator<<(std::ostream& str, const Box& box) {
  str << "[(" << box.x1_ << "," << box.y1_ << "),(" << box.x2_ << "," << box.y2_
      << ")]";
  return str;
}

static void NMS(const float nms_threshold, //
                size_t batch_idx,          //
                size_t cls_idx,            //
                size_t spatial_dimension,  //
                const float* boxes,        //
                const float* scores,
                std::vector<std::tuple<int64_t, int64_t, int64_t>>& ret) {
  std::vector<std::pair<size_t, float>> order(spatial_dimension);
  for (auto i = 0u; i < spatial_dimension; ++i) {
    order[i].first = i;
    order[i].second = scores[i];
  }
  std::sort(
      order.begin(), order.end(),
      [](const std::pair<size_t, float>& ls,
         const std::pair<size_t, float>& rs) { return ls.second > rs.second; });
  std::vector<size_t> keep;
  std::vector<bool> exist_box(spatial_dimension, true);
  for (auto i = 0u; i < spatial_dimension; ++i) {
    auto idx = order[i].first;
    if (!exist_box[idx]) {
      continue;
    }
    keep.emplace_back(idx);
    for (auto j = i + 1; j < spatial_dimension; ++j) {
      auto kept_idx = order[j].first;
      if (!exist_box[kept_idx]) {
        continue;
      }
      auto b1_ptr = boxes + idx * 4;
      auto b2_ptr = boxes + kept_idx * 4;
      auto b1 = Box(b1_ptr);
      auto b2 = Box(b2_ptr);
      auto intersect = b1.intersect(b2);
      auto sum_area = b1.area() + b2.area();
      auto overlap = intersect / (sum_area - intersect);
      if (overlap >= nms_threshold) {
        exist_box[kept_idx] = false;
      }
    }
  }
  for (auto i = 0u; i < keep.size(); ++i) {
    ret.emplace_back((int64_t)batch_idx, (int64_t)cls_idx, (int64_t)keep[i]);
  }
}

static std::vector<std::tuple<int64_t, int64_t, int64_t>>
NMS(const float nms_threshold, //
    size_t num_batches,        //
    size_t spatial_dimension,  //
    size_t num_classes,
    const float* boxes,        //
    const float* scores) {
  auto ret = std::vector<std::tuple<int64_t, int64_t, int64_t>>();
  ret.reserve(num_batches * spatial_dimension * num_classes);
  for (auto batch_idx = 0u; batch_idx < num_batches; ++batch_idx) {
    auto new_boxes = boxes + batch_idx * spatial_dimension * 4;
    for (auto cls_idx = 0u; cls_idx < num_classes; ++cls_idx) {
      auto new_scores = scores + batch_idx * num_classes * spatial_dimension +
                        cls_idx * spatial_dimension;
      NMS(nms_threshold, batch_idx, cls_idx, spatial_dimension, new_boxes,
          new_scores, ret);
    }
  }
  return ret;
}

void MyCustomOp::Compute(const OrtApi* api, OrtKernelContext* context) const {
  if (Ort::Global<void>::api_ == nullptr) {
    Ort::Global<void>::api_ = api;
  }

  Ort::KernelContext ctx(context);
  auto num_inputs = ctx.GetInputCount();
  auto num_outputs = ctx.GetOutputCount();
  LOG(INFO) << "num_inputs " << num_inputs << " "   //
            << "num_outputs " << num_outputs << " " //
      ;
  CHECK_GE(num_inputs, 2u);
  for (auto idx = 0u; idx < num_inputs; ++idx) {
    auto input_tensor = ctx.GetInput(idx);
    auto tensor_info = input_tensor.GetTensorTypeAndShapeInfo();
    auto tensor_type = tensor_info.GetElementType();
    auto tensor_shape = tensor_info.GetShape();
    auto element_num = tensor_info.GetElementCount();
    LOG(INFO) << "element_num " << element_num << " " //
              << "tensor_type " << tensor_type << " " //
              << "shape: " << shape_to_string(tensor_shape);
  }
  auto boxes_tensor = ctx.GetInput(0);
  auto boxes_shape = boxes_tensor.GetTensorTypeAndShapeInfo().GetShape();
  auto scores_tensor = ctx.GetInput(1);
  auto scores_shape = scores_tensor.GetTensorTypeAndShapeInfo().GetShape();
  CHECK_EQ(boxes_shape.size(), 3u);
  auto num_batches = boxes_shape[0];
  auto spatial_dimension = boxes_shape[1];
  CHECK_EQ(boxes_shape[2], 4);
  CHECK_EQ(scores_shape.size(), 3u);
  CHECK_EQ(num_batches, scores_shape[0]);
  auto num_classes = scores_shape[1];
  CHECK_EQ(spatial_dimension, scores_shape[2]);
  auto boxes = boxes_tensor.GetTensorData<float>();
  auto scores = scores_tensor.GetTensorData<float>();
  auto iou_threshold = 0.6499999761581421f;
  auto selected_indices = NMS(iou_threshold, num_batches, spatial_dimension,
                              num_classes, boxes, scores);
  // process
  for (auto idx = 0u; idx < num_outputs; ++idx) {
    auto output_tensor =
        ctx.GetOutput(idx, {(int64_t)selected_indices.size(), 3});
    auto tensor_info = output_tensor.GetTensorTypeAndShapeInfo();
    auto tensor_type = tensor_info.GetElementType();
    auto tensor_shape = tensor_info.GetShape();
    auto element_num = tensor_info.GetElementCount();
    LOG(INFO) << "element_num " << element_num << " " //
              << "tensor_type " << tensor_type << " " //
              << "shape: " << shape_to_string(tensor_shape);
  }
  auto output_tensor = ctx.GetOutput(0, {(int64_t)selected_indices.size(), 3});
  LOG(INFO) << "void* " << (void*)selected_indices.data();
  auto out_base = output_tensor.GetTensorMutableData<int64_t>();
  auto out = out_base;
  for (auto& x : selected_indices) {
    out[0] = std::get<0>(x);
    out[1] = std::get<1>(x);
    out[2] = std::get<2>(x);
    out = out + 3;
  }
  LOG(INFO) << "HELLO  HERE";
}
} // namespace vaip_nms_custom_op
