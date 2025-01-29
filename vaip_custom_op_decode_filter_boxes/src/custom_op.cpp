/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

#include "onnxruntime_api.hpp"

#include <glog/logging.h>
#include <sstream>

#include "custom_op.hpp"
#include "xf_aie_host_utils.hpp"

#include "vitis/ai/profiling.hpp"

#include <cmath>
#include <fstream>

DEF_ENV_PARAM(DEBUG_DECODE_FITLER_BOX_CUSTOM_OP, "0")
DEF_ENV_PARAM_2(XLNX_VART_FIRMWARE, "", std::string)
#define LOG_THIS(n)                                                            \
  LOG_IF(INFO, ENV_PARAM(DEBUG_DECODE_FITLER_BOX_CUSTOM_OP) >= n)

namespace vaip_decode_filter_boxes_custom_op {

template <typename DType, typename Type, int Dim>
static void ReadFromFile(std::string& filename,
                         std::vector<std::array<Type, Dim>>& buffer) {
  std::ifstream file(filename);
  CHECK_EQ(file.is_open(), true);
  std::string line;
  while (std::getline(file, line)) {
    std::istringstream items(line);
    std::array<Type, Dim> arr;
    for (int i = 0; i < Dim; ++i) {
      DType item;
      items >> item;
      arr[i] = static_cast<Type>(item);
    }
    buffer.push_back(arr);
  }
}

MyCustomOp::MyCustomOp(std::shared_ptr<const PassContext> context,
                       const std::shared_ptr<MetaDefProto>& meta_def,
                       onnxruntime::Model* model)
    : CustomOpImp(context, meta_def, model)
#ifndef FDPOST_CPU_KERNEL
      ,
      kernel_name_(PPGetPSKernelName(PP_FD_POST))
#endif
{
#ifndef FDPOST_CPU_KERNEL
  // Backward compatibility.
  auto xclbin_file = ENV_PARAM(XLNX_VART_FIRMWARE);
  auto cfg_sess_opts = context->get_config_proto().provider_options();
  auto it = cfg_sess_opts.find("xclbin");
  if (it != cfg_sess_opts.end() && !it->second.empty()) {
    xclbin_file = it->second;
  }

  bool share_context = false;
  if (cfg_sess_opts.contains(vaip::Context::CTX_SHARE_OPTION_KEY)) {
    try {
      share_context =
          std::stoi(cfg_sess_opts.at(vaip::Context::CTX_SHARE_OPTION_KEY));
    } catch (...) {
      LOG_THIS(1) << "failed to convert provider option \""
                  << vaip::Context::CTX_SHARE_OPTION_KEY << "\" value \""
                  << cfg_sess_opts.at(vaip::Context::CTX_SHARE_OPTION_KEY)
                  << "\" to int, disable context sharing.";
    }
  }
  if (!share_context) {
    throw std::runtime_error("must enable share context in provider options");
  }

  auto device_id = 0;
  auto context_id = 0;
  context_ = vaip::Context::create_shared_context(*context, device_id,
                                                  context_id, xclbin_file);

  // Get attributes
  auto attrs = context_->get_attrs();

  LOG_THIS(1) << "Using XRT device from vaip";

  // Create kernel
  context_->create_kernel(kernel_name_, kernel_name_);

  LOG_THIS(1) << "Kernel Name: " << kernel_name_;

  auto device = attrs->get_attr<xrt::device*>("xrt_device");
  auto kernel = attrs->get_attr<xrt::kernel*>(kernel_name_.c_str());

  // Create compute kernel
  kernel_ = std::make_unique<FDPOST>(*device, *kernel);

  // Create Sub BO
  auto sub_bo_size = ((kernel_->get_instr_size() + 31) / 32) * 32;

  // get sub-bo offset
  size_t offset = 0;
  if (attrs->has_attr("bo_offset"))
    offset = attrs->get_attr<size_t>("bo_offset");

  LOG_THIS(1) << "SRAM BO Start offset: " << offset;

  auto bo_sram = attrs->get_attr<xrt::bo*>("bo_sram");
  // create sub-bo for kernel
  instr_bo_ = xrt::bo(*bo_sram, sub_bo_size, offset);
  offset += sub_bo_size;

  LOG_THIS(1) << "SRAM BO End offset: " << offset;

  // update offset for sub-bo creation
  attrs->set_attr<size_t>("bo_offset", offset);

  // Sync Instructions
  kernel_->sync_instructions(instr_bo_);
#else
  ReadFromFile<float, float, 4>(std::string("C:\\ssd_anchors.txt"), anchors_);
#endif
}

MyCustomOp::~MyCustomOp() {}

#ifdef FDPOST_CPU_KERNEL
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

static std::vector<size_t> ValiateAndFilterBoxes(const float* boxes,
                                                 const float* scores,
                                                 size_t spatial_dim) {
  constexpr float score_threshold = 0.5f;
  std::vector<size_t> keepIdx;
  for (auto i = 0u; i < spatial_dim; ++i) {
    auto box_ptr = boxes + i * 4;
    // Validate box
    if ((box_ptr[0] >= box_ptr[2]) || (box_ptr[1] >= box_ptr[3]))
      continue;
    if ((box_ptr[0] < 0.0f) || (box_ptr[0] > 1.0f))
      continue;
    if ((box_ptr[1] < 0.0f) || (box_ptr[1] > 1.0f))
      continue;
    if ((box_ptr[2] < 0.0f) || (box_ptr[2] > 1.0f))
      continue;
    if ((box_ptr[3] < 0.0f) || (box_ptr[3] > 1.0f))
      continue;
    if (scores[i] < score_threshold)
      continue;
    keepIdx.push_back(i);
  }
  return keepIdx;
}

std::ostream& operator<<(std::ostream& str, const Box& box) {
  str << "[(" << box.x1_ << "," << box.y1_ << "),(" << box.x2_ << "," << box.y2_
      << ")]";
  return str;
}

static void
NMS(const float nms_threshold, size_t batch_idx, size_t cls_idx,
    size_t spatial_dimension, const float* boxes, const float* scores,
    std::vector<std::tuple<float, float, float, float, float, float>>& ret) {

  // Validate and Filter boxes
  auto keep = ValiateAndFilterBoxes(boxes, scores, spatial_dimension);

  // Sort filtered boxes by score values
  std::vector<std::pair<size_t, float>> order(keep.size());
  for (auto i = 0u; i < keep.size(); ++i) {
    order[i].first = keep[i];
    order[i].second = scores[keep[i]];
  }
  std::sort(
      order.begin(), order.end(),
      [](const std::pair<size_t, float>& ls,
         const std::pair<size_t, float>& rs) { return ls.second > rs.second; });

  // IOU Compute
  std::vector<size_t> selected_indices;
  std::vector<bool> exist_box(keep.size(), true);
  for (auto i = 0u; i < keep.size(); ++i) {
    auto idx = order[i].first;
    auto box_ptr = boxes + idx * 4;

    if (!exist_box[i]) {
      continue;
    }
    selected_indices.emplace_back(idx);
    for (auto j = i + 1; j < keep.size(); ++j) {
      auto kept_idx = order[j].first;
      if (!exist_box[j]) {
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
        exist_box[j] = false;
      }
    }
  }

  // Fill output buffer
  for (auto i = 0u; i < order.size(); ++i) {
    if (std::find(selected_indices.begin(), selected_indices.end(),
                  order[i].first) != selected_indices.end()) {
      auto box_ptr = boxes + order[i].first * 4;
      auto score = order[i].second;
      ret.emplace_back((float)box_ptr[0], (float)box_ptr[1], (float)box_ptr[2],
                       (float)box_ptr[3], (float)score, (float)cls_idx);
    }
  }
}

static std::vector<std::tuple<float, float, float, float, float, float>>
NMSSelectBox(const float nms_threshold, size_t num_batches,
             size_t spatial_dimension, size_t num_classes, const float* boxes,
             const float* scores) {
  auto ret =
      std::vector<std::tuple<float, float, float, float, float, float>>();
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
#endif

void MyCustomOp::Compute(const OrtApi* api, OrtKernelContext* context) const {
  if (Ort::Global<void>::api_ == nullptr) {
    Ort::Global<void>::api_ = api;
  }
  __TIC__(DecodeAndFilterBoxCompute);
  Ort::KernelContext ctx(context);
  auto num_inputs = ctx.GetInputCount();
  auto num_outputs = ctx.GetOutputCount();

  LOG_THIS(1) << "num_inputs " << num_inputs << " "
              << "num_outputs " << num_outputs << " ";

  for (auto idx = 0u; idx < num_inputs; ++idx) {
    auto input_tensor = ctx.GetInput(idx);
    auto tensor_info = input_tensor.GetTensorTypeAndShapeInfo();
    auto tensor_type = tensor_info.GetElementType();
    auto tensor_shape = tensor_info.GetShape();
    LOG_THIS(1) << "Input -> Tensor type: " << tensor_type << ", "
                << "Shape: " << shape_to_string(tensor_shape);
  }

  CHECK_EQ(num_inputs, 2u);
  CHECK_EQ(num_outputs, 1u);

  // Decode Box
  auto encoded_boxes = ctx.GetInput(0);
  auto boxes_shape = encoded_boxes.GetTensorTypeAndShapeInfo().GetShape();
  // Score tensors & shape
  auto scores = ctx.GetInput(1);
  auto scores_shape = scores.GetTensorTypeAndShapeInfo().GetShape();

  // Raw input data
  auto boxes_raw = static_cast<const float*>(encoded_boxes.GetTensorRawData());
  auto score_raw = static_cast<const float*>(scores.GetTensorRawData());

  // Number of total boxes
  auto spatial_dimension = boxes_shape[1];
  // batch size
  auto num_batches = boxes_shape[0];

  CHECK_EQ(boxes_shape.size(), 3u);
  CHECK_EQ(boxes_shape[2], 10u);
  CHECK_EQ(num_batches, 1u);

#ifndef FDPOST_CPU_KERNEL
  // Fill input data with pad to BOs
  auto boxes_padded = kernel_->get_host_buffer_boxes();
  auto score_padded = kernel_->get_host_buffer_scores();

  int idx = 0;
  auto dims = boxes_shape[2];
  // Pad boxes
  for (int spd = 0; spd < spatial_dimension; ++spd) {
    for (int d = 0; d < dims; ++d) {
      // get value in fixed point
      boxes_padded[idx++] =
          static_cast<uint8_t>(boxes_raw[(spd * dims) + d] * 8);
    }
    boxes_padded[idx++] = 0;
    boxes_padded[idx++] = 0;
  }

  idx = 0;
  dims = scores_shape[2];
  // Pad boxes
  for (int spd = 0; spd < spatial_dimension; ++spd) {
    for (int d = 0; d < dims; ++d) {
      // get value in fixed point
      score_padded[idx++] =
          static_cast<uint8_t>(score_raw[(spd * dims) + d] * 128);
    }
    score_padded[idx++] = 0;
    score_padded[idx++] = 0;
  }

  // Run AIE Kernel
  auto attrs = context_->get_attrs();
  if (attrs->has_attr(kernel_name_.c_str()) && attrs->has_attr("bo_sram")) {
    auto bo_sram = const_cast<xrt::bo*>(&instr_bo_);
    auto kernel = attrs->get_attr<xrt::kernel*>(kernel_name_.c_str());
    kernel_->exec(*kernel, *bo_sram);
  } else {
    throw std::runtime_error("Could not find kernel/BO");
  }

  auto aie_detected_boxes = kernel_->get_host_buffer_out_box();
  auto n_aie_det_boxes = aie_detected_boxes[0];

  // process
  for (auto idx = 0u; idx < num_outputs; ++idx) {
    auto output_tensor = ctx.GetOutput(idx, {(int64_t)n_aie_det_boxes, 6});
    auto tensor_info = output_tensor.GetTensorTypeAndShapeInfo();
    auto tensor_type = tensor_info.GetElementType();
    auto tensor_shape = tensor_info.GetShape();
    LOG_THIS(1) << "Output -> Tensor type: " << tensor_type << ", "
                << "Shape: " << shape_to_string(tensor_shape);
  }

  auto output_tensor = ctx.GetOutput(0, {(int64_t)n_aie_det_boxes, 6});

  auto out = output_tensor.GetTensorMutableData<float>();
  auto aie_box_ptr = aie_detected_boxes + 1;
  for (int bidx = 0; bidx < n_aie_det_boxes; ++bidx) {
    auto fscale = static_cast<float>(box_scale_);
    auto ymin = static_cast<float>(aie_box_ptr[0]) / fscale;
    auto xmin = static_cast<float>(aie_box_ptr[1]) / fscale;
    auto ymax = static_cast<float>(aie_box_ptr[2]) / fscale;
    auto xmax = static_cast<float>(aie_box_ptr[3]) / fscale;
    auto score = 1.0f;
    auto classid = 1.0f;
    out[0] = ymin;
    out[1] = xmin;
    out[2] = ymax;
    out[3] = xmax;
    out[4] = score;
    out[5] = classid;
    out += 6;
    aie_box_ptr += 4;
  }
#else // FDPOST_CPU_KERNEL
  // Decoded boxes
  std::vector<float> decoded_boxes;
  // Decode Boxes
  for (int bidx = 0; bidx < spatial_dimension; ++bidx) {
    auto box_raw = boxes_raw + bidx * 10;
    auto ycenter =
        ((box_raw[0] / 10.0f) * anchors_[bidx][2]) + anchors_[bidx][0];
    auto xcenter =
        ((box_raw[1] / 10.0f) * anchors_[bidx][3]) + anchors_[bidx][1];

    auto half_h = 0.5f * (std::expf(box_raw[2] / 5.0f)) * anchors_[bidx][2];
    auto half_w = 0.5f * (std::expf(box_raw[3] / 5.0f)) * anchors_[bidx][3];
    auto y_min = ycenter - half_h;
    auto x_min = xcenter - half_w;
    auto y_max = ycenter + half_h;
    auto x_max = xcenter + half_w;
    decoded_boxes.push_back(y_min);
    decoded_boxes.push_back(x_min);
    decoded_boxes.push_back(y_max);
    decoded_boxes.push_back(x_max);
  }

  auto iou_threshold = 0.6000000238418579f;
  auto num_classes = scores_shape[2];

  // Transpose scores
  std::vector<float> score_tr(scores_shape[0] * scores_shape[1] *
                              scores_shape[2]);

  for (auto sd = 0; sd < spatial_dimension; ++sd) {
    for (auto nc = 0; nc < num_classes; ++nc) {
      score_tr[nc * spatial_dimension + sd] = score_raw[sd * num_classes + nc];
    }
  }

  auto detected_boxes = NMSSelectBox(
      iou_threshold, num_batches, spatial_dimension, num_classes,
      static_cast<const float*>(decoded_boxes.data()), score_tr.data());

  // process
  for (auto idx = 0u; idx < num_outputs; ++idx) {
    auto output_tensor =
        ctx.GetOutput(idx, {(int64_t)detected_boxes.size(), 6});
    auto tensor_info = output_tensor.GetTensorTypeAndShapeInfo();
    auto tensor_type = tensor_info.GetElementType();
    auto tensor_shape = tensor_info.GetShape();
    LOG_THIS(1) << "Output -> Tensor type: " << tensor_type << ", "
                << "Shape: " << shape_to_string(tensor_shape);
  }

  auto output_tensor = ctx.GetOutput(0, {(int64_t)detected_boxes.size(), 6});

  auto out_base = output_tensor.GetTensorMutableData<float>();
  auto out = out_base;
  for (auto& x : detected_boxes) {
    auto ymin = std::get<0>(x);
    auto xmin = std::get<1>(x);
    auto ymax = std::get<2>(x);
    auto xmax = std::get<3>(x);
    auto score = std::get<4>(x);
    auto classid = std::get<5>(x);
    out[0] = ymin;
    out[1] = xmin;
    out[2] = ymax;
    out[3] = xmax;
    out[4] = score;
    out[5] = classid;
    out = out + 6;
  }
#endif
  __TOC__(DecodeAndFilterBoxCompute);
}
} // namespace vaip_decode_filter_boxes_custom_op
