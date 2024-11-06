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

#include <glog/logging.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <opencv2/opencv.hpp>

#include "core/framework/compute_capability.h" // For ‘onnxruntime::ComputeCapability’
#include "core/framework/execution_provider.h" // for IExecutionProvider
#include "core/graph/model.h"
#include "core/providers/vitisai/vitisai_provider_factory.h"
#include "core/session/abi_session_options_impl.h" // for OrtSessionOptions
#include "core/session/environment.h" //  for  class onnxruntime::Environment
#include "core/session/experimental_onnxruntime_cxx_api.h"
// #include "core/session/onnxruntime_cxx_api.h"
#include "core/session/ort_env.h" // for OrtEnv
#include "vaip/xir_ops/xir_ops_defs.hpp"
#include "vitis/ai/env_config.hpp"
DEF_ENV_PARAM(XLNX_ENABLE_DUMP, "0");

using namespace std;
using namespace onnxruntime;

static std::vector<cv::Mat> read_images(const std::vector<std::string>& files,
                                        int64_t batch);
static cv::Mat croppedImage(const cv::Mat& image, int height, int width);
static cv::Mat preprocess_image(const cv::Mat& image, cv::Size size);
static void set_input_image(const cv::Mat& image, float* data);
static std::vector<float> softmax(float* data, int64_t size);
static std::vector<std::pair<int, float>> topk(const std::vector<float>& score,
                                               int K);
static void print_topk(const std::vector<std::pair<int, float>>& topk);
static const char* lookup(int index);

// resnet50 preprocess
static void preprocess_resnet50(const std::vector<std::string>& files,
                                std::vector<float>& input_tensor_values,
                                std::vector<int64_t>& input_shape) {
  auto batch = input_shape[0];
  auto channel = input_shape[1];
  auto height = input_shape[2];
  auto width = input_shape[3];
  auto batch_size = channel * height * width;

  auto size = cv::Size((int)width, (int)height);
  auto images = read_images(files, batch);
  CHECK_EQ(images.size(), batch)
      << "images number be read into input buffer must be equal to batch";

  for (auto index = 0; index < batch; ++index) {
    auto resize_image = preprocess_image(images[index], size);
    set_input_image(resize_image,
                    input_tensor_values.data() + batch_size * index);
    if (ENV_PARAM(XLNX_ENABLE_DUMP)) {
      for (auto i = 0; i < 64; i++) {
        if (i % 16 == 0)
          std::cout << "\n";
        std::cout << input_tensor_values[batch_size * index + i] << "  ";
      }
      auto filename =
          "/tmp/onnx_input_batch_" + std::to_string(index) + "_float.bin";
      auto mode =
          std::ios_base::out | std::ios_base::binary | std::ios_base::trunc;
      CHECK(std::ofstream(filename, mode)
                .write((char*)input_tensor_values.data() + batch_size * index,
                       batch_size * sizeof(float))
                .good())
          << " faild to write to " << filename;
    }
  }
}

// resnet50 postprocess
static void postprocess_resnet50(Ort::Value& output_tensor) {
  auto output_shape = output_tensor.GetTensorTypeAndShapeInfo().GetShape();
  auto batch = output_shape[0];
  auto channel = output_shape[1];
  auto output_tensor_ptr = output_tensor.GetTensorMutableData<float>();
  for (auto index = 0; index < batch; ++index) {
    if (ENV_PARAM(XLNX_ENABLE_DUMP)) {
      auto filename =
          "/tmp/onnx_output_batch_" + std::to_string(index) + "_float.bin";
      auto size = channel * sizeof(float);
      auto mode =
          std::ios_base::out | std::ios_base::binary | std::ios_base::trunc;
      CHECK(std::ofstream(filename, mode)
                .write((char*)output_tensor_ptr + size * index, size)
                .good())
          << " faild to write to " << filename;
    }
    auto softmax_output = softmax(output_tensor_ptr + channel * index, channel);
    auto tb_top5 = topk(softmax_output, 5);
    std::cout << "batch_index: " << index << std::endl;
    print_topk(tb_top5);
  }
}

#define CHECK_STATUS_OK(expr)                                                  \
  do {                                                                         \
    Status _tmp_status = (expr);                                               \
    CHECK(_tmp_status.IsOK()) << _tmp_status;                                  \
  } while (0)

static void CheckStatus(OrtStatus* status) {
  if (status != NULL) {
    const char* msg = Ort::GetApi().GetErrorMessage(status);
    fprintf(stderr, "%s\n", msg);
    Ort::GetApi().ReleaseStatus(status);
    exit(1);
  }
}

// pretty prints a shape dimension vector
static std::string print_shape(const std::vector<int64_t>& v) {
  std::stringstream ss("");
  for (size_t i = 0; i < v.size() - 1; i++)
    ss << v[i] << "x";
  ss << v[v.size() - 1];
  return ss.str();
}

static int calculate_product(const std::vector<int64_t>& v) {
  int total = 1;
  for (auto& i : v)
    total *= (int)i;
  return total;
}

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cerr << "usage :" << argv[0] << " <model_name>"
              << " <image_url> [<image_url> ...]" << std::endl;
    abort();
  }
  auto model_name = std::string(argv[1]);
  std::vector<std::string> g_image_files;
  for (auto i = 2; i < argc; i++) {
    g_image_files.push_back(std::string(argv[i]));
  }

  vaip::register_xir_ops();
  auto env = Ort::Env();
  auto logger = ((OrtEnv*)env)
                    ->GetEnvironment()
                    .GetLoggingManager()
                    ->CreateLogger("VAIP");
  // auto model = load_model(std::string(argv[1]), *logger);
  auto session_options = Ort::SessionOptions();
  const char* export_runtime_module = "abc";
  const char* load_runtime_module = "abc";
  CheckStatus(OrtSessionOptionsAppendExecutionProvider_VITISAI(
      session_options, "VITISAI", 0 /*device id*/, export_runtime_module,
      load_runtime_module));
  auto session = Ort::Experimental::Session(env, model_name, session_options);

  // print name/shape of inputs
  std::vector<std::string> input_names = session.GetInputNames();
  std::vector<std::vector<int64_t>> input_shapes = session.GetInputShapes();
  cout << "Input Node Name/Shape (" << input_names.size() << "):" << endl;
  for (size_t i = 0; i < input_names.size(); i++) {
    cout << "\t" << input_names[i] << " : " << print_shape(input_shapes[i])
         << endl;
  }

  // print name/shape of outputs
  std::vector<std::string> output_names = session.GetOutputNames();
  std::vector<std::vector<int64_t>> output_shapes = session.GetOutputShapes();
  cout << "Output Node Name/Shape (" << output_names.size() << "):" << endl;
  for (size_t i = 0; i < output_names.size(); i++) {
    cout << "\t" << output_names[i] << " : " << print_shape(output_shapes[i])
         << endl;
  }

  // Assume model has 1 input node and 1 output node.
  assert(input_names.size() == 1 && output_names.size() == 1);

  // Create a single Ort tensor of random numbers
  auto input_shape = input_shapes[0];
  int total_number_elements = calculate_product(input_shape);
  std::vector<float> input_tensor_values(total_number_elements);
  preprocess_resnet50(g_image_files, input_tensor_values, input_shape);

  std::vector<Ort::Value> input_tensors;
  input_tensors.push_back(Ort::Experimental::Value::CreateTensor<float>(
      input_tensor_values.data(), input_tensor_values.size(), input_shape));

  // double-check the dimensions of the input tensor
  assert(input_tensors[0].IsTensor() &&
         input_tensors[0].GetTensorTypeAndShapeInfo().GetShape() ==
             input_shape);
  cout << "\ninput_tensor shape: "
       << print_shape(input_tensors[0].GetTensorTypeAndShapeInfo().GetShape())
       << endl;

  // pass data through model
  cout << "Running model...";
  try {
    auto output_tensors = session.Run(session.GetInputNames(), input_tensors,
                                      session.GetOutputNames());
    cout << "done" << endl;

    // double-check the dimensions of the output tensors
    // NOTE: the number of output tensors is equal to the number of output nodes
    // specifed in the Run() call
    assert(output_tensors.size() == session.GetOutputNames().size() &&
           output_tensors[0].IsTensor());
    auto output_shape =
        output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    cout << "output_tensor_shape: " << print_shape(output_shape) << endl;
    postprocess_resnet50(output_tensors[0]);
  } catch (const Ort::Exception& exception) {
    cout << "ERROR running model inference: " << exception.what() << endl;
    exit(-1);
  }
  return 0;
}

static std::vector<cv::Mat> read_images(const std::vector<std::string>& files,
                                        int64_t batch) {
  std::vector<cv::Mat> images(batch);
  for (auto index = 0u; index < batch; ++index) {
    const auto& file = files[index % files.size()];
    images[index] = cv::imread(file);
    CHECK(!images[index].empty()) << "cannot read image from " << file;
  }
  return images;
}

static cv::Mat croppedImage(const cv::Mat& image, int height, int width) {
  cv::Mat cropped_img;
  int offset_h = (image.rows - height) / 2;
  int offset_w = (image.cols - width) / 2;
  cv::Rect box(offset_w, offset_h, width, height);
  cropped_img = image(box).clone();
  return cropped_img;
}

static cv::Mat preprocess_image(const cv::Mat& image, cv::Size size) {
  float smallest_side = 256;
  float scale = smallest_side / ((image.rows > image.cols) ? (float)image.cols
                                                           : (float)image.rows);
  cv::Mat resized_image;
  cv::resize(image, resized_image,
             cv::Size(image.cols * (int)scale, image.rows * (int)scale));
  return croppedImage(resized_image, size.height, size.width);
}

//(image_data - mean) * scale, BRG2RGB and hwc2chw
static void set_input_image(const cv::Mat& image, float* data) {
  float mean[3] = {103.53f, 116.28f, 123.675f};
  float scales[3] = {0.017429f, 0.017507f, 0.01712475f};
  for (int c = 0; c < 3; c++) {
    for (int h = 0; h < image.rows; h++) {
      for (int w = 0; w < image.cols; w++) {
        auto c_t = abs(c - 2); // BRG to RGB
        auto image_data =
            (image.at<cv::Vec3b>(h, w)[c_t] - mean[c_t]) * scales[c_t];
        data[c * image.rows * image.cols + h * image.cols + w] =
            (float)image_data;
      }
    }
  }
}

static std::vector<float> softmax(float* data, int64_t size) {
  auto output = std::vector<float>(size);
  std::transform(data, data + size, output.begin(), expf);
  auto sum = accumulate(output.begin(), output.end(), 0.0f, std::plus<float>());
  std::transform(output.begin(), output.end(), output.begin(),
                 [sum](float v) { return v / sum; });
  return output;
}

static std::vector<std::pair<int, float>> topk(const std::vector<float>& score,
                                               int K) {
  auto indices = std::vector<int>(score.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::partial_sort(indices.begin(), indices.begin() + K, indices.end(),
                    [&score](int a, int b) { return score[a] > score[b]; });
  auto ret = std::vector<std::pair<int, float>>(K);
  std::transform(
      indices.begin(), indices.begin() + K, ret.begin(),
      [&score](int index) { return std::make_pair(index, score[index]); });
  return ret;
}

static void print_topk(const std::vector<std::pair<int, float>>& topk) {
  for (const auto& v : topk) {
    std::cout << std::setiosflags(std::ios::left) << std::setw(11)
              << "score[" + std::to_string(v.first) + "]"
              << " =  " << std::setw(12) << v.second
              << " text: " << lookup(v.first)
              << std::resetiosflags(std::ios::left) << std::endl;
  }
}

static const char* lookup(int index) {
  static const char* table[] = {
#include "word_list.inc"
  };

  if (index < 0) {
    return "";
  } else {
    return table[index];
  }
}
