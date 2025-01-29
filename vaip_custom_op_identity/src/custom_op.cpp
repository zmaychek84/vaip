/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

#include "onnxruntime_api.hpp"

#include <fstream>
#include <glog/logging.h>
#include <sstream>
//
#include "./custom_op.hpp"

namespace vaip_identity_custom_op {

MyCustomOp::MyCustomOp(std::shared_ptr<const PassContext> context,
                       const std::shared_ptr<MetaDefProto>& meta_def,
                       onnxruntime::Model* model)
    : CustomOpImp(context, meta_def, model) {
  std::cout << " Custom Identity op constructed." << std::endl;
  std::string const_filename = meta_def->generic_param().at("const_filename");
  unsigned int size = (unsigned int)fs::file_size(const_filename);
  num_elements_ = stoi(meta_def->generic_param().at("num_elements"));
  if (meta_def->generic_param().at("const_dtype") == "uint8") {
    dtype_ = "uint8";
    const_int_.reserve(size);
    auto infile1 =
        std::ifstream(const_filename, std::ios::in | std::ios::binary);
    infile1.read((char*)const_int_.data(), size);
    infile1.close();
  } else {
    dtype_ = "float";
    const_fl_.reserve(size / 4);
    auto infile1 =
        std::ifstream(const_filename, std::ios::in | std::ios::binary);
    infile1.read((char*)const_fl_.data(), size);
    infile1.close();
  }
  std::cout << "num of elements " << num_elements_ << std::endl;
}

MyCustomOp::~MyCustomOp() {
  // std::cout << " Custom Identity op destructed." <<std::endl;
}

void MyCustomOp::Compute(const OrtApi* api, OrtKernelContext* context) const {
  // std::cout << " Custom identity computed." <<std::endl;
  if (Ort::Global<void>::api_ == nullptr) {
    Ort::Global<void>::api_ = api;
  }

  Ort::KernelContext ctx(context);
  auto output_tensor = ctx.GetOutput(0, {num_elements_});
  if (dtype_ == "uint8") {
    auto out_data = output_tensor.GetTensorMutableData<uint8_t>();
    for (int i = 0; i < num_elements_; i++)
      out_data[i] = const_int_[i];
  } else

  {
    auto out_data = output_tensor.GetTensorMutableData<float>();
    for (int i = 0; i < num_elements_; i++)
      out_data[i] = const_fl_[i];
  }
}
} // namespace vaip_identity_custom_op
