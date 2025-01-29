/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */
#include "vaip/model.hpp"
#include <gtest/gtest.h>

TEST(ModelTest, LoadModel) {
  // Test loading a valid model
  std::string modelPath = "path/to/model.onnx";
  auto model = vaip_cxx::Model::load(modelPath);
  ASSERT_TRUE(model != nullptr);
  EXPECT_EQ(model->name(), "model_name");

  // Test loading an invalid model
  std::string invalidModelPath = "path/to/invalid_model.onnx";
  auto invalidModel = vaip_cxx::Model::load(invalidModelPath);
  ASSERT_TRUE(invalidModel == nullptr);
}