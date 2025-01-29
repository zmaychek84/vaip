/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

#include <gtest/gtest.h>
#ifdef __GNUC__
#  pragma GCC diagnostic ignored "-Wpedantic"
#  pragma GCC diagnostic ignored "-Wconversion"
#  pragma GCC diagnostic ignored "-Wsign-compare"
#  pragma GCC diagnostic ignored "-Wunused-variable"
#  pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#endif
#include <onnxruntime_cxx_api.h>

int main(int argc, char** argv) {
  auto env =
      std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_ERROR, "vaip_unit_test");
  Ort::SessionOptions().AppendExecutionProvider_VitisAI();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
