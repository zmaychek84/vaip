/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

#include "../vaip/src/config.hpp"
#include "debug_logger.hpp"
#include "unit_test_env_params.hpp"
#include "vaip/vaip.hpp"
#include <filesystem>
#include <fstream>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <limits>
class ConfigTest : public DebugLogger {};
TEST_F(ConfigTest, Simple) {
  const char config[] =
      R"json(
{
   "sessionOptions": {
     "cache_dir" : "hello2",
     "cacheDir" : "hello1",
      "cache_key" : "key",
      "enable_cache_file_io_in_mem":"1"
   }
}
)json";
  auto config_proto = vaip_core::Config::parse_from_string(config);
  LOG(INFO) << "config: " << config_proto.DebugString();
  // EXPECT_EQ("hello1", config_proto.cache_dir());
  // EXPECT_TRUE(config_proto.enable_cache_file_io_in_mem());
}
