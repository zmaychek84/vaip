/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */
#include "initialize_vaip.hpp"
#include "vaip/vaip.hpp"
#include "vaip/vaip_ort.hpp"
using namespace vaip_core;
using namespace std;
int main(int argc, char* argv[]) {
  initialize_vaip();
  auto model = model_load(argv[1]);
  cout << "OK" << endl;
  return 0;
}
