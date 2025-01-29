/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

#include <cstdlib>
#include <iostream>
#include <vector>
using namespace std;
#define VAIP_EXPORT_DLL 0
#include "vaip/vaip.hpp"
int main(int argc, char* argv[]) {
  vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f,  5.0f,  6.0f,
                     7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
  vector<float> y(12, 0.0f);
  vaip_core::transpose_f(&x[0], &y[0], {1, 3, 2, 2}, {0, 2, 3, 1});
  //  vaip_core::transpose_f(&x[0], &y[0], {1, 3, 2, 2}, {2, 0, 1, 3});
  for (auto& v : y) {
    cout << v << endl;
  }
  cout << "VAIP_USER " << VAIP_USER << " " << endl;
  return 0;
}
