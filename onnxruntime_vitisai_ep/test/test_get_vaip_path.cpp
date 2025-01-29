/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */

// must include glog/logging before vaip.hpp
#include <glog/logging.h>
//
#include "vaip/vaip.hpp"
using namespace vaip_core;
using namespace std;
int main(int argc, char* argv[]) {
  cerr << "MY PATH = " << get_vaip_path() << endl;
  cerr << "BYE" << endl;
}
