#
# Copyright (C) 2022 Xilinx, Inc. All rights reserved.
# Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
#
set(_so_file ${CMAKE_ARGV3})
set(_symbol_file ${CMAKE_ARGV4})
execute_process(
  COMMAND readelf --wide --symbols ${_so_file}
  COMMAND awk "$5 == \"GLOBAL\" && $8 ~ /_hook$/ {print $8}"
  COMMAND sort
  COMMAND uniq
  OUTPUT_FILE ${_symbol_file} COMMAND_ECHO STDOUT)
