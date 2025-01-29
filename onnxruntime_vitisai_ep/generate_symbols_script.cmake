#
#  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
#  Licensed under the MIT License.
#
set(_so_file ${CMAKE_ARGV3})
set(_symbol_file ${CMAKE_ARGV4})
execute_process(
  COMMAND readelf --wide --symbols ${_so_file}
  COMMAND awk "$5 == \"GLOBAL\" && $8 ~ /_hook$/ {print $8}"
  COMMAND sort
  COMMAND uniq
  OUTPUT_FILE ${_symbol_file} COMMAND_ECHO STDOUT)
