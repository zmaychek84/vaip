#
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
write_file("${CMAKE_CURRENT_BINARY_DIR}/reserved_symbols.cpp1.inc" "")
write_file("${CMAKE_CURRENT_BINARY_DIR}/reserved_symbols.cpp2.inc" "")
message(STATUS "Generating ${CMAKE_CURRENT_BINARY_DIR}/reserved_symbols.cpp1.inc")
message(STATUS "Generating ${CMAKE_CURRENT_BINARY_DIR}/reserved_symbols.cpp2.inc")


foreach(symbol_txt_file IN LISTS SYMBOL_TXT_FILES)
    message(STATUS "Processing symbols in ${symbol_txt_file}")
    file(STRINGS ${symbol_txt_file} TMP_SYMBOL_CONTENT)
    foreach(sym IN LISTS TMP_SYMBOL_CONTENT)
        write_file("${CMAKE_CURRENT_BINARY_DIR}/reserved_symbols.cpp1.inc" "extern \"C\" void * ${sym};\n" APPEND)
        write_file("${CMAKE_CURRENT_BINARY_DIR}/reserved_symbols.cpp2.inc" "${sym},\n" APPEND)
    endforeach(sym IN LISTS TMP_SYMBOL_CONTENT)
endforeach(symbol_txt_file in LISTS SYMBOL_TXT_FILES)
