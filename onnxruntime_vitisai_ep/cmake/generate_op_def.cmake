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
write_file("${CMAKE_CURRENT_BINARY_DIR}/op_def.cpp.inc" "")
message(STATUS "Generating ${CMAKE_CURRENT_BINARY_DIR}/op_def.cpp.inc")

foreach(opdef_txt_file IN LISTS OPDEF_TXT_FILES)
    message(STATUS "Processing opdef in ${opdef_txt_file}")
    file(STRINGS ${opdef_txt_file} TMP_OPDEF_CONTENT)
    foreach(sym IN LISTS TMP_OPDEF_CONTENT)
        write_file("${CMAKE_CURRENT_BINARY_DIR}/op_def.cpp.inc" "\"vaip-${sym}\"," APPEND)
    endforeach(sym IN LISTS TMP_OPDEF_CONTENT)
endforeach(opdef_txt_file in LISTS OPDEF_TXT_FILES)
