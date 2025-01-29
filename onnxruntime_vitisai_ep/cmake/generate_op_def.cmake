#
#  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
#  Licensed under the MIT License.
#
write_file("${CMAKE_CURRENT_BINARY_DIR}/op_def.cpp.inc" "")
message(STATUS "Generating ${CMAKE_CURRENT_BINARY_DIR}/op_def.cpp.inc")

foreach(opdef_txt_file IN LISTS OPDEF_TXT_FILES)
    message(STATUS "Processing opdef in ${opdef_txt_file}")
    file(STRINGS ${opdef_txt_file} TMP_OPDEF_CONTENT)
    foreach(sym IN LISTS TMP_OPDEF_CONTENT)
        write_file("${CMAKE_CURRENT_BINARY_DIR}/op_def.cpp.inc" "\"vaip-${sym}\"," APPEND)
    endforeach(sym IN LISTS TMP_OPDEF_CONTENT)
endforeach(opdef_txt_file in LISTS OPDEF_TXT_FILES)
