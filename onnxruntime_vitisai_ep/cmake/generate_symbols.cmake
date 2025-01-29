#
#  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
#  Licensed under the MIT License.
#
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
