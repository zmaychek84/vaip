#
#  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
#  Licensed under the MIT License.
#
include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/vaip_common.cmake)

option(WITH_OPENSSL "include openssl or not" OFF)
if(WITH_OPENSSL)
  find_package(OpenSSL REQUIRED)
endif(WITH_OPENSSL)

vai_add_library(
  NAME
  encryption
  SRCS
  src/encryption.cpp
  src/encryption.hpp)


if(WITH_OPENSSL)
  target_compile_definitions(encryption PUBLIC -DWITH_OPENSSL)
  target_link_libraries(encryption PRIVATE OpenSSL::SSL OpenSSL::Crypto ${CMAKE_DL_LIBS})
  if (MSVC)
    target_link_libraries(encryption PRIVATE WS2_32.LIB GDI32.LIB ADVAPI32.LIB CRYPT32.LIB USER32.LIB)
  endif()
endif(WITH_OPENSSL)
