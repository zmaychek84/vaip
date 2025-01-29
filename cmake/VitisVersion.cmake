#
#  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
#  Licensed under the MIT License.
#


execute_process(
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMAND date +%F-%T
  OUTPUT_VARIABLE BUILD_DATE)
string(STRIP "${BUILD_DATE}" BUILD_DATE)
if ("${GIT_VERSION}" STREQUAL "")
  execute_process(
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMAND git rev-parse HEAD
    OUTPUT_VARIABLE GIT_VERSION)
endif()
if ("${PROJECT_GIT_COMMIT_ID}" STREQUAL "")
  execute_process(
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMAND git rev-parse HEAD
    OUTPUT_VARIABLE PROJECT_GIT_COMMIT_ID
    OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()

string(STRIP "${GIT_VERSION}" GIT_VERSION)

if (DEFINED ENV{BUILD_ID})
  set(BUILD_ID "$ENV{BUILD_ID}")
else()
  set(BUILD_ID "")
endif()

configure_file(${CMAKE_CURRENT_LIST_DIR}/vitis_version.c.in ${CMAKE_CURRENT_BINARY_DIR}/version.c)
