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

cmake_minimum_required(VERSION 3.5)
project(unit-test VERSION 1.0.0 LANGUAGES C CXX)
include(CTest)
enable_testing()
find_package(PythonInterp REQUIRED)
cmake_policy(SET CMP0135 NEW)
find_package(googletest REQUIRED)


file(DOWNLOAD https://xcoartifactory.xilinx.com/artifactory/PHX_test_case_package/vaip_model/pt_resnet50.onnx
  ${CMAKE_CURRENT_BINARY_DIR}/pt_resnet50.onnx
  SHOW_PROGRESS
  EXPECTED_MD5 3f97582a85b8ae1be41127f326bbf9b2
  STATUS RESNET_50_ONNX_PATH_STATUS)

find_file(RESNET_50_ONNX_PATH "pt_resnet50.onnx" HINTS  ${CMAKE_CURRENT_BINARY_DIR}
 "${CMAKE_SOURCE_DIR}/../test_onnx_runner/data"
  NO_CACHE
  REQUIRED
  ENV VAIP_DATA_PATH)
message(STATUS "Found test model ${RESNET_50_ONNX_PATH}")
file(TO_NATIVE_PATH "${RESNET_50_ONNX_PATH}" RESNET_50_ONNX_PATH)

if(NOT RESNET_50_ONNX_PATH)
  message(WARNING "cannot find model pt_resnet50.onnx for testing. RESNET_50_ONNX_PATH=${RESNET_50_ONNX_PATH}")
else(NOT RESNET_50_ONNX_PATH)
  message(STATUS "Found model pt_resnet50.onnx for testing. ${RESNET_50_ONNX_PATH}")
endif(NOT RESNET_50_ONNX_PATH)
string(REPLACE "\\" "\\\\" RESNET_50_ONNX_PATH "${RESNET_50_ONNX_PATH}")
add_compile_definitions(RESNET_50_PATH="${RESNET_50_ONNX_PATH}")

configure_file(debug_logger.hpp.in ${CMAKE_CURRENT_BINARY_DIR}/debug_logger.hpp @ONLY)
configure_file(unit_test_env_params.hpp.in ${CMAKE_CURRENT_BINARY_DIR}/unit_test_env_params.hpp @ONLY)
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
find_package(vart COMPONENTS util REQUIRED)
## add a new test for class vaip_cxx::Model
set(TEST_EXE_NAME vaip_unit_tests)
add_executable(${TEST_EXE_NAME}
  main.cpp
  vaip/test_config.cpp
  vaip/test_model.cpp
  vaip/test_graph.cpp
  vaip/test_const_data.cpp
  vaip/test_anchor_point.cpp
  vaip/test_immutable_map.cpp
  vaip/test_pattern.cpp
  vaip/test_pass_context.cpp
  vaip/test_node_builder.cpp
  getenv.cpp
  getenv.c
  test_onnx_runner/test_onnx_runner.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/debug_logger.hpp
  ${CMAKE_CURRENT_BINARY_DIR}/unit_test_env_params.hpp
)
find_library(ORT_LIBRARY onnxruntime HINTS "${CMAKE_INSTALL_PREFIX}/lib" REQUIRED)
target_link_libraries(${TEST_EXE_NAME} PRIVATE vaip::onnxruntime_vitisai_ep glog::glog ${ORT_LIBRARY} googletest::gtest)
if(MSVC)
## pitfalls for debugging these test programm, most of test
## programs, e.g. onnx_grep etc, depends on onnxruntime.dll, but
## onnxruntime.dll load onnxruntime_vitisai_ep.dll from the same
## directory, we have to copy these 3 dll files here to make sure only
## one version of onnxruntime_vitisai_ep.dll is loaded.
    add_custom_command(
        TARGET ${TEST_EXE_NAME}
        POST_BUILD
        COMMAND
          ${CMAKE_COMMAND} -E copy
          $<TARGET_FILE:vaip::onnxruntime_vitisai_ep>
          ${CMAKE_INSTALL_PREFIX}/bin/onnxruntime.dll
          ${CMAKE_INSTALL_PREFIX}/bin/onnxruntime_providers_shared.dll
          ${CMAKE_INSTALL_PREFIX}/bin/onnxruntime_providers_vitisai.dll
          ${CMAKE_INSTALL_PREFIX}/xrt/xrt_coreutil.dll
          $<TARGET_FILE_DIR:${TEST_EXE_NAME}>
    )
endif(MSVC)
add_custom_command(
        TARGET ${TEST_EXE_NAME}
        POST_BUILD
        COMMAND
         ${CMAKE_COMMAND} -E echo
          ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/vaip/test_constant_initializer.py ${CMAKE_CURRENT_BINARY_DIR}/test_constant_initializer.onnx
        COMMAND
          ${PYTHON_EXECUTABLE} -m pip install onnx==1.16.0
        COMMAND
          ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/vaip/test_constant_initializer.py ${CMAKE_CURRENT_BINARY_DIR}/test_constant_initializer.onnx
        COMMAND
          ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/vaip/create_sample_onnx_model.py ${CMAKE_CURRENT_BINARY_DIR}/sample.onnx
          )
add_custom_command(
        TARGET ${TEST_EXE_NAME}
        POST_BUILD
        COMMAND
         ${CMAKE_COMMAND} -E echo
          ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/vaip/test_custom_op.py ${CMAKE_CURRENT_BINARY_DIR}/test_custom_op.onnx
        COMMAND
          ${PYTHON_EXECUTABLE} -m pip install onnx==1.16.0
        COMMAND
          ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/vaip/test_custom_op.py ${CMAKE_CURRENT_BINARY_DIR}/test_costom_op.onnx
          )
if(WIN32)
  set(PATH_ENV_VAR_NAME "PATH")
else(WIN32)
  set(PATH_ENV_VAR_NAME "LD_LIBRARY_PATH")
endif(WIN32)
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/unit_test_env.txt" VAIP_UNIT_TEST_ENV_IN)
string(CONFIGURE "${VAIP_UNIT_TEST_ENV_IN}" VAIP_UNIT_TEST_ENV @ONLY)
string(REPLACE "\n" ";" VAIP_UNIT_TEST_ENV_LIST ${VAIP_UNIT_TEST_ENV})
message(STATUS "LIST OF ENVIRONMENT VARIABLES in ${CMAKE_CURRENT_SOURCE_DIR}/unit_test_env.txt :")
configure_file("vs.runsettings.in" ${CMAKE_BINARY_DIR}/.runsettings)
set(TEST_ENV "")
foreach(element IN LISTS VAIP_UNIT_TEST_ENV_LIST)
  message(STATUS "   ENV VAR FOR TESTING:  ${element}")
  list(APPEND TEST_ENV ENVIRONMENT "${element}")
endforeach(element IN LISTS TMP_COMPONENT_CONTENT)
include(GoogleTest)
gtest_discover_tests(${TEST_EXE_NAME} DISCOVERY_MODE "POST_BUILD" PROPERTIES
  ${TEST_ENV}
  DISCOVERY_MODE "PRE_TEST"
)

# configure_file(set_environment_variable.cmake.in "${CMAKE_CURRENT_BINARY_DIR}/set_environment_variable.cmake" @ONLY)
# set_property(DIRECTORY APPEND PROPERTY TEST_INCLUDE_FILES "${CMAKE_CURRENT_BINARY_DIR}/set_environment_variable.cmake")
