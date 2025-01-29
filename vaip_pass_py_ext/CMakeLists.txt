#
#  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
#  Licensed under the MIT License.
#

cmake_minimum_required(VERSION 3.17)
project(
  vaip
  VERSION 1.0.0
  LANGUAGES C CXX)

find_package(unilog)

vai_add_library(NAME pass_py_ext INCLUDE_DIR include SRCS src/pass_main.cpp)

if(CMAKE_CROSSCOMPILING)
  set_property(
    TARGET pass_py_ext
    APPEND
    PROPERTY INCLUDE_DIRECTORIES ${VAI_PYTHON_INCLUDE_DIRS})
  target_link_libraries(pass_py_ext PRIVATE -l${VAI_PYTHON_LIB})
else(CMAKE_CROSSCOMPILING)
  target_link_libraries(pass_py_ext PRIVATE ${PYTHON_LIBRARIES})
  set_property(
    TARGET pass_py_ext
    APPEND
    PROPERTY INCLUDE_DIRECTORIES "${PYTHON_INCLUDE_DIRS}")
endif(CMAKE_CROSSCOMPILING)

target_link_libraries(pass_py_ext PRIVATE vaip::core vart::util glog::glog
                                          pybind11::embed)



set(voe_sources
  src/common.hpp
  src/vaip_pass_py_ext_py_module.cpp
  src/util.cpp
  src/util.hpp
)
if(USE_STATIC_PYTHON)
  pybind11_add_module(
    voe_cpp2py_export
    STATIC
    ${voe_sources}
  )
else()
  vai_add_pybind11_module(
  voe_cpp2py_export
  PACKAGE_NAME
  voe
  ${voe_sources})
endif()

target_link_libraries(voe_cpp2py_export PRIVATE vaip::core vart::util
                                                glog::glog)
if (USE_STATIC_PYTHON)
  install(
    TARGETS voe_cpp2py_export
    EXPORT voe_cpp2py_export-targets
    RUNTIME DESTINATION bin
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib)
  install(
    EXPORT voe_cpp2py_export-targets
    NAMESPACE vaip::
    COMPONENT base
    DESTINATION share/cmake/vaip)
    file(APPEND ${CMAKE_BINARY_DIR}/components.txt "voe_cpp2py_export\n")
endif()