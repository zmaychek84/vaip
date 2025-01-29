#
#  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
#  Licensed under the MIT License.
#
cmake_minimum_required(VERSION 3.17)
project(
  onnxruntime_vitisai_ep
  VERSION 1.0.0
  LANGUAGES C CXX)
#option(ENABLE_BUILD_VOE_WHEEL ON "build voe.whl or not")
#include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/VitisVersion.cmake)

option(WITH_CPURUNNER "include vart cpu-runner or not" OFF)
find_package(vart REQUIRED util runner mem-manager dummy-runner trace
             OPTIONAL_COMPONENTS cpu-runner)
if(MSVC)
  find_package(trace-logging)
endif(MSVC)
if(MSVC)
  find_package(trace-logging)
endif(MSVC)
find_package(Eigen3)

vai_add_library(
  NAME
  onnxruntime_vitisai_ep
  SRCS
  STATIC
  src/main.cpp
  INCLUDE_DIR
  include)


if(WIN32)
  target_link_libraries(onnxruntime_vitisai_ep PUBLIC vaip::core vart::dummy-runner)
  target_compile_definitions(onnxruntime_vitisai_ep PUBLIC "-DONNX_NAMESPACE=onnx")
  get_target_property(core_include vaip::core INTERFACE_INCLUDE_DIRECTORIES)
  target_include_directories(onnxruntime_vitisai_ep PUBLIC ${core_include})
else()
  target_link_libraries(onnxruntime_vitisai_ep PUBLIC vaip::core vart::dummy-runner)
endif()

file(STRINGS "${CMAKE_BINARY_DIR}/components.txt" TMP_COMPONENT_CONTENT)

set(SYMBOL_TXT_FILES "")
foreach(element IN LISTS TMP_COMPONENT_CONTENT)
  if(element MATCHES "^pass_" OR element MATCHES "^vaip_custom_op" OR element MATCHES "^op_def")
    if(TARGET "vaip::${element}")
      message(STATUS "Linking onnxruntime_vitisai_ep to vaip::${element}")
      target_link_libraries(onnxruntime_vitisai_ep PUBLIC vaip::${element})
      get_target_property(tmp_cmake_current_source_dir "vaip::${element}" "SOURCE_DIR")
      if(EXISTS "${tmp_cmake_current_source_dir}/symbols.txt")
        message(STATUS "READ SYMBOL ${tmp_cmake_current_source_dir}/symbols.txt")
        list(APPEND SYMBOL_TXT_FILES "${tmp_cmake_current_source_dir}/symbols.txt")
      # else(EXISTS "${tmp_cmake_current_source_dir}/symbols.txt")
      #   message(STATUS "NO SYMBOL ${tmp_cmake_current_source_dir}/symbols.txt")
      endif(EXISTS "${tmp_cmake_current_source_dir}/symbols.txt")
    endif()
  endif()
endforeach()
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/reserved_symbols.cpp1.inc
         ${CMAKE_CURRENT_BINARY_DIR}/reserved_symbols.cpp2.inc
  COMMENT "Generating reserved_symbols.cpp1.inc reserved_symbols.cpp2.inc"
  DEPENDS ${SYMBOL_TXT_FILES} ${CMAKE_CURRENT_LIST_DIR}/cmake/generate_symbols.cmake
  COMMAND ${CMAKE_COMMAND} -E echo ${CMAKE_COMMAND}
  -DCMAKE_CURRENT_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}
  "-DSYMBOL_TXT_FILES=${SYMBOL_TXT_FILES}"
  -P ${CMAKE_CURRENT_LIST_DIR}/cmake/generate_symbols.cmake
  COMMAND ${CMAKE_COMMAND}
          -DCMAKE_CURRENT_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}
          "-DSYMBOL_TXT_FILES=${SYMBOL_TXT_FILES}"
          -P ${CMAKE_CURRENT_LIST_DIR}/cmake/generate_symbols.cmake
  VERBATIM)
target_sources(onnxruntime_vitisai_ep PUBLIC
               ${CMAKE_CURRENT_BINARY_DIR}/reserved_symbols.cpp1.inc
               ${CMAKE_CURRENT_BINARY_DIR}/reserved_symbols.cpp2.inc)


set(OPDEF_TXT_FILES "")
foreach(element IN LISTS TMP_COMPONENT_CONTENT)
  if(TARGET "vaip::${element}")
    get_target_property(tmp_source_dir "vaip::${element}" "SOURCE_DIR")
    if(EXISTS "${tmp_source_dir}/op_def.txt")
      list(APPEND OPDEF_TXT_FILES "${tmp_source_dir}/op_def.txt")
    endif(EXISTS "${tmp_source_dir}/op_def.txt")
  endif()
endforeach()

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/op_def.cpp.inc
  COMMENT "Generating op_def.cpp.inc"
  DEPENDS ${OPDEF_TXT_FILES} ${CMAKE_CURRENT_LIST_DIR}/cmake/generate_op_def.cmake
  COMMAND ${CMAKE_COMMAND} -E echo ${CMAKE_COMMAND}
  -DCMAKE_CURRENT_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}
  "-DOPDEF_TXT_FILES=${OPDEF_TXT_FILES}"
  -P ${CMAKE_CURRENT_LIST_DIR}/cmake/generate_op_def.cmake
  COMMAND ${CMAKE_COMMAND}
          -DCMAKE_CURRENT_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}
          "-DOPDEF_TXT_FILES=${OPDEF_TXT_FILES}"
          -P ${CMAKE_CURRENT_LIST_DIR}/cmake/generate_op_def.cmake
  VERBATIM)

target_sources(onnxruntime_vitisai_ep PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/op_def.cpp.inc)

if(TARGET vaip::vaip_pattern_zoo)
   target_link_libraries(onnxruntime_vitisai_ep PUBLIC vaip::vaip_pattern_zoo)
endif(TARGET vaip::vaip_pattern_zoo)
if(MSVC) # static linking.
  target_sources(onnxruntime_vitisai_ep PUBLIC src/symbols.cpp
                                                src/vaip_symbols.cpp)
endif(MSVC)
find_package(graph-engine QUIET)

if(TARGET graph-engine::graph-engine)
  find_package(XRT REQUIRED)
  target_link_libraries(onnxruntime_vitisai_ep
                        PUBLIC graph-engine::graph-engine)
  target_compile_definitions(onnxruntime_vitisai_ep
                             PUBLIC "-DFOUND_GRAPH_ENGINE")

  if(BUILD_SHARED_LIBS)
    target_compile_definitions(onnxruntime_vitisai_ep
                               PUBLIC -DGRAPH_ENGINE_USE_DLL=1)
  else(BUILD_SHARED_LIBS)
    target_compile_definitions(onnxruntime_vitisai_ep
                               PUBLIC -DGRAPH_ENGINE_USE_DLL=0)
  endif(BUILD_SHARED_LIBS)

endif(TARGET graph-engine::graph-engine)
if(TARGET vart::cpu-runner AND WITH_CPURUNNER)
  target_link_libraries(onnxruntime_vitisai_ep PUBLIC vart::cpu-runner)
  target_compile_definitions(onnxruntime_vitisai_ep
                             PUBLIC "-DFOUND_CPU_RUNNER")
endif()

target_compile_definitions(onnxruntime_vitisai_ep
                           PUBLIC "-DONNXRUNTIME_VITISAI_EP_EXPORT_DLL=1")
target_compile_definitions(onnxruntime_vitisai_ep
                           PUBLIC "-DVAIP_USE_DLL=1")
if(MSVC)
  find_program(DUMPBIN_EXECUTABLE dumpbin REQUIRED)
  add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/vaip_core_symbol.txt"
    # TARGET onnxruntime_vitisai_ep
    # PRE_BUILD
    COMMAND "${DUMPBIN_EXECUTABLE}" "/symbols" "$<TARGET_FILE:vaip::core>" > "${CMAKE_CURRENT_BINARY_DIR}/vaip_core_symbol.txt")

  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime_vitisai_ep.def
    COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/dump_symbol.py
      ${CMAKE_CURRENT_BINARY_DIR}/vaip_core_symbol.txt
      ${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime_vitisai_ep.def.in
      ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime_vitisai_ep.def
    DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/vaip_core_symbol.txt"
    # TARGET onnxruntime_vitisai_ep
    # PRE_BUILD
    #BYPRODUCTS ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime_vitisai_ep.def
  )
  target_sources(onnxruntime_vitisai_ep PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime_vitisai_ep.def)
endif(MSVC)
if (EN_LLM_DOD_OPS)
  target_compile_definitions(onnxruntime_vitisai_ep
                           PUBLIC "-DEN_LLM_DOD_OPS=1")
endif()

target_include_directories(
  onnxruntime_vitisai_ep
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

set_target_properties(
  onnxruntime_vitisai_ep
  PROPERTIES VERSION "${PROJECT_VERSION}"
             SOVERSION "${PROJECT_VERSION_MAJOR}"
             OUTPUT_NAME "onnxruntime_vitisai_ep")

#include(generate_symbols.cmake)
add_executable (voe_py_pass src/dummy_main.cpp)
