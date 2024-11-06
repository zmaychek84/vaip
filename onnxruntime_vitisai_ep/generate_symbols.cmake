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
function(vaip_ort_add_dependency)
  set(options "")
  set(oneValueArgs OUTPUT)
  set(multiValueArgs DEPENDS)
  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}"
                        ${ARGN})
  find_package(Python COMPONENTS Interpreter)
  set(_dependencies "")
  message("ARG_OUTPUT is ${ARG_OUTPUT}")
  foreach(_target ${ARG_DEPENDS})
    if(NOT MSVC)
      get_target_property(_imported_target ${_target} IMPORTED)
      if(_imported_target) # for external project
        get_target_property(_target_lib_loc ${_target} LOCATION)
        get_filename_component(_target_lib_loc.sym ${_target_lib_loc} NAME)
        set(_target_lib_loc.sym
            "${CMAKE_CURRENT_BINARY_DIR}/${_target_lib_loc.sym}")
        add_custom_command(
          OUTPUT ${_target_lib_loc.sym}
          MAIN_DEPENDENCY ${_target_lib_loc}
          DEPENDS ${CMAKE_CURRENT_LIST_DIR}/generate_symbols_script.cmake
          COMMAND
            ${CMAKE_COMMAND} -P
            ${CMAKE_CURRENT_LIST_DIR}/generate_symbols_script.cmake
            ${_target_lib_loc} ${_target_lib_loc.sym})
        list(APPEND _dependencies ${_target_lib_loc.sym})
      endif(_imported_target)
    endif(NOT MSVC)
    if(MSVC) # build single dll
      target_link_libraries(onnxruntime_vitisai_ep PRIVATE ${_target})
    endif(MSVC)
  endforeach(_target)
  if(NOT MSVC)
    add_custom_command(
      OUTPUT ${ARG_OUTPUT}
      COMMAND Python::Interpreter ${CMAKE_CURRENT_LIST_DIR}/generate_sym_cpp.py
              ${ARG_OUTPUT} ${_dependencies}
      DEPENDS ${_dependencies} ${CMAKE_CURRENT_LIST_DIR}/generate_sym_cpp.py)
    get_filename_component(FILE_BASENAME ${ARG_OUTPUT} NAME_WE)
    add_custom_target(${FILE_BASENAME} ALL DEPENDS ${ARG_OUTPUT})
  endif(NOT MSVC)
endfunction(vaip_ort_add_dependency)

list(APPEND LIBS vart::dummy-runner)
vaip_ort_add_dependency(OUTPUT ${CMAKE_CURRENT_LIST_DIR}/src/symbols.cpp
                        DEPENDS ${LIBS})
