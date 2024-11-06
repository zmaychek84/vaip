#   The Xilinx Vitis AI Vaip in this distribution are provided under the following free
#   and permissive binary-only license, but are not provided in source code form.  While the following free
#   and permissive license is similar to the BSD open source license, it is NOT the BSD open source license
#   nor other OSI-approved open source license.
#
#    Copyright (C) 2022 Xilinx, Inc. All rights reserved.
#    Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
#
#    Redistribution and use in binary form only, without modification, is permitted provided that the following conditions are met:
#
#    1. Redistributions must reproduce the above copyright notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the distribution.
#
#    2. The name of Xilinx, Inc. may not be used to endorse or promote products redistributed with this software without specific
#    prior written permission.
#
#    THIS SOFTWARE IS PROVIDED BY XILINX, INC. "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
#    THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL XILINX, INC.
#    BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
#    PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
#    OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE
#

set(vaip_common_cmake_internal_dir ${CMAKE_CURRENT_LIST_DIR} CACHE INTERNAL "")

set(CMAKE_CXX_STANDARD 17)
if(NOT MSVC)
  set(CMAKE_CXX_FLAGS_DEBUG
      "${CMAKE_CXX_FLAGS_DEBUG} -ggdb -O0 -U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=0 -fno-inline"
  )
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")
  string(APPEND CMAKE_CXX_FLAGS
         " -std=gnu++17 -Wall -Werror -Wconversion -pedantic -Wextra")
  string(APPEND CMAKE_CXX_FLAGS " -Wno-unused-parameter")

  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Werror -Wconversion")
  set(CMAKE_SHARED_LINKER_FLAGS
      "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-undefined")

  if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    string(APPEND CMAKE_CXX_FLAGS " -Wno-sign-conversion")
    string(APPEND CMAKE_CXX_FLAGS " -Wno-nested-anon-types")
    string(APPEND CMAKE_CXX_FLAGS " -Wno-invalid-utf8")
    ## TODO: it is not good to disable the following warnings.
    string(APPEND CMAKE_CXX_FLAGS " -Wno-shorten-64-to-32")
    string(APPEND CMAKE_CXX_FLAGS " -Wno-implicit-int-conversion")
    string(APPEND CMAKE_CXX_FLAGS " -Wno-self-assign")
    string(APPEND CMAKE_CXX_FLAGS " -Wno-delete-non-abstract-non-virtual-dtor")
  endif()
  if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    string(APPEND CMAKE_CXX_FLAGS " -Wno-attributes")
  endif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")

  set(CMAKE_MACOSX_RPATH 1)
else(NOT MSVC)
  # `/Zc:__cplusplus`: This option ensures that the `__cplusplus`
  # macro reflects the correct version of the C++ standard used by the
  # compiler. By default, MSVC might not update this macro correctly
  # to reflect the C++ standard version. This option forces the
  # compiler to update the macro appropriately, which can be crucial
  # for conditional compilation depending on the C++ standard version.

  # `/Zi`: This option enables the generation of complete debugging
  # information. It allows for the creation of a PDB (Program
  # Database) file, which stores debugging and project state
  # information. The PDB file is used by debuggers to provide
  # source-level debugging, including setting breakpoints, stepping
  # through code, and inspecting variables.

  # `/Qspectre`: This option enables mitigations against the Spectre
  # vulnerability, a hardware vulnerability that affects modern
  # microprocessors that perform branch prediction. By enabling this
  # option, the compiler will generate code that is protected against
  # this class of vulnerabilities, at the potential cost of some
  # performance overhead.

  # `/ZH:SHA_256`: This option specifies the hash algorithm used for
  # generating content hashes in the PDB file. Setting it to `SHA_256`
  # uses the SHA-256 algorithm, which is more secure than the default
  # MD5, providing better protection against hash collision attacks.

  # `/guard:cf`: This option enables Control Flow Guard (CFG), a
  # security feature that checks that the target of a call or jump is
  # valid at runtime. This can help protect against attacks that
  # attempt to hijack the control flow of the program. It adds a
  # runtime check but can significantly increase the security of the
  # application.

  # `/sdl`: Stands for "Security Development Lifecycle". This option
  # enables additional security checks and makes warnings more
  # stringent. It's part of a broader approach to developing software
  # that reduces vulnerabilities and security issues.

  # Microsoft requested
  add_compile_options(
    /Zc:__cplusplus
    /Zi
    /Qspectre
    /ZH:SHA_256
    /guard:cf
    /sdl
    /MP
  )


  add_link_options(
   # `/DEBUG`: This option instructs the linker to generate debug
   # information for the compiled binaries. This debug information is
   # crucial for debugging the application, as it maps the binary code
   # back to the source code, allowing developers to step through the
   # code, set breakpoints, and inspect variables during a debugging
   # session. The generated debug information is typically stored in a
   # PDB (Program Database) file.

   # `/guard:cf`: This option enables Control Flow Guard (CFG) in the
   # linked binary. CFG is a security feature that helps protect
   # against attacks that attempt to hijack the control flow of the
   # program. It works by inserting runtime checks that validate the
   # target of indirect function calls, making it harder for an
   # attacker to execute arbitrary code through techniques like
   # return-oriented programming (ROP). Enabling CFG can significantly
   # enhance the security of the application by mitigating a class of
   # common exploits.

   # `/CETCOMPAT`: This option enables compatibility with Control-flow
   # Enforcement Technology (CET), a hardware-based security feature
   # designed to prevent certain types of attacks by enforcing
   # stricter control flow integrity. CET works by introducing new CPU
   # instructions that mark legitimate targets for indirect calls and
   # returns, effectively creating a shadow stack. This helps protect
   # against return-oriented programming (ROP) and call-oriented
   # programming (COP) attacks. Enabling `/CETCOMPAT` ensures that the
   # generated binary can take advantage of CET if it's supported by
   # the hardware, further enhancing the security of the application.
    /DEBUG
    /CETCOMPAT
  )
  if (CMAKE_SYSTEM_VERSION VERSION_GREATER_EQUAL "10.0.17763") # on later Wdinows version, there are bunch of errors in system header files
    add_compile_options(/wd5105)
  endif()
  # unreferenced formal parameter
  # non - DLL-interface class 'class_1' used as base for DLL-interface class 'class_2'
  add_compile_options(/W1 /wd4100 /wd4275 /wd4267 /wd4554 /WX)
  # some optimizations
  message(STATUS "CMAKE_BUILD_TYPE is ${CMAKE_BUILD_TYPE}")
  add_compile_options(
    /favor:AMD64
  )
  add_compile_options(
    # - `/Od`: This option disables optimization, instructing the
    # - compiler to generate code that is easy to debug. With
    # - optimizations turned off, the compiled code more closely
    # - mirrors the source code, making it easier to step through
    # - code, inspect variables, and understand program flow during
    # - debugging sessions. This setting is typically used for debug
    # - builds, where the priority is on debugging capabilities rather
    # - than execution speed or binary size.

    # - `/RTC1`: This option enables runtime error checks,
    # - specifically both `/RTCs` (Stack Frames) and `/RTCu`
    # - (Uninitialized Variables). It provides safeguards against
    # - common programming mistakes. `/RTCs` checks for stack frame
    # - runtime errors, such as stack corruption, and `/RTCu` checks
    # - for the use of uninitialized variables. Both of these checks
    # - can help catch errors that might be difficult to diagnose
    # - otherwise because they can lead to unpredictable
    # - behavior. Enabling these checks can significantly aid in
    # - debugging, especially during the development phase, by
    # - providing immediate feedback when certain types of errors
    # - occur.
    "$<$<CONFIG:DEBUG>:/Od;/RTC1>"
    )
    # - `/O2`: This option tells the compiler to optimize the code for
    # - maximum speed. It enables a mix of optimizations that aim to
    # - reduce the execution time of the compiled program without
    # - increasing its size too much. This is a common choice for
    # - release builds where performance is critical.

    # - `/Oi`: This option enables intrinsic functions. Intrinsic
    # - functions are special functions recognized by the compiler
    # - that can be replaced with optimized assembly code. This can
    # - lead to significant performance improvements since it allows
    # - the compiler to take advantage of specific processor features.

    # - `/Ot`: This option favors fast code by instructing the
    # - compiler to optimize for speed. It's similar to `/O2` but
    # - focuses more specifically on making function calls and loops
    # - run faster. This is achieved by, for example, keeping
    # - variables in registers instead of memory when possible.

    # - `/GL`: This option enables whole program optimization (also
    # - known as Link-Time Code Generation, LTCG). When this option is
    # - enabled, the compiler is able to perform optimizations across
    # - the boundaries of individual source files. This can lead to
    # - better optimization decisions since the compiler has a more
    # - complete view of the program's behavior. However, it can
    # - increase compile time and memory usage during compilation.

    # FIXME: which options we should use?
    # add_compile_options(
    #   /O2
    #   /Oi
    #   /Ot
    #   /GL
    # )
    add_link_options("$<$<CONFIG:RELEASE>:/OPT:REF;/OPT:ICF>")

endif(NOT MSVC)

function(vai_add_library)
  set(options)
  set(oneValueArgs NAME INCLUDE_DIR SRC_DIR TEST_DIR SKIP_INSTALL)
  set(multiValueArgs SRCS DEPENDS)
  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}"
                        ${ARGN})

  # start check include dir
  if(NOT ARG_INCLUDE_DIR)
    set(ARG_INCLUDE_DIR "include")
  endif(NOT ARG_INCLUDE_DIR)
  # end check include dir

  # start to check src dir
  if(NOT ARG_SRC_DIR)
    set(ARG_SRC_DIR "src")
  endif(NOT ARG_SRC_DIR)
  # end check src dir

  # start to check test dir
  if(NOT ARG_TEST_DIR)
    set(ARG_TEST_DIR "test")
  endif(NOT ARG_TEST_DIR)
  # end check test dir

  # start check target name
  if(NOT ARG_NAME)
    get_filename_component(ARG_NAME "${CMAKE_CURRENT_SOURCE_DIR}" NAME)
    set(COMPONENT_NAME
        ${ARG_NAME}
        PARENT_SCOPE)
  endif(NOT ARG_NAME)
  # end check target name

  # create the target
  message(STATUS "create target ${ARG_NAME} SHARED ${ARG_SRCS}")
  add_library(${ARG_NAME} ${ARG_SRCS})
  # create alias
  add_library(${PROJECT_NAME}::${ARG_NAME} ALIAS ${ARG_NAME})

  target_link_libraries(${ARG_NAME} PUBLIC ${ARG_DEPENDS})
  # target_link_libraries(${ARG_NAME} PUBLIC -ltvm)
  target_include_directories(
    ${ARG_NAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                       $<INSTALL_INTERFACE:${INSTALL_INCLUDEDIR}>)
  target_include_directories(
    ${ARG_NAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

  # set all properties
  set_target_properties(${ARG_NAME} PROPERTIES OUTPUT_NAME
                                               ${PROJECT_NAME}-${ARG_NAME})
  target_compile_definitions(
    ${ARG_NAME} PRIVATE -DOUTPUT_NAME="${PROJECT_NAME}-${ARG_NAME}")
  file(APPEND ${CMAKE_BINARY_DIR}/components.txt "${ARG_NAME}\n")

  set_target_properties(${ARG_NAME} PROPERTIES FOLDER "VAIP")

endfunction(vai_add_library)

function(vai_add_test name)
  set(options "")
  set(oneValueArgs ENABLE_IF)
  set(multiValueArgs REQUIRE SOURCES)
  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}"
                        ${ARGN})

  set(_enable TRUE)
  if(ARG_ENABLE_IF)
    set(_enable ${${ARG_ENABLE_IF}})
  endif(ARG_ENABLE_IF)
  if(_enable)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test/${name}.cpp)
      add_executable(${name} test/${name}.cpp ${ARG_SOURCES})
    elseif(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test/${name}.c)
      add_executable(${name} test/${name}.c)
    else()
      message(
        FATAL_ERROR "cannot find either test/${name}.c or test/${name}.cpp")
    endif()
    # target_link_libraries(${name} ${PROJECT_NAME}::${COMPONENT_NAME})
    target_link_libraries(${name})
    if(ARG_REQUIRE)
      target_link_libraries(${name} ${ARG_REQUIRE})
    endif(ARG_REQUIRE)
    install(TARGETS ${name} DESTINATION bin/${PROJECT_NAME}/)
  endif(_enable)
endfunction(vai_add_test)

if(CMAKE_CROSSCOMPILING)
  set(_IS_CROSSCOMPILING ON)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64.*|AARCH64.*|arm64.*|ARM64.*)")
  set(_IS_CROSSCOMPILING ON)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm.*|ARM.*)")
  set(_IS_CROSSCOMPILING ON)
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
  set(_IS_CROSSCOMPILING OFF)
else()
  set(_IS_CROSSCOMPILING OFF)
endif(CMAKE_CROSSCOMPILING)

if(_IS_CROSSCOMPILING)
  set(EANBLE_COMPILER_DEFAULE_VALUE OFF)
else()
  set(EANBLE_COMPILER_DEFAULE_VALUE ON)
endif(_IS_CROSSCOMPILING)

function(vai_add_debug_command target_name)
    set(options "")
    set(oneValueArgs COMMAND WORKING_DIRECTORY)
    set(multiValueArgs ENVIRONMENT ARGUMENTS)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}"
                ${ARGN})

    ## for development debugguing purpose
    #file(TO_NATIVE_PATH ${vaip_common_cmake_internal_dir}/../vaip/etc/vaip_config.json VAIP_CONFIG_JSON_PATH)
    set(tmp_paths "${CMAKE_INSTALL_PREFIX}/bin" "${CMAKE_INSTALL_PREFIX}/xrt")
    if(PYTHON_EXECUTABLE)
        get_filename_component(tmp_python_path ${PYTHON_EXECUTABLE} DIRECTORY)
        file(TO_NATIVE_PATH ${tmp_python_path} tmp_python_path)
        list(APPEND tmp_paths ${tmp_python_path})
    endif(PYTHON_EXECUTABLE)
    cmake_path(CONVERT "${tmp_paths}" TO_NATIVE_PATH_LIST DEBUG_PATH)
    # valid variable in expanding debug_env.txt.in
    #     DEBUG_PATH
    #     VAIP_CONFIG_JSON_PATH
    set(CURRENT_DEBUG_ENV "")
    if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/debug_env.txt.in")
        file(READ "${CMAKE_CURRENT_SOURCE_DIR}/debug_env.txt.in" tmp_debug_env_in)
        string(CONFIGURE "${tmp_debug_env_in}" CURRENT_DEBUG_ENV @ONLY)
    endif()
    if(EXISTS "${vaip_common_cmake_internal_dir}/../debug_env.txt.in")
       file(READ "${vaip_common_cmake_internal_dir}/../debug_env.txt.in" tmp_debug_env_in)
       string(CONFIGURE "${tmp_debug_env_in}" DEBUG_ENV @ONLY)
    endif()
    string(APPEND DEBUG_ENV "\n${CURRENT_DEBUG_ENV}")
    string(APPEND DEBUG_ENV "\n${CURRENT_DEBUG_ENV}")
    string(REPLACE ";" "\n" ARG_ENVIRONMENT "${ARG_ENVIRONMENT}")
    string(APPEND DEBUG_ENV "\n${ARG_ENVIRONMENT}")
    if (NOT ARG_COMMAND)
        set(ARG_COMMAND "$(TargetPath)")
    endif(NOT ARG_COMMAND)
    if (NOT ARG_WORKING_DIRECTORY)
        set(ARG_WORKING_DIRECTORY "$(ProjectDir)")
    endif(NOT ARG_WORKING_DIRECTORY)

    set_target_properties(${target_name} PROPERTIES
      VS_DEBUGGER_COMMAND "${ARG_COMMAND}"
      VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
      VS_DEBUGGER_COMMAND_ARGUMENTS "${ARG_ARGUMENTS}"
      VS_DEBUGGER_ENVIRONMENT "${DEBUG_ENV}"
  )
endfunction(vai_add_debug_command)
