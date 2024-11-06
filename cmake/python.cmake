function(get_host_python)
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
    set(HOST_PYTHON_EXE ${Python3_EXECUTABLE} PARENT_SCOPE)
endfunction()


if(USE_STATIC_PYTHON)
    # create a new scope, do not contaminate the variables
    get_host_python()

    find_package(static_python REQUIRED COMPONENTS static_python header)
    find_package(unilog REQUIRED util)
else()
    find_package(unilog REQUIRED util)
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
    if(BUILD_PYTHON_EXT)
        find_package(pybind11 REQUIRED)
    endif()
    set(HOST_PYTHON_EXE ${Python3_EXECUTABLE})
endif()