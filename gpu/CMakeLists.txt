cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(santa LANGUAGES CXX CUDA)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-Ofast")

# Executables
add_executable(k_opt src/k_opt.cu)

# Tests
find_package(CxxTest)

if(CXXTEST_FOUND)
    include_directories(${CXXTEST_INCLUDE_DIR})
    enable_testing()
    CXXTEST_ADD_TEST(
        test_problem
        test_problem.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/test/test_problem.hpp
    )
    CXXTEST_ADD_TEST(
        test_io
        test_io.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/test/test_io.hpp
    )
endif()

# https://devblogs.nvidia.com/building-cuda-applications-cmake/
    
if(APPLE)
    # We need to add the path to the driver (libcuda.dylib) as an rpath, 
    # so that the static cuda runtime can find it at runtime.
    set_property(TARGET main
                PROPERTY
                BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()
