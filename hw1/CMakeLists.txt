cmake_minimum_required(VERSION 3.6)
project(hw1)

set(CMAKE_CXX_STANDARD 14)

# FIND OPENMP
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

set(SOURCE_FILES src/hw1.cpp src/main.cpp)
add_executable(hw1 ${SOURCE_FILES})
