cmake_minimum_required(VERSION 3.28)

project(es-opt LANGUAGES CXX C)

find_package(MLIR REQUIRED CONFIG)

include_directories(include)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # for clangd
set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(BUILD_DEPS ON)

set(NO_RTTI "-fno-rtti")
add_definitions(${NO_RTTI})

add_subdirectory(lib)
add_subdirectory(tools)

find_package(LLVM REQUIRED CONFIG)
include_directories(include)
