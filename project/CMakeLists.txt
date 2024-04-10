cmake_minimum_required(VERSION 3.28)

project(es-opt LANGUAGES CXX C)

find_package(MLIR REQUIRED CONFIG)

include_directories(include)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # for clangd

add_subdirectory(lib)
add_subdirectory(tools)
