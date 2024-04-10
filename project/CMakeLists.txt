cmake_minimum_required(VERSION 3.28)

project(es-opt LANGUAGES CXX C)

find_package(MLIR REQUIRED CONFIG)

add_subdirectory(tools)
