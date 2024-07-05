project(es-opt LANGUAGES CXX C)
add_library(Support Graph.cpp)
include_directories (${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

