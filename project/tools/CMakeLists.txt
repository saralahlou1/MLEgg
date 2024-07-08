project(es-opt LANGUAGES CXX C)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(libs GLOBAL PROPERTY MLIR_LIBS)

get_property(MLIR_LIBS GLOBAL PROPERTY MLIR_ALL_LIBS)


set (LIBS ${dialect_libs} MLIROptLib)

set (LIBS
    ${libs}
    ${dialect_libs}
    ${conversion_libs}
    MLIROptLib
    MLIRPass
    LLVMSupport
    Support        
    MLIRAnalysis
    MLIRAsmParser
    MLIRDialect
    MLIRExecutionEngine
    MLIRIR
    MLIRParser
    MLIRRewrite
    MLIRTableGen
    MLIRTransforms
ES
)

include_directories (${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

add_executable(es-opt es-opt.cpp)
target_link_libraries(es-opt PRIVATE ES MLIRTransforms MLIRAnalysis MLIRSupport ${LIBS} ${MLIR_LIBS})
target_include_directories(es-opt PRIVATE ${MLIR_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS})
