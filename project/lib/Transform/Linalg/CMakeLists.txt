project(es-opt LANGUAGES CXX C)
get_property(libs GLOBAL PROPERTY MLIR_LIBS)

add_library(ES 
	EqualitySaturationPass.cpp
)

#get_property(MLIR_LIBS GLOBAL PROPERTY MLIR_ALL_LIBS)


set (LIBS
#    ${libs}
    ${dialect_libs}
    ${conversion_libs}
    MLIROptLib
#    MLIRPass
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
)


target_link_libraries(ES Support ${LIBS})
include_directories (${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
# target_link_libraries(ES Support)

