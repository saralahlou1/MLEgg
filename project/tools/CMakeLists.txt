get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

set (LIBS ${dialect_libs} MLIROptLib)

add_executable(es-opt es-opt.cpp)
target_link_libraries(es-opt Support ES ${LIBS})
