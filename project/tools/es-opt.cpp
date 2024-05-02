#include <mlir/InitAllDialects.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Pass/PassRegistry.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>

#include "Transform/Linalg/EqualitySaturationPass.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);

  mlir::registerEqualitySaturationPass();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Equality saturated MLIR", registry));
}
