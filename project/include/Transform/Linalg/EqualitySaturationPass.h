#ifndef EQUALITY_SATURATION_PASS_H
#define EQUALITY_SATURATION_PASS_H

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>
#include "mlir/IR/Operation.h"



namespace mlir {

struct EqualitySaturationPass : public mlir::PassWrapper<EqualitySaturationPass, OperationPass<ModuleOp>> {
    // info about the pass
    // the flag used to run the pass
    mlir::StringRef getArgument() const final { return "equality-saturation"; }
    // description shown in help text
    mlir::StringRef getDescription() const final { return "Runs equality saturation"; }
    // main method equiv
    void runOnOperation() override;
};

void registerEqualitySaturationPass();
} // namespace mlir
bool is_number(std::string);

#endif
