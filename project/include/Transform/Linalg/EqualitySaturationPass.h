#ifndef EQUALITY_SATURATION_PASS_H
#define EQUALITY_SATURATION_PASS_H

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>

// oh god do i want a namespace here
namespace {
// can specify what operations we operate on. we only operate on moduleops
struct EqualitySaturationPass : public mlir::PassWrapper<EqualitySaturationPass, mlir::OperationPass<mlir::ModuleOp>> {
    // info about the pass
    // the flag used to run the pass
    mlir::StringRef getArgument() const final { return "equality-saturation"; }
    // description shown in help text
    mlir::StringRef getDescription() const final { return "Runs equality saturation"; }
    // main method equiv
    void runOnOperation() override;
};
} //namespace

namespace mlir {
void registerEqualitySaturationPass();
} // namespace mlir

#endif
