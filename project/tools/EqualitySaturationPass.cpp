// this is a pass
#include <mlir/Dialect/Linalg/IR/LinalgInterfaces.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <vector>

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
    void runOnOperation() {
        // assume structure.
        // simplify, simplify.
        mlir::ModuleOp moduleOp = getOperation();

        // we operate on blocks. get each block from the region (same as BodyRegion)
        for (mlir::Block &block : moduleOp.getRegion().getBlocks()) {
            // there's also only one block, but not the point
            // in each block. find the linalg operations
            // we care about the named operations and maybe generic
            // (fill, dot, matmul, conv)
            // start with matmul

            // get a list of all the ops we care about in this block
            // would be cool to have a custom iterator for the ops we care about
            // TODO: make sure traversal works better
            std::vector<std::reference_wrapper<int>> filtered_ops;
            for (mlir::Operation &op : block.getOperations()) {
                if (llvm::isa<mlir::linalg::MatmulOp, mlir::linalg::DotOp>(op)) {
                    // add to list

                }
            }

            // build graph

            // pass it to rust program
            // TODO: FFI binding?

            // get the result from the rust program

            // reassociate the nodes in the result with the nodes in the block

            // make the necessary changes

        }
    }
};
} // namespace

// actually register the pass
namespace mlir {
void registerMyPass() {
    PassRegistration<EqualitySaturationPass>();
}
} // namespace mlir
