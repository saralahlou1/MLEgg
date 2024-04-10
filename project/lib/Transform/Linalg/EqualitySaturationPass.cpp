// this is a pass
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <vector>

#include "Transform/Linalg/EqualitySaturationPass.h"

void EqualitySaturationPass::runOnOperation() {
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
        // would be cool to have a custom iterator for the ops we care about instead
        // TODO: make sure traversal works better
        // also todo: this is unnecessarily specific
        std::vector<mlir::linalg::LinalgOp> filtered_ops;
        for (mlir::Operation &op : block.getOperations()) {
            if (llvm::isa<mlir::linalg::MatmulOp, mlir::linalg::DotOp>(op)) {
                // add to list
                filtered_ops.push_back(llvm::dyn_cast<mlir::linalg::LinalgOp>(op)); // will always work; also an implementation detail
            }
        }

        // build graph
        // for each operation, its node name is its identifier
        // its arguments are its ins
        // we don't have to consider its outs (for linalg, it's inits only)
        // we don't have to worry about dimensions either (presumably they're right)
        for (mlir::linalg::LinalgOp op : filtered_ops) {
            // we love dot.
            // i wrote a helper class for dot.
            // each function that we care about is a node with two children
            // the two children are the operands
            mlir::Operation *internalOperation = op.getOperation();
            // the internalOperation.name() is the "linalg.x"
            // the internalOperation.getOperands() is the "ins" and "outs"
            // we only want the ins
            // a value might or might not have an operation associated with it
            // if it doesn't, then it's an egg variable
            // and if it does, then the name is meaningless past as an id
            // value names are unique by definition

        }


        // pass it to rust program
        // TODO: FFI binding?

        // get the result from the rust program

        // reassociate the nodes in the result with the nodes in the block

        // make the necessary changes

}
}

// actually register the pass
namespace mlir {
void registerEqualitySaturationPass() {
    PassRegistration<EqualitySaturationPass>();
}
} // namespace mlir
