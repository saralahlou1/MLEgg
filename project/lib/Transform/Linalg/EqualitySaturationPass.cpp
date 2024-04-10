// this is a pass
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/Value.h>
#include <vector>

#include "Transform/Linalg/EqualitySaturationPass.h"
#include "Support/Graph.h"

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
        // TODO: make this a template?
        Graph graph;
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
            // and if it does, then the name is meaningless past as an id
            // value names are unique by definition

            // ok so all of that above was irrelevant.

            // now that we have every linalg op, we're going to go through and follow all the relevant values
            // we're going to be redundant and verbose because i don't care
            mlir::OperationName name = internalOperation->getName();
            // get the result
            mlir::Value result = internalOperation->getResult(0);
            // get only the ins
            mlir::Value arg1 = internalOperation->getOperand(0);
            mlir::Value arg2 = internalOperation->getOperand(1);

            // // do we follow the values?
            // no, because they're unique (because of SSA)
            // how do i order these?
            auto& node = graph.add_node(name.getIdentifier().str(), result);
            Graph::Node& child_1 = graph.add_node("", arg1);
            Graph::Node& child_2 = graph.add_node("", arg2);
            node.children.push_back(child_1);
            node.children.push_back(child_2);
        }

        graph.to_file("out.gv");

        // pass it to rust program
        // TODO: FFI binding?
        system("eqsat out.gv");

        // get the result from the rust program
        // read the file
        Graph returned = graph.from_file("out2.gv");

        // reassociate the nodes in the result with the nodes in the block
        // these are numbered the same as the original graph; we can use them together to rebuild

        // make the necessary changes

}
}

// actually register the pass
namespace mlir {
void registerEqualitySaturationPass() {
    PassRegistration<EqualitySaturationPass>();
}
} // namespace mlir
