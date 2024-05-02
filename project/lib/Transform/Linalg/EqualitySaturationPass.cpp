// this is a pass
#include <algorithm>
#include <iostream>
#include <llvm/ADT/APInt.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/OperationSupport.h>
#include <llvm/ADT/Hashing.h>
#include <mlir/IR/Value.h>
#include <stdexcept>
#include <vector>
#include <map>

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
        std::map<llvm::hash_code, int> value_to_id;
        std::map<int, mlir::Value&> id_to_value;
        int id_counter = 0;
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

            // get the node ids these map to
            int result_id = value_to_id.try_emplace(mlir::hash_value(result), id_counter++).first->second;
            std::cout << id_counter << " " << result_id << '\n';
            int arg1_id = value_to_id.try_emplace(mlir::hash_value(arg1), id_counter++).first->second;
            std::cout << id_counter << " " << arg1_id << '\n';
            int arg2_id = value_to_id.try_emplace(mlir::hash_value(arg2), id_counter++).first->second;
            std::cout << id_counter << " " << arg2_id << '\n';

            id_to_value.try_emplace(result_id, result);
            id_to_value.try_emplace(arg1_id, arg1);
            id_to_value.try_emplace(arg2_id, arg2);

            // // do we follow the values?
            // no, because they're unique (because of SSA)
            // how do i order these?
            Graph::Node& node = graph.add_node(result_id, name.getIdentifier().str());
            graph.add_node(arg1_id, "");
            graph.add_node(arg2_id, "");
            node.children.push_back(arg1_id);
            node.children.push_back(arg2_id);
        }

        graph.to_file("out.gv");

        // pass it to rust program
        // TODO: FFI binding?
        if (!system("eq-sub out.gv out2.gv")) {
            // the Rust program failed!
        }

        // get the result from the rust program
        // read the file
        Graph returned = graph.from_file("out2.gv");

        // reassociate the nodes in the result with the nodes in the block
        // these are numbered the same as the original graph; we can use them together to rebuild
        // what do i do assuming i have a functional graph?
        // this is janky, but we can't just use the standard rewrite application -- which assumes

        // insert at end of block. this isn't necessarily good -- it would be better to have an alg
        // figure out where the best place to insert would be
        mlir::OpBuilder builder(&(block.back()));
        for (auto node : returned.get_nodes()) {
            // c++ doesn't have good switches.
            // if the data is an operation, match on the operation and create a new one to insert
            if (node.second.data == "linalg.matmul") {
                auto& result_val = id_to_value.find(node.first)->second;
                auto& arg1_val = id_to_value.find(node.second.children[0])->second;
                auto& arg2_val = id_to_value.find(node.second.children[1])->second;
                builder.create<mlir::linalg::MatmulOp>(builder.getUnknownLoc(), std::vector<mlir::Value>{arg1_val, arg2_val}, std::vector<mlir::Value>{result_val});    
            } else {
                // data is just a link to a previous op. ignore
            }
        }

        // delete the old operations (everything in the list)
        for (auto op : filtered_ops) {
            op.erase();
        }
}
}

// actually register the pass
namespace mlir {
void registerEqualitySaturationPass() {
    PassRegistration<EqualitySaturationPass>();
}
} // namespace mlir
