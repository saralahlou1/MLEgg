// this is from the mlir-project-tests
// this is a pass
#include <algorithm>
#include <iostream>
#include <llvm/ADT/APInt.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Operation.h"
#include <mlir/IR/Builders.h>
#include <mlir/IR/OperationSupport.h>
#include <llvm/ADT/Hashing.h>
#include <mlir/IR/Value.h>
#include <stdexcept>
#include <vector>
#include <map>
#include "mlir/IR/PatternMatch.h"

#include <mlir/Pass/PassManager.h>
#include <mlir/Pass/PassRegistry.h>

#include "Transform/Linalg/EqualitySaturationPass.h"
#include "Support/Graph.h"

void mlir::EqualitySaturationPass::runOnOperation() {
    // assume structure.
    // simplify, simplify.
    mlir::ModuleOp moduleOp = getOperation();

    // we operate on blocks. get each block from the region (same as BodyRegion)
    for (mlir::Block &block : moduleOp.getRegion().getBlocks()) {
        // there's also only one block, but not the point
        // in each block. find the linalg operations
        // we care about the named operations and maybe generic
        // (dot, matmul, add, transpose)

        // get a list of all the ops we care about in this block
        std::vector<mlir::linalg::LinalgOp> filtered_ops;
        std::vector<mlir::Operation*> all_ops;
        for (mlir::Operation &op : block.getOperations()) {
            if (llvm::isa<mlir::linalg::MatmulOp, mlir::linalg::DotOp, mlir::linalg::TransposeOp, mlir::linalg::AddOp>(op)) {
                // add to list
                // will always work since we only consider linalg for now; also an implementation detail
                filtered_ops.push_back(llvm::dyn_cast<mlir::linalg::LinalgOp>(op)); 
                all_ops.push_back(&op);
                // this is just to print the dims of the operands
                for (Value operand : op.getOperands()) {
                    // Get the type of the operand
                    Type operandType = operand.getType();
                    // Check if it's a tensor type
                    if (auto tensorType = operandType.dyn_cast<RankedTensorType>()) {
                        // Get the shape of the tensor type
                        ArrayRef<int64_t> shape = tensorType.getShape();
                        // Do something with the shape (dimensions)
                        // For example, print the dimensions
                        llvm::errs() << "Dimensions of operand: ";
                        for (int64_t dim : shape) {
                        llvm::errs() << dim << " ";
                        }
                        llvm::errs() << "\n";
                    }
                }

            }
            else if (llvm::isa<mlir::tensor::ExtractSliceOp>(op) || llvm::isa<mlir::linalg::BroadcastOp>(op)){
                all_ops.push_back(&op);
                // this is just to print the dims of the operands
                for (Value operand : op.getOperands()) {
                    // Get the type of the operand
                    Type operandType = operand.getType();
                    // Check if it's a tensor type
                    if (auto tensorType = operandType.dyn_cast<RankedTensorType>()) {
                        // Get the shape of the tensor type
                        ArrayRef<int64_t> shape = tensorType.getShape();
                        // Do something with the shape (dimensions)
                        // For example, print the dimensions
                        llvm::errs() << "Dimensions of operand of extract slice op: ";
                        for (int64_t dim : shape) {
                        llvm::errs() << dim << " ";
                        }
                        llvm::errs() << "\n";
                    }
                }
                std::cout << "Number of operands in extract slice op: " << op.getNumOperands() << "\n";

                for (Value result : op.getResults()) {
                    // Get the type of the operand
                    Type operandType = result.getType();
                    // Check if it's a tensor type
                    if (auto tensorType = operandType.dyn_cast<RankedTensorType>()) {
                        // Get the shape of the tensor type
                        ArrayRef<int64_t> shape = tensorType.getShape();
                        // Do something with the shape (dimensions)
                        // For example, print the dimensions
                        llvm::errs() << "Dimensions of result of extract slice op: ";
                        for (int64_t dim : shape) {
                        llvm::errs() << dim << " ";
                        }
                        llvm::errs() << "\n";
                    }
                }
                
                std::cout << "Number of results in extract slice op: " << op.getNumResults() << "\n";
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
        std::map<int, mlir::Value> id_to_value;
        int id_counter = 0;
        for (auto op : all_ops) {
            // we love dot.
            // i wrote a helper class for dot.
            // each function that we care about is a node with two children
            // the two children are the operands
            mlir::Operation *internalOperation = op;
            // the internalOperation.name() is the "linalg.x"
            // the internalOperation.getOperands() is the "ins" and "outs"
            // we only want the ins
            // a value might or might not have an operation associated with it
            // and if it does, then the name is meaningless past as an id
            // value names are unique by definition

            // check the number of operands. This gives nb of operands plus the one for the result
            unsigned numOperands = internalOperation->getNumOperands();
            std::cout << "Number of operands: " << numOperands << "\n";

            // ok so all of that above was irrelevant.

            // now that we have every linalg op, we're going to go through and follow all the relevant values
            // we're going to be redundant and verbose because i don't care
            mlir::OperationName name = internalOperation->getName();
            // get the result
            mlir::Value result = internalOperation->getResult(0);
            // we start with the first operand
            // get only the ins
            mlir::Value arg1 = internalOperation->getOperand(0);
            
            // get the node ids these map to
            int result_id = value_to_id.try_emplace(mlir::hash_value(result), id_counter++).first->second;
            std::cout << id_counter << " " << result_id << '\n';
            int arg1_id = value_to_id.try_emplace(mlir::hash_value(arg1), id_counter++).first->second;
            std::cout << id_counter << " " << arg1_id << '\n';
            id_to_value.try_emplace(result_id, result);
            id_to_value.try_emplace(arg1_id, arg1);


            // we prepare the default dimensions of the arguments and results
            int rows = 1;
            int columns = 1;
            // Get the type of arg1
            Type result_type = result.getType();
            // Check if it's a tensor type
            if (auto tensorType = result_type.dyn_cast<RankedTensorType>()) {
                // Get the shape of the tensor type
                ArrayRef<int64_t> shape = tensorType.getShape();
                // Do something with the shape (dimensions)
                // For example, print the dimensions
                llvm::errs() << "Dimensions of arg1: ";
                for (int64_t dim : shape) {
                    llvm::errs() << dim << " ";
                }
                llvm::errs() << "\n";
                if (shape.size() > 0){
                    rows = static_cast<int>(shape[0]);
                }
                if (shape.size() > 1){
                columns = static_cast<int>(shape[1]);
                }
                llvm::errs() << "imputed rows: " << rows << " and columns: " << columns << "\n";

            }

            // // do we follow the values?
            // no, because they're unique (because of SSA)
            // how do i order these?
            Graph::Node& node = graph.add_node_with_dims(result_id, name.getIdentifier().str(), rows, columns, 0, 0);


            // we reinitialize the default values
            rows = 1;
            columns = 1;
            // Get the type of arg1
            Type arg1_type = arg1.getType();
            // Check if it's a tensor type
            if (auto tensorType = arg1_type.dyn_cast<RankedTensorType>()) {
                // Get the shape of the tensor type
                ArrayRef<int64_t> shape = tensorType.getShape();
                // Do something with the shape (dimensions)
                // For example, print the dimensions
                llvm::errs() << "Dimensions of arg1: ";
                for (int64_t dim : shape) {
                    llvm::errs() << dim << " ";
                }
                llvm::errs() << "\n";
                if (shape.size() > 0){
                    rows = static_cast<int>(shape[0]);
                }
                if (shape.size() > 1){
                columns = static_cast<int>(shape[1]);
                }
                llvm::errs() << "imputed rows: " << rows << " and columns: " << columns << "\n";

            }

            graph.add_node_with_dims(arg1_id, std::to_string(rows) + "x" + std::to_string(columns) + " matrix", rows, columns, 0, 0);
            node.children.push_back(arg1_id);


            // we reinitialize the default values
            rows = 1;
            columns = 1;
            // we check if there is a second operand
            if (numOperands > 2) {
                // if so we repeate the same procedure
                mlir::Value arg2 = internalOperation->getOperand(1);
                int arg2_id = value_to_id.try_emplace(mlir::hash_value(arg2), id_counter++).first->second;
                std::cout << id_counter << " " << arg2_id << '\n';
                id_to_value.try_emplace(arg2_id, arg2);

                // Get the type of arg2
                Type arg2_type = arg2.getType();
                // Check if it's a tensor type
                if (auto tensorType = arg2_type.dyn_cast<RankedTensorType>()) {
                    // Get the shape of the tensor type
                    ArrayRef<int64_t> shape = tensorType.getShape();
                    // Do something with the shape (dimensions)
                    // For example, print the dimensions
                    llvm::errs() << "Dimensions of arg2: ";
                    for (int64_t dim : shape) {
                        llvm::errs() << dim << " ";
                    }
                    llvm::errs() << "\n";
                    if (shape.size() > 0){
                        rows = static_cast<int>(shape[0]);
                    }
                    if (shape.size() > 1){
                    columns = static_cast<int>(shape[1]);
                    }
                    llvm::errs() << "imputed rows: " << rows << " and columns: " << columns << "\n";

                }
                graph.add_node_with_dims(arg2_id, std::to_string(rows) + "x" + std::to_string(columns) + " matrix", rows, columns, 0, 0);

                
                node.children.push_back(arg2_id);
            }

        }

        // we will keep track of the variable representing the largest key in the graph
        // this is to later check if the old_id refers to an op that already existed or a new one
        // this case doesn't happen for now but it's better to make the code robust
        // and handle this in case it becomes an issue one day
        int largest_id = 0;
        for (auto node : graph.get_nodes()) {
            if (node.first > largest_id){
                largest_id = node.first;
            }
        }

        std::cout << "largest id in old graph: " << largest_id << " \n";


        graph.to_file("out.gv");

        // pass it to rust program
        if (!system("eq-sub out.gv out2.gv")) {
            // the Rust program failed!
        }

        // get the result from the rust program
        // read the file
        Graph returned = graph.from_file("out2.gv");

        std::cout << "constructed the graph! \n";
        
        // insert at end of block. this isn't necessarily good -- it would be better to have an alg
        // figure out where the best place to insert would be
        mlir::OpBuilder builder(&(block.back()));


        for (const auto& pair : id_to_value) {
            auto shape = pair.second.getType().cast<ShapedType>().getShape();

            std::cout << "id " << pair.first << " with type " ;
            for (int64_t dim : shape) {
                std::cout << dim << " ";
            }
            std::cout << "\n";
        
        }


        // to make sure the ops are always considered in the correct order
        // we should do breath first traversal to order the nodes and then traverse them 
        // in reverse order
        // this way we avoid refering to an op not yet created
        // this is not done yet (I think it is already done automatically)

        auto nodes = returned.get_nodes();
        for (auto node = nodes.rbegin(); node != nodes.rend(); ++node) {
            // if the data is an operation, match on the operation and create a new one to insert
            int old_id = node->second.old_id;
            int old_op_id = node->second.old_op_id;
            
            // this should be our first case
            if (old_id != old_op_id){
                // this could refer to an op that was the result of 
                // transpose (traspose (X)) = X
                // before an optimisations, we have oldId = oldOpId
                // after the optimisations, when we replace op A with B,
                // B retains its oldId (its mlir value stays the same)
                // but its old op id become the old op id of A 
                // (it's like we refer to the operation to replace)
                // following this train of thoughts, transpose (traspose (X)) = X
                // X retains its old Id, but its old op id now refers to the one of the outer transpose
                // thus, we check the if oldId != oldOpId to see if we need any rewrites
                // if that's the case, we replace the oldOpId uses with oldId uses and erase that op
                

                std::cout << "in loop op where old op id and old id are different old id: " << old_id << " old op id: "<< old_op_id << " and data: " << node->second.data << " \n";
                
                // we search for the operations that old id refers to
                // we need to look into all the operations in the block
                // not just the filterd ones
                bool found = false;
                mlir::Operation* matrixOp;
                for (mlir::Operation &op : block.getOperations()) {
                    mlir::Value result;
                    if (op.getNumResults() != 0){
                        result = op.getResult(0);
                            for (const auto& pair : id_to_value) {
                            if (pair.second == result) {
                                int id = pair.first;
                                if (id == old_id){
                                    std::cout << "Found the tensor! \n";
                                    found = true;
                                    break;
                                }
                            }
                        }
                        if (found){
                            matrixOp = &op;
                            break;
                        }
                    } else {
                        continue;
                    }
                    
                }

                // we also store the values corresponding to both old id and old op id
                auto& result_val = id_to_value.find(old_id)->second;
                auto& old_op_val = id_to_value.find(old_op_id)->second;
                

                // we only make any changes if the old operation existed before
                // not very sure about this part but it doesn't cause issues with our present rules
                if (old_id <= largest_id) {
                    // we now search for the operation to replace (corresponds to old op id)
                    // we only need to search in the filtered operations this time
                    found = false;
                    for (mlir::linalg::LinalgOp op : filtered_ops) {
                        mlir::Operation *internalOperation = op.getOperation();
                        mlir::Value result;
                        if (internalOperation->getNumResults() != 0){
                            result = internalOperation->getResult(0);
                                for (const auto& pair : id_to_value) {
                                if (pair.second == result) {
                                    int id = pair.first;
                                    if (id == old_op_id){
                                        found = true;
                                        break;
                                    }
                                }
                            }
                            if (found){
                                std::cout << "Replacing the op! \n";
                                // we replace the uses of the old op with the new one and then erase it
                                op->replaceAllUsesWith(matrixOp);
                                op->replaceUsesOfWith(old_op_val, result_val);
                                op.erase();
                                break;
                            }
                        } else {
                            continue;
                        }
                    }
                }

                // sometimes even if old id != old op id, there can be no operation to replace
                // for exemple, if we introduce the transpose ourselves while doing the opts 
                // then there won't be anything we need to replace here:
                // dot((transpose A) B) => matmul((transpose (transpose A)) B) => matmul A B
                // here even if A has old id != old op id, we don't need to perform any replacements
                // since we were the ones to introduce the op
                if (found){
                    // in this case, we already found and replaced an operation
                    // so we move on to the next iteration
                    continue;
                }
                // else we go to the other cases to perform any necessary rewrite
                std::cout << "Checking other cases... \n";
            }
            
            if (node->second.data == "linalg.matmul") {
                std::cout << "writing matmul op... \n";
                // int old_id = node->second.old_id;
                int old_child1_id = returned.get_nodes().find(node->second.children[0])->second.old_id;
                int old_child2_id = returned.get_nodes().find(node->second.children[1])->second.old_id;
                
                auto& arg1_val = id_to_value.find(old_child1_id)->second;
                auto& arg2_val = id_to_value.find(old_child2_id)->second;

                // check if dims are nxf32. in this case cast it to nx1xf32. 
                // start with arg1
                if (arg1_val.getType().cast<ShapedType>().getShape().size() < 2){
                    
                    Type elementType = arg1_val.getType().cast<ShapedType>().getElementType();
                    SmallVector<int64_t, 2> newShape = {arg1_val.getType().cast<ShapedType>().getShape()[0], 1};
                    auto newType = RankedTensorType::get(newShape, elementType);

                    SmallVector<ReassociationIndices, 1> reassociation = {ReassociationIndices{0, 1}};

                    auto expandedTensor = builder.create<tensor::ExpandShapeOp>(
                        builder.getUnknownLoc(), newType, arg2_val, reassociation);
                    
                    arg2_val = expandedTensor;
                }

                // same for arg2
                if (arg2_val.getType().cast<ShapedType>().getShape().size() < 2){
                    
                    Type elementType = arg2_val.getType().cast<ShapedType>().getElementType();
                    SmallVector<int64_t, 2> newShape = {arg2_val.getType().cast<ShapedType>().getShape()[0], 1};
                    auto newType = RankedTensorType::get(newShape, elementType);

                    SmallVector<ReassociationIndices, 1> reassociation = {ReassociationIndices{0, 1}};

                    auto expandedTensor = builder.create<tensor::ExpandShapeOp>(
                        builder.getUnknownLoc(), newType, arg2_val, reassociation);
                    
                    arg2_val = expandedTensor;
                }
                
                // we prepare the tensor for the result with the correct dims
                auto lhsType = arg1_val.getType().cast<ShapedType>();
                auto rhsType = arg2_val.getType().cast<ShapedType>();
                auto lhsShape = lhsType.getShape();
                auto rhsShape = rhsType.getShape();
                Type elementType = lhsType.getElementType();
                
                SmallVector<int64_t, 2> resultShape = {lhsShape[0], rhsShape[1]};
                auto resultType = RankedTensorType::get(resultShape, lhsType.getElementType());
                
                std::cout << "result shape... " << lhsShape[0] << "  " << rhsShape[1] << "\n";

                mlir::Location loc = builder.getUnknownLoc();
                Value resultTensor = builder.create<tensor::EmptyOp>(loc, resultType.getShape(), elementType);


                // this is to keep track of correctness to avoid bugs
                std::cout << "in loop matmul op: \n";
                std::cout << "old id " << old_id << " with type " << resultTensor.getType().cast<ShapedType>().getShape()[0] << " and " << resultTensor.getType().cast<ShapedType>().getShape()[1] << "\n";
                std::cout << "old child1 id " << old_child1_id << " with type " << arg1_val.getType().cast<ShapedType>().getShape()[0] << " and " << arg1_val.getType().cast<ShapedType>().getShape()[1] << "\n";
                std::cout << "old child2 id " << old_child2_id << " with type " << arg2_val.getType().cast<ShapedType>().getShape()[0] << " and " << arg2_val.getType().cast<ShapedType>().getShape()[1] << "\n";
                
                mlir::Operation *newOp;
                newOp = builder.create<mlir::linalg::MatmulOp>(builder.getUnknownLoc(), std::vector<mlir::Value>{arg1_val, arg2_val}, std::vector<mlir::Value>{resultTensor});    

                mlir::Value new_result_val = newOp->getResult(0);

                // before checking the old result type and searching for an op to replace,
                // we check if this is a newly introduced op. 
                // If so, we don't need to replace or erase or check anything
                if (old_id > largest_id) {
                    // we update the result val in the map and move to the next iteration
                    // (since it can be used later on as an argument to some other op)
                    id_to_value[old_id] = new_result_val;
                    continue;
                }
                
                // here check the original dims of the old result. if it was f32, then transfor the current result
                // which is 1x1xf32 to f32
                auto& old_result_val = id_to_value.find(old_id)->second;
                auto newResultShape = new_result_val.getType().cast<ShapedType>().getShape();
                
                // we also do the check that the dims of the new result are 1x1xf32 to avoid any error
                if (old_result_val.getType().cast<ShapedType>().getShape().size() == 0
                && newResultShape.size() == 2 && newResultShape[0] == 1 && newResultShape[1] == 1){
                    // determine starting index of each dim of the slice
                    // in this case we only have one choice since the tensor is 1x1
                    SmallVector<OpFoldResult, 2> offsets = {builder.getIndexAttr(0), builder.getIndexAttr(0)};
                    // determine the number of elements to include in each dim of the slice
                    SmallVector<OpFoldResult, 2> sizes = {builder.getIndexAttr(1), builder.getIndexAttr(1)};
                    // step size in each dim
                    SmallVector<OpFoldResult, 2> strides = {builder.getIndexAttr(1), builder.getIndexAttr(1)};

                    // we determine the new type to be just a scalar like it was before
                    auto newType = RankedTensorType::get({}, new_result_val.getType().cast<ShapedType>().getElementType());
                    
                    // extract the wanted slice to reduce the dim
                    auto scalarTensor = builder.create<tensor::ExtractSliceOp>(
                        builder.getUnknownLoc(), newType, new_result_val, offsets, sizes, strides);
                    
                    // update newOp to point to the correct new operation
                    newOp = scalarTensor;

                    // Update newResult with the scalar tensor
                    new_result_val = scalarTensor;
                }


                // the issue that sometimes comes is caused by the code bellow
                // find the corresponding operation to replace
                bool found = false;

                for (mlir::linalg::LinalgOp op : filtered_ops) {
                    mlir::Operation *internalOperation = op.getOperation();
                    mlir::Value result;
                    if (internalOperation->getNumResults() != 0){
                        result = internalOperation->getResult(0);
                        for (const auto& pair : id_to_value) {
                            if (pair.second == result) {
                                int id = pair.first;
                                if (id == old_id){
                                    found = true;
                                    std::cout << "Found \n";
                                    break;
                                }
                            }
                        }
                        if (found){
                            op->replaceAllUsesWith(newOp);
                            op.erase();
                            std::cout << "erased the op \n";
                            break;
                        }
                    } else {
                        continue;
                    }
                    
                }

                id_to_value[old_id] = new_result_val;

            } else if (node->second.data == "linalg.dot") {
                std::cout << "in loop dot op: \n";
                // int old_id = node->second.old_id;
                int old_child1_id = returned.get_nodes().find(node->second.children[0])->second.old_id;
                int old_child2_id = returned.get_nodes().find(node->second.children[1])->second.old_id;
                
                auto& arg1_val = id_to_value.find(old_child1_id)->second;
                auto& arg2_val = id_to_value.find(old_child2_id)->second;

                // check if dims are nx1xf32. in this case cast it to nxf32. 
                // this is because dot only takes operands eith nxf32.
                // start with arg1
                auto arg1Shape = arg1_val.getType().cast<ShapedType>().getShape();
                if (arg1Shape.size() > 1 && arg1Shape[1] == 1){
                    
                    // determine starting index of each dim of the slice
                    SmallVector<OpFoldResult, 2> offsets = {builder.getIndexAttr(0), builder.getIndexAttr(0)};
                    // determine the number of elements to include in each dim of the slice
                    // we include every element in the first dimension. (n elements)
                    SmallVector<OpFoldResult, 2> sizes = {builder.getIndexAttr(arg1Shape[0]), builder.getIndexAttr(1)};

                    // step size in each dim
                    SmallVector<OpFoldResult, 2> strides = {builder.getIndexAttr(1), builder.getIndexAttr(1)};

                    // we determine the new type to be a vector with the corresponding size
                    auto newType = RankedTensorType::get({arg1Shape[0]}, arg1_val.getType().cast<ShapedType>().getElementType());
                    
                    std::cout << "The size of the vector for arg1 is: " << arg1Shape[0] << "\n";

                    // extract the wanted slice to reduce the dim by 1
                    auto reducedTensor = builder.create<tensor::ExtractSliceOp>(
                        builder.getUnknownLoc(), newType, arg1_val, offsets, sizes, strides);
                    
                    // Update arg1_val with the new tensor
                    arg1_val = reducedTensor.getResult();
                }

                // the arg2
                auto arg2Shape = arg2_val.getType().cast<ShapedType>().getShape();
                if (arg2Shape.size() > 1 && arg2Shape[1] == 1){
                    
                    // determine starting index of each dim of the slice
                    SmallVector<OpFoldResult, 2> offsets = {builder.getIndexAttr(0), builder.getIndexAttr(0)};
                    // determine the number of elements to include in each dim of the slice
                    // we include every element in the first dimension like we did for arg1. (n elements)
                    SmallVector<OpFoldResult, 2> sizes = {builder.getIndexAttr(arg2Shape[0]), builder.getIndexAttr(1)};

                    // step size in each dim
                    SmallVector<OpFoldResult, 2> strides = {builder.getIndexAttr(1), builder.getIndexAttr(1)};

                    // we determine the new type to be a vector with the corresponding size
                    auto newType = RankedTensorType::get({arg2Shape[0]}, arg2_val.getType().cast<ShapedType>().getElementType());
                    
                    std::cout << "The size of the vector for arg2 is: " << arg1Shape[0] << "\n";
                    
                    // extract the wanted slice to reduce the dim by 1
                    auto reducedTensor = builder.create<tensor::ExtractSliceOp>( builder.getUnknownLoc(), newType, arg2_val, offsets, sizes, strides);
                    
                    // Update arg1_val with the new tensor
                    arg2_val = reducedTensor.getResult();
                }

                // the result in dot is always a scalar
                // we initialize a tensor for the result
                auto lhsType = arg1_val.getType().cast<ShapedType>();
                Type elementType = lhsType.getElementType();
                auto resultType = RankedTensorType::get({}, lhsType.getElementType());
                Value resultTensor = builder.create<tensor::EmptyOp>(builder.getUnknownLoc(), resultType.getShape(), elementType);

                
                mlir::Operation *newOp;
                newOp = builder.create<mlir::linalg::DotOp>(builder.getUnknownLoc(), std::vector<mlir::Value>{arg1_val, arg2_val}, std::vector<mlir::Value>{resultTensor});    


                mlir::Value new_result_val = newOp->getResult(0);

                // similar to matmul
                // we check if this is a newly introduced op. 
                if (old_id > largest_id) {
                    id_to_value[old_id] = new_result_val;
                    continue;
                }

                // now check the original dims of the old result. if it was 1x1xf32, 
                // then transfor the current result which is f32 to 1x1xf32
                // else, we leave it since it's of the correct type

                auto& old_result_val = id_to_value.find(old_id)->second;
                auto newResultShape = new_result_val.getType().cast<ShapedType>().getShape();
                auto oldResultShape = old_result_val.getType().cast<ShapedType>().getShape();
                
                if (oldResultShape.size() == 2 && oldResultShape[0] == 1 && oldResultShape[1] == 1){

                    std::cout << "Converting in dot section shape with dim: " << newResultShape.size() << " to the old dim: " << oldResultShape.size() <<"\n";
                    
                    // we define the shape and type to be 1x1xf32
                    SmallVector<int64_t, 2> newShape = {1, 1};
                    auto newType = RankedTensorType::get(newShape, new_result_val.getType().cast<ShapedType>().getElementType());

                    // Create a constant op for the new shape
                    // this is required for building the operation tensor.reshape
                    // we need to define a shape variable
                    auto shapeAttr = DenseIntElementsAttr::get(RankedTensorType::get({2}, builder.getI64Type()), newShape);
                    auto shapeValue = builder.create<mlir::arith::ConstantOp>(builder.getUnknownLoc(), shapeAttr);

                    // Create the reshape op to hold the result in the correct type
                    auto expandedTensor = builder.create<tensor::ReshapeOp>(builder.getUnknownLoc(), newType, new_result_val, shapeValue);
                    
                    // update newOp to point to the correct new operation 
                    newOp = expandedTensor;

                    // Update newResult with the scalar tensor
                    new_result_val = expandedTensor;
                }

                // find the corresponding operation to replace
                bool found = false;

                for (mlir::linalg::LinalgOp op : filtered_ops) {
                    mlir::Operation *internalOperation = op.getOperation();
                    mlir::Value result;
                    if (internalOperation->getNumResults() != 0){
                        result = internalOperation->getResult(0);
                    } else {
                        continue;
                    }
                    
                    for (const auto& pair : id_to_value) {
                        if (pair.second == result) {
                            int id = pair.first;
                            if (id == old_id){
                                found = true;
                                break;
                            }
                        }
                    }
                    if (found){
                        op->replaceAllUsesWith(newOp);
	                    op.erase();
                        break;
                    }
                    
                }

                id_to_value[old_id] = new_result_val;


            } else if (node->second.data == "linalg.add") {
                std::cout << "writing add op... \n";
                int old_child1_id = returned.get_nodes().find(node->second.children[0])->second.old_id;
                int old_child2_id = returned.get_nodes().find(node->second.children[1])->second.old_id;
                
                auto& arg1_val = id_to_value.find(old_child1_id)->second;
                auto& arg2_val = id_to_value.find(old_child2_id)->second;

                // we prepare the tensor for the result with the correct dims
                // the result has the same dims as the operands
                auto lhsType = arg1_val.getType().cast<ShapedType>();
                auto lhsShape = lhsType.getShape();
                Type elementType = lhsType.getElementType();
                
                SmallVector<int64_t, 2> resultShape = {lhsShape[0], lhsShape[1]};

                auto resultType = RankedTensorType::get(resultShape, lhsType.getElementType());

                mlir::Location loc = builder.getUnknownLoc();
                
                std::cout << "result shape... " << lhsShape[0] << "  " << lhsShape[1] << "\n";
                // here same issue as in matmul
                // dot too probably
                // Ensure resultType and elementType are properly initialized
                if (!resultType) {
                    llvm::errs() << "Result type is null\n";
                    return;
                }
                if (!elementType) {
                    llvm::errs() << "Element type is null\n";
                    return;
                }

                
                Value resultTensor = builder.create<tensor::EmptyOp>(loc, resultType.getShape(), elementType);


                // this is to keep track of correctness to avoid bugs
                std::cout << "in loop add op: \n";
                std::cout << "old id " << old_id << " with type " << resultTensor.getType().cast<ShapedType>().getShape()[0] << " and " << resultTensor.getType().cast<ShapedType>().getShape()[1] << "\n";
                std::cout << "old child1 id " << old_child1_id << " with type " << arg1_val.getType().cast<ShapedType>().getShape()[0] << " and " << arg1_val.getType().cast<ShapedType>().getShape()[1] << "\n";
                std::cout << "old child2 id " << old_child2_id << " with type " << arg2_val.getType().cast<ShapedType>().getShape()[0] << " and " << arg2_val.getType().cast<ShapedType>().getShape()[1] << "\n";
                
                mlir::Operation *newOp;
                newOp = builder.create<mlir::linalg::AddOp>(builder.getUnknownLoc(), std::vector<mlir::Value>{arg1_val, arg2_val}, std::vector<mlir::Value>{resultTensor});    

                mlir::Value new_result_val = newOp->getResult(0);

                // similar to matmul
                // we check if this is a newly introduced op. 
                if (old_id > largest_id) {
                    id_to_value[old_id] = new_result_val;
                    continue;
                }

                // find the corresponding operation to replace
                bool found = false;

                for (mlir::linalg::LinalgOp op : filtered_ops) {
                    
                    mlir::Operation *internalOperation = op.getOperation();
                    mlir::Value result;
                    if (internalOperation->getNumResults() != 0){
                        result = internalOperation->getResult(0);
                            for (const auto& pair : id_to_value) {
                            if (pair.second == result) {
                                int id = pair.first;
                                if (id == old_id){
                                    found = true;
                                    break;
                                }
                            }
                        }
                        if (found){
                            op->replaceAllUsesWith(newOp);
                            op.erase();
                            break;
                        }
                    } else {
                        continue;
                    }
                    
                }

                id_to_value[old_id] = new_result_val;
            }
            
            // no need to check the operands dims like in matmul and dot for now for add op
            // if need ever arises, follow template used in matmul and dot

            // another approch would be to encode these information in the egg mlir language
            // this way, we handle these issues in equality saturation instead of here
            // for now both seem equivilent so I'm keeping this method
            // Note that it wouldn't be general and wouldn't spare us the need to do this for new ops
            // this is because we would need to encode it in our egg mlir language too for every new op
            // I find it easier the do it here this way
            

            // maybe a good thing to do if possible is loop through the old filtered ops and 
            // check the number of uses they have. If there are none then erase the op
            // There is no need to do this after consideration
            // after this pass, we will always run the generic optimisations such as code elimination
        }


        

    }
}

bool is_number(std::string str) {
    for (int i = 0; i < str.length(); i++){
        if (isdigit(str[i]) == false){
            return false;
        }
    }
      return true;
}

// actually register the pass
namespace mlir {
void registerEqualitySaturationPass() {
    mlir::PassRegistration<EqualitySaturationPass>();
}
} // namespace mlir
